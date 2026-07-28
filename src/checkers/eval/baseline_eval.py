"""Versioned, replay-complete baseline evaluation and gate reporting."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import yaml

from checkers.agents.greedy_agent import GreedyAgent
from checkers.agents.minimax_agent import MinimaxAgent
from checkers.agents.random_agent import RandomAgent
from checkers.eval.arena import STREAMS_PER_GAME, AgentSpec, GameRecord, MatchResult
from checkers.eval.elo import EloEstimate, elo_estimate
from checkers.eval.population import (
    PayoffMatrix,
    fixed_anchor_scores,
    league_elo,
    three_cycles,
)
from checkers.eval.power import MatchScore, PowerPlan, plan_score_test
from checkers.eval.suites import (
    TacticalCase,
    TacticalEvaluation,
    compare_tactical,
    evaluate_tactical,
    load_dev_tactical_suite,
)
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome, TerminationReason

BASELINE_AGENTS = ("random", "greedy", "minimax(1)", "minimax(2)")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
UINT64_MAX = (1 << 64) - 1
TACTICAL_SUBSTANTIAL_GAIN = 5
INVERSION_FLOOR = 0.40
PAIR_SIZE = 2
MIN_TACTICAL_DEPTHS = 2
NEUTRAL_SCORE = 0.5
CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "experiment_label",
        "seed",
        "games_per_match",
        "confidence",
        "smallest_effect",
        "null_score",
        "alpha",
        "target_power",
        "max_plies",
        "repetition_draws",
        "agents",
        "comparisons",
        "tactical_depths",
    }
)


def _mapping(value: object, field_name: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(dict[object, object], value)


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return cast(list[object], value)


def _required(mapping: dict[object, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"missing required field {key!r}")
    return mapping[key]


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    checked = _integer(value, field_name)
    if checked < 1:
        raise ValueError(f"{field_name} must be positive")
    return checked


def _seed(value: object, field_name: str = "seed") -> int:
    checked = _integer(value, field_name)
    if not 0 <= checked <= UINT64_MAX:
        raise ValueError(f"{field_name} must be an unsigned 64-bit integer")
    return checked


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{field_name} must be finite")
    return checked


def _open_probability(value: object, field_name: str) -> float:
    checked = _number(value, field_name)
    if not 0.0 < checked < 1.0:
        raise ValueError(f"{field_name} must be strictly between zero and one")
    return checked


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _sha256(value: object, field_name: str) -> str:
    checked = _string(value, field_name)
    if SHA256_PATTERN.fullmatch(checked) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return checked


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    items = _list(value, field_name)
    return tuple(_string(item, field_name) for item in items)


def _parse_comparison(value: object) -> tuple[str, str]:
    items = _list(value, "comparison")
    if len(items) != PAIR_SIZE:
        raise ValueError("comparisons must contain exactly two agents")
    return (
        _string(items[0], "comparison agent"),
        _string(items[1], "comparison agent"),
    )


def _validate_stream_capacity(
    *,
    root_seed: int,
    games_per_match: int,
    comparison_count: int,
) -> None:
    required_streams = games_per_match * comparison_count * STREAMS_PER_GAME
    if root_seed > UINT64_MAX - required_streams:
        raise ValueError("seed leaves insufficient uint64 space for disjoint comparison streams")


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Frozen powered round-robin and tactical evaluation contract."""

    schema_version: int
    experiment_id: str
    experiment_label: str
    seed: int
    games_per_match: int
    confidence: float
    smallest_effect: float
    null_score: float
    alpha: float
    target_power: float
    max_plies: int
    repetition_draws: bool
    agents: tuple[str, ...]
    comparisons: tuple[tuple[str, str], ...]
    tactical_depths: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _string(self.experiment_id, "experiment_id")
        if self.experiment_label != "baseline":
            raise ValueError("experiment_label must be 'baseline'")
        root_seed = _seed(self.seed)
        games_per_match = _positive_integer(self.games_per_match, "games_per_match")
        _positive_integer(self.max_plies, "max_plies")
        _boolean(self.repetition_draws, "repetition_draws")
        _open_probability(self.confidence, "confidence")
        effect = _open_probability(self.smallest_effect, "smallest_effect")
        null = _open_probability(self.null_score, "null_score")
        if null + effect >= 1.0:
            raise ValueError("null_score plus smallest_effect must be below one")
        _open_probability(self.alpha, "alpha")
        _open_probability(self.target_power, "target_power")
        if self.agents != BASELINE_AGENTS:
            raise ValueError("agents must equal the frozen baseline-agent order")
        if not isinstance(self.comparisons, tuple):
            raise TypeError("comparisons must be a tuple")
        expected = {frozenset(pair) for pair in combinations(BASELINE_AGENTS, 2)}
        observed: set[frozenset[str]] = set()
        for pair in self.comparisons:
            if not isinstance(pair, tuple) or len(pair) != PAIR_SIZE:
                raise TypeError("comparisons must contain two-name tuples")
            first, second = pair
            _string(first, "comparison agent")
            _string(second, "comparison agent")
            if first == second or first not in BASELINE_AGENTS or second not in BASELINE_AGENTS:
                raise ValueError("comparisons must use two distinct baseline agents")
            observed.add(frozenset(pair))
        if observed != expected or len(self.comparisons) != len(expected):
            raise ValueError("comparisons must contain every unordered pair exactly once")
        _validate_stream_capacity(
            root_seed=root_seed,
            games_per_match=games_per_match,
            comparison_count=len(self.comparisons),
        )
        if (
            not isinstance(self.tactical_depths, tuple)
            or len(self.tactical_depths) < MIN_TACTICAL_DEPTHS
            or len(set(self.tactical_depths)) != len(self.tactical_depths)
            or tuple(sorted(self.tactical_depths)) != self.tactical_depths
        ):
            raise ValueError(
                "tactical_depths must be a sorted unique tuple with at least two depths"
            )
        for depth in self.tactical_depths:
            _positive_integer(depth, "tactical depth")
        if self.games_per_match != self.power_plan.balanced_games:
            raise ValueError("games_per_match must equal the colour-balanced power plan")

    @property
    def power_plan(self) -> PowerPlan:
        """Return the predeclared normal-approximation power calculation.

        Returns:
            Validated raw and colour-balanced game counts plus achieved power.
        """

        return plan_score_test(
            null_score=self.null_score,
            alternative_score=self.null_score + self.smallest_effect,
            alpha=self.alpha,
            target_power=self.target_power,
        )


def load_baseline_config(text: str) -> BaselineConfig:
    """Parse and fully validate one baseline YAML document.

    Args:
        text: Complete YAML text with no unknown fields.

    Returns:
        Immutable, power-consistent experiment configuration.

    Raises:
        TypeError: If the document or a field has an invalid runtime type.
        ValueError: If a required field is absent or violates the frozen schema.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    loaded: object = yaml.safe_load(text)
    root = _mapping(loaded, "baseline config")
    schema_version = _integer(_required(root, "schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be 1")
    unknown_fields = set(root) - CONFIG_FIELDS
    if unknown_fields:
        names = ", ".join(sorted(str(field) for field in unknown_fields))
        raise ValueError(f"baseline config contains unknown fields: {names}")
    agents = _string_list(_required(root, "agents"), "agents")
    typed_comparisons = tuple(
        _parse_comparison(pair) for pair in _list(_required(root, "comparisons"), "comparisons")
    )
    tactical_depths = tuple(
        _positive_integer(item, "tactical depth")
        for item in _list(_required(root, "tactical_depths"), "tactical_depths")
    )
    return BaselineConfig(
        schema_version=schema_version,
        experiment_id=_string(_required(root, "experiment_id"), "experiment_id"),
        experiment_label=_string(_required(root, "experiment_label"), "experiment_label"),
        seed=_seed(_required(root, "seed")),
        games_per_match=_positive_integer(_required(root, "games_per_match"), "games_per_match"),
        confidence=_open_probability(_required(root, "confidence"), "confidence"),
        smallest_effect=_open_probability(_required(root, "smallest_effect"), "smallest_effect"),
        null_score=_open_probability(_required(root, "null_score"), "null_score"),
        alpha=_open_probability(_required(root, "alpha"), "alpha"),
        target_power=_open_probability(_required(root, "target_power"), "target_power"),
        max_plies=_positive_integer(_required(root, "max_plies"), "max_plies"),
        repetition_draws=_boolean(_required(root, "repetition_draws"), "repetition_draws"),
        agents=agents,
        comparisons=typed_comparisons,
        tactical_depths=tactical_depths,
    )


def agent_spec(name: str, *, max_plies: int) -> AgentSpec:
    """Build one frozen baseline policy specification.

    Args:
        name: Exact frozen policy name.
        max_plies: Positive search terminal boundary.

    Returns:
        Deterministic seeded factory with the declared policy name.

    Raises:
        TypeError: If inputs have invalid runtime types.
        ValueError: If the name is unsupported or ``max_plies`` is invalid.
    """

    checked_name = _string(name, "name")
    checked_max_plies = _positive_integer(max_plies, "max_plies")
    if checked_name == "random":
        return AgentSpec(name=checked_name, factory=lambda seed: RandomAgent(seed=seed))
    if checked_name == "greedy":
        return AgentSpec(name=checked_name, factory=lambda seed: GreedyAgent(seed=seed))
    if checked_name == "minimax(1)":
        return AgentSpec(
            name=checked_name,
            factory=lambda seed: MinimaxAgent(depth=1, seed=seed, max_plies=checked_max_plies),
        )
    if checked_name == "minimax(2)":
        return AgentSpec(
            name=checked_name,
            factory=lambda seed: MinimaxAgent(depth=2, seed=seed, max_plies=checked_max_plies),
        )
    raise ValueError(f"unknown baseline agent {checked_name!r}")


def _state_record(state: State) -> dict[str, object]:
    return {
        "men": list(state.men),
        "kings": list(state.kings),
        "side_to_move": int(state.side_to_move),
        "capture_in_progress": state.capture_in_progress,
        "moving_square": state.moving_square,
        "sequence_origin": state.sequence_origin,
        "captured_pending": state.captured_pending,
        "no_progress": list(state.no_progress),
        "ply": state.ply,
    }


def _parse_integer_pair(value: object, field_name: str) -> tuple[int, int]:
    items = _list(value, field_name)
    if len(items) != PAIR_SIZE:
        raise ValueError(f"{field_name} must contain exactly two integers")
    return (_integer(items[0], field_name), _integer(items[1], field_name))


def _optional_integer(value: object, field_name: str) -> int | None:
    return None if value is None else _integer(value, field_name)


def _parse_state(value: object) -> State:
    root = _mapping(value, "state")
    return State(
        men=_parse_integer_pair(_required(root, "men"), "men"),
        kings=_parse_integer_pair(_required(root, "kings"), "kings"),
        side_to_move=PlayerId(_integer(_required(root, "side_to_move"), "side_to_move")),
        capture_in_progress=_boolean(_required(root, "capture_in_progress"), "capture_in_progress"),
        moving_square=_optional_integer(_required(root, "moving_square"), "moving_square"),
        sequence_origin=_optional_integer(_required(root, "sequence_origin"), "sequence_origin"),
        captured_pending=_integer(_required(root, "captured_pending"), "captured_pending"),
        no_progress=_parse_integer_pair(_required(root, "no_progress"), "no_progress"),
        ply=_integer(_required(root, "ply"), "ply"),
    )


def _score_record(score: MatchScore) -> dict[str, object]:
    return {
        "wins": score.wins,
        "draws": score.draws,
        "losses": score.losses,
        "score": score.score,
        "low": score.low,
        "high": score.high,
        "confidence": score.confidence,
    }


def _parse_score(value: object) -> MatchScore:
    root = _mapping(value, "score")
    return MatchScore(
        wins=_integer(_required(root, "wins"), "wins"),
        draws=_integer(_required(root, "draws"), "draws"),
        losses=_integer(_required(root, "losses"), "losses"),
        score=_number(_required(root, "score"), "score"),
        low=_number(_required(root, "low"), "low"),
        high=_number(_required(root, "high"), "high"),
        confidence=_open_probability(_required(root, "confidence"), "confidence"),
    )


def _game_record(record: GameRecord) -> dict[str, object]:
    return {
        "red_agent": record.red_agent,
        "white_agent": record.white_agent,
        "red_seed": record.red_seed,
        "white_seed": record.white_seed,
        "environment_seed": record.environment_seed,
        "outcome": {
            "winner": None if record.outcome.winner is None else int(record.outcome.winner),
            "reason": record.outcome.reason.value,
        },
        "actions": list(record.actions),
        "moves": list(record.moves),
    }


def match_result_record(match: MatchResult) -> dict[str, object]:
    """Serialize every field required to validate and replay a match.

    Args:
        match: Validated colour-balanced match.

    Returns:
        JSON-compatible replay record including every seed, action, and outcome.

    Raises:
        TypeError: If ``match`` is not a ``MatchResult``.
    """

    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    return {
        "first_agent": match.first_agent,
        "second_agent": match.second_agent,
        "seed": match.seed,
        "initial_state": _state_record(match.initial_state),
        "max_plies": match.max_plies,
        "repetition_draws": match.repetition_draws,
        "confidence": match.confidence,
        "score": _score_record(match.score),
        "games": [_game_record(record) for record in match.records],
    }


def _parse_game(value: object) -> GameRecord:
    root = _mapping(value, "game")
    outcome_root = _mapping(_required(root, "outcome"), "outcome")
    winner_value = _required(outcome_root, "winner")
    winner = None if winner_value is None else PlayerId(_integer(winner_value, "winner"))
    return GameRecord(
        red_agent=_string(_required(root, "red_agent"), "red_agent"),
        white_agent=_string(_required(root, "white_agent"), "white_agent"),
        red_seed=_seed(_required(root, "red_seed"), "red_seed"),
        white_seed=_seed(_required(root, "white_seed"), "white_seed"),
        environment_seed=_seed(_required(root, "environment_seed"), "environment_seed"),
        outcome=Outcome(
            winner=winner,
            reason=TerminationReason(_string(_required(outcome_root, "reason"), "reason")),
        ),
        actions=tuple(
            _integer(action, "action") for action in _list(_required(root, "actions"), "actions")
        ),
        moves=tuple(_string(move, "move") for move in _list(_required(root, "moves"), "moves")),
    )


def parse_match_result_record(value: object) -> MatchResult:
    """Parse a replay-complete match and re-run all ``MatchResult`` invariants.

    Args:
        value: Untrusted JSON-compatible match record.

    Returns:
        Fully reconstructed and schedule-validated match.

    Raises:
        TypeError: If any field has an invalid runtime type.
        ValueError: If content, outcome counts, or the seed schedule is inconsistent.
    """

    root = _mapping(value, "match record")
    return MatchResult(
        first_agent=_string(_required(root, "first_agent"), "first_agent"),
        second_agent=_string(_required(root, "second_agent"), "second_agent"),
        seed=_seed(_required(root, "seed")),
        initial_state=_parse_state(_required(root, "initial_state")),
        max_plies=_positive_integer(_required(root, "max_plies"), "max_plies"),
        repetition_draws=_boolean(_required(root, "repetition_draws"), "repetition_draws"),
        confidence=_open_probability(_required(root, "confidence"), "confidence"),
        records=tuple(_parse_game(game) for game in _list(_required(root, "games"), "games")),
        score=_parse_score(_required(root, "score")),
    )


def _validate_termination_counts(
    values: object,
    *,
    games: int,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError("termination_counts must be a tuple")
    seen: set[str] = set()
    total = 0
    for item in values:
        if not isinstance(item, tuple) or len(item) != PAIR_SIZE:
            raise TypeError("termination_counts must contain name/count tuples")
        reason = _string(item[0], "termination reason")
        if reason in seen:
            raise ValueError("termination_counts must contain unique reasons")
        seen.add(reason)
        count = _integer(item[1], "termination count")
        if count < 0:
            raise ValueError("termination count must be non-negative")
        total += count
    if total != games:
        raise ValueError("termination counts must sum to games")


@dataclass(frozen=True, slots=True)
class BaselineMatchSummary:
    """Compact point/interval, colour, terminal, timing, and identity evidence."""

    first_agent: str
    second_agent: str
    seed: int
    games: int
    first_as_red_games: int
    score: MatchScore
    elo: EloEstimate
    elapsed_seconds: float
    mean_steps: float
    mean_completed_moves: float
    termination_counts: tuple[tuple[str, int], ...]
    records_sha256: str

    def __post_init__(self) -> None:
        first = _string(self.first_agent, "first_agent")
        second = _string(self.second_agent, "second_agent")
        if first == second:
            raise ValueError("summary agents must be distinct")
        _seed(self.seed)
        games = _positive_integer(self.games, "games")
        if games % 2:
            raise ValueError("summary games must be even")
        if self.first_as_red_games * 2 != games:
            raise ValueError("summary colour schedule must be exactly balanced")
        if not isinstance(self.score, MatchScore):
            raise TypeError("score must be a MatchScore")
        if self.score.games != games:
            raise ValueError("summary score game count must equal games")
        if not isinstance(self.elo, EloEstimate):
            raise TypeError("elo must be an EloEstimate")
        for value, field_name in (
            (self.elapsed_seconds, "elapsed_seconds"),
            (self.mean_steps, "mean_steps"),
            (self.mean_completed_moves, "mean_completed_moves"),
        ):
            if _number(value, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        _validate_termination_counts(self.termination_counts, games=games)
        _sha256(self.records_sha256, "records_sha256")


def summarize_match(match: MatchResult, *, elapsed_seconds: float) -> BaselineMatchSummary:
    """Summarize one complete match without discarding its replay identity.

    Args:
        match: Replay-complete arena result.
        elapsed_seconds: Non-negative measured wall time.

    Returns:
        Immutable score/Elo/terminal/timing summary bound to a records digest.

    Raises:
        TypeError: If an input has an invalid runtime type.
        ValueError: If elapsed time is negative or non-finite.
    """

    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    elapsed = _number(elapsed_seconds, "elapsed_seconds")
    if elapsed < 0.0:
        raise ValueError("elapsed_seconds must be non-negative")
    counts = tuple(
        (reason.value, sum(record.outcome.reason is reason for record in match.records))
        for reason in TerminationReason
        if any(record.outcome.reason is reason for record in match.records)
    )
    records = [_game_record(record) for record in match.records]
    return BaselineMatchSummary(
        first_agent=match.first_agent,
        second_agent=match.second_agent,
        seed=match.seed,
        games=match.games,
        first_as_red_games=match.first_as_red_games,
        score=match.score,
        elo=elo_estimate(match.score),
        elapsed_seconds=elapsed,
        mean_steps=sum(record.steps for record in match.records) / match.games,
        mean_completed_moves=sum(record.completed_moves for record in match.records) / match.games,
        termination_counts=counts,
        records_sha256=_canonical_sha256(records),
    )


def _finite_or_label(value: float) -> float | str:
    if value == math.inf:
        return "+Infinity"
    if value == -math.inf:
        return "-Infinity"
    return value


def _elo_record(estimate: EloEstimate) -> dict[str, object]:
    return {
        "difference": _finite_or_label(estimate.difference),
        "low": _finite_or_label(estimate.low),
        "high": _finite_or_label(estimate.high),
        "confidence": estimate.confidence,
    }


def _summary_record(summary: BaselineMatchSummary) -> dict[str, object]:
    return {
        "first_agent": summary.first_agent,
        "second_agent": summary.second_agent,
        "seed": summary.seed,
        "games": summary.games,
        "first_as_red_games": summary.first_as_red_games,
        "score": _score_record(summary.score),
        "elo": _elo_record(summary.elo),
        "elapsed_seconds": summary.elapsed_seconds,
        "mean_steps": summary.mean_steps,
        "mean_completed_moves": summary.mean_completed_moves,
        "termination_counts": [
            {"reason": reason, "count": count} for reason, count in summary.termination_counts
        ],
        "records_sha256": summary.records_sha256,
    }


def _wdl_record(wins: int, draws: int, losses: int) -> dict[str, object]:
    games = wins + draws + losses
    score = NEUTRAL_SCORE if games == 0 else (wins + NEUTRAL_SCORE * draws) / games
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": games,
        "score": score,
    }


def _validated_matches(matches: object) -> tuple[MatchResult, ...]:
    if not isinstance(matches, tuple):
        raise TypeError("matches must be a tuple")
    if not all(isinstance(match, MatchResult) for match in matches):
        raise TypeError("matches must contain MatchResult values")
    return cast(tuple[MatchResult, ...], matches)


def build_population_report(matches: tuple[MatchResult, ...]) -> dict[str, object]:
    """Build the full W/D/L population view without hiding non-transitivity.

    Args:
        matches: Exactly one result for every frozen unordered policy pair.

    Returns:
        Full matrix, league projection, residuals, cycles, and fixed anchors.

    Raises:
        TypeError: If the match container has invalid runtime types.
        ValueError: If the population is incomplete or duplicated.
    """

    checked_matches = _validated_matches(matches)
    matrix = PayoffMatrix.from_matches(agents=BASELINE_AGENTS, matches=checked_matches)
    league = league_elo(matrix)
    cycles = three_cycles(matrix)
    anchors = fixed_anchor_scores(
        matrix,
        candidates=("minimax(2)",),
        anchors=("random", "greedy", "minimax(1)"),
    )
    return {
        "agents": list(matrix.agents),
        "pairs": [
            {
                "first_agent": pair.first,
                "second_agent": pair.second,
                **_wdl_record(pair.wdl.wins, pair.wdl.draws, pair.wdl.losses),
            }
            for pair in matrix.pairs
        ],
        "wdl_matrix": [
            [_wdl_record(cell.wins, cell.draws, cell.losses) for cell in row]
            for row in matrix.rows()
        ],
        "league_elo": {
            "ratings": [
                {
                    "agent": rating.agent,
                    "rating": rating.rating,
                    "low": rating.low,
                    "high": rating.high,
                }
                for rating in league.ratings
            ],
            "confidence": league.confidence,
            "ci_assumption": league.ci_assumption,
            "transitivity_threshold_elo": league.transitivity_threshold,
            "max_abs_residual_elo": league.max_abs_residual,
            "weighted_rmse_elo": league.weighted_rmse,
            "approximately_transitive": league.approximately_transitive,
        },
        "three_cycles": {
            "count": cycles.count,
            "total_magnitude": cycles.total_magnitude,
            "max_magnitude": cycles.max_magnitude,
            "cycles": [
                {
                    "agents": list(cycle.agents),
                    "edge_margins": list(cycle.edge_margins),
                    "magnitude": cycle.magnitude,
                }
                for cycle in cycles.cycles
            ],
        },
        "fixed_anchor_scores": [
            {
                "candidate": anchor.candidate,
                "anchor": anchor.anchor,
                **_wdl_record(anchor.wdl.wins, anchor.wdl.draws, anchor.wdl.losses),
            }
            for anchor in anchors
        ],
        "exploitability_proxy": {
            "status": "NOT_EVALUATED",
            "reason": (
                "A trained short-budget best response does not exist before Phase 7; "
                "substituting a hand-coded baseline would not measure the declared proxy."
            ),
            "scheduled_phase": 7,
        },
        "external_anchor": {
            "status": "NOT_AVAILABLE",
            "reason": "No permitted offline external engine was available for this baseline run.",
        },
    }


def _evaluate_minimax_tactical(
    *,
    depth: int,
    seed: int,
    cases: tuple[TacticalCase, ...],
) -> TacticalEvaluation:
    checked_depth = _positive_integer(depth, "depth")
    checked_seed = _seed(seed)
    if not isinstance(cases, tuple) or not all(isinstance(case, TacticalCase) for case in cases):
        raise TypeError("cases must be a tuple of TacticalCase values")
    if not cases:
        raise ValueError("cases must not be empty")
    results = tuple(
        evaluate_tactical(
            MinimaxAgent(depth=checked_depth, seed=checked_seed),
            (case,),
        ).results[0]
        for case in cases
    )
    return TacticalEvaluation.from_case_results(
        agent_name=f"minimax({checked_depth})",
        results=results,
    )


def build_tactical_report(*, seed: int, depths: tuple[int, ...]) -> dict[str, object]:
    """Evaluate declared minimax depths on the independently solved dev suite.

    Args:
        seed: Shared deterministic tie-break seed.
        depths: Sorted unique completed-move search depths.

    Returns:
        Per-case selections and the shallow-versus-deep gate comparison.

    Raises:
        TypeError: If a seed or depth has an invalid runtime type.
        ValueError: If depths are missing, duplicated, unsorted, or non-positive.
    """

    checked_seed = _seed(seed)
    if not isinstance(depths, tuple) or len(depths) < MIN_TACTICAL_DEPTHS:
        raise ValueError("depths must be a tuple containing at least two depths")
    if len(set(depths)) != len(depths) or tuple(sorted(depths)) != depths:
        raise ValueError("depths must be sorted and unique")
    checked_depths = tuple(_positive_integer(depth, "depth") for depth in depths)
    suite = load_dev_tactical_suite()
    evaluations = tuple(
        _evaluate_minimax_tactical(
            depth=depth,
            seed=checked_seed,
            cases=suite.cases,
        )
        for depth in checked_depths
    )
    comparison = compare_tactical(
        evaluations[0],
        evaluations[-1],
        substantial_gain=TACTICAL_SUBSTANTIAL_GAIN,
    )
    evaluation_records = {
        evaluation.agent_name: {
            "solved": evaluation.solved,
            "total": evaluation.total,
            "accuracy": evaluation.accuracy,
            "results": [
                {
                    "case_id": result.case_id,
                    "selected_action": result.selected_action,
                    "solved": result.solved,
                }
                for result in evaluation.results
            ],
        }
        for evaluation in evaluations
    }
    comparison_record = {
        "shallow_agent": comparison.shallow_agent,
        "deep_agent": comparison.deep_agent,
        "gained_case_ids": list(comparison.gained_case_ids),
        "lost_case_ids": list(comparison.lost_case_ids),
        "gained": comparison.gained,
        "net_gain": comparison.net_gain,
        "substantial_gain_threshold": comparison.substantial_gain,
        "deep_is_superset": comparison.deep_is_superset,
        "substantially_more": comparison.substantially_more,
        "passes_gate": comparison.passes_gate,
    }
    return {
        "suite_name": suite.manifest.name,
        "suite_case_count": len(suite.cases),
        "suite_split": suite.manifest.split,
        "cases_sha256": suite.manifest.cases_sha256,
        "generator_source_sha256": suite.manifest.generator_source_sha256,
        "rules_source_sha256": suite.manifest.rules_source_sha256,
        "goal_sha256": suite.manifest.goal_sha256,
        "oracle": "independent terminal-only exhaustive AND/OR forced-win solver",
        "seed": checked_seed,
        "case_evaluation_seed_policy": "fresh policy with the declared seed per case",
        "depths": list(checked_depths),
        "evaluations": evaluation_records,
        f"depth_{checked_depths[0]}_vs_{checked_depths[-1]}": comparison_record,
    }


def _power_record(plan: PowerPlan) -> dict[str, object]:
    return {
        "null_score": plan.null_score,
        "alternative_score": plan.alternative_score,
        "smallest_effect": abs(plan.alternative_score - plan.null_score),
        "alpha_two_sided": plan.alpha,
        "target_power": plan.target_power,
        "raw_games": plan.raw_games,
        "balanced_games": plan.balanced_games,
        "achieved_approximate_power": plan.achieved_power,
        "calculation": "normal approximation, conservatively treating bounded scores as Bernoulli",
    }


def _config_record(config: BaselineConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "experiment_label": config.experiment_label,
        "seed": config.seed,
        "games_per_match": config.games_per_match,
        "confidence": config.confidence,
        "smallest_effect": config.smallest_effect,
        "null_score": config.null_score,
        "alpha": config.alpha,
        "target_power": config.target_power,
        "max_plies": config.max_plies,
        "repetition_draws": config.repetition_draws,
        "agents": list(config.agents),
        "comparisons": [list(pair) for pair in config.comparisons],
        "comparison_seed_stride": config.games_per_match * STREAMS_PER_GAME,
        "seed_schedule": "disjoint contiguous SplitMix64 input block per comparison",
        "tactical_depths": list(config.tactical_depths),
    }


def _validate_summaries(
    matches: tuple[MatchResult, ...],
    summaries: object,
) -> tuple[BaselineMatchSummary, ...]:
    if not isinstance(summaries, tuple) or not all(
        isinstance(summary, BaselineMatchSummary) for summary in summaries
    ):
        raise TypeError("summaries must be a tuple of BaselineMatchSummary values")
    checked = cast(tuple[BaselineMatchSummary, ...], summaries)
    if len(checked) != len(matches):
        raise ValueError("summaries must correspond one-to-one with matches")
    for match, summary in zip(matches, checked, strict=True):
        if (
            summary.first_agent != match.first_agent
            or summary.second_agent != match.second_agent
            or summary.seed != match.seed
            or summary.games != match.games
            or summary.score != match.score
            or summary.records_sha256
            != _canonical_sha256([_game_record(record) for record in match.records])
        ):
            raise ValueError("summary does not correspond to its match")
    return checked


def _tactical_gate(tactical: dict[str, object], depths: tuple[int, ...]) -> bool:
    key = f"depth_{depths[0]}_vs_{depths[-1]}"
    comparison = _mapping(_required(cast(dict[object, object], tactical), key), key)
    return _boolean(_required(comparison, "passes_gate"), "passes_gate")


def _tactical_depth_observations(
    tactical: dict[str, object],
    depths: tuple[int, ...],
) -> list[dict[str, object]]:
    root = cast(dict[object, object], tactical)
    evaluations = _mapping(_required(root, "evaluations"), "tactical evaluations")
    observations: list[dict[str, object]] = []
    for shallower, deeper in zip(depths[:-1], depths[1:], strict=True):
        shallow_name = f"minimax({shallower})"
        deep_name = f"minimax({deeper})"
        shallow = _mapping(_required(evaluations, shallow_name), shallow_name)
        deep = _mapping(_required(evaluations, deep_name), deep_name)
        shallow_solved = _integer(_required(shallow, "solved"), "shallow solved")
        deep_solved = _integer(_required(deep, "solved"), "deep solved")
        observations.append(
            {
                "shallower_depth": shallower,
                "deeper_depth": deeper,
                "shallower_solved": shallow_solved,
                "deeper_solved": deep_solved,
                "deeper_point_estimate_is_lower": deep_solved < shallow_solved,
            }
        )
    return observations


def _source_records() -> list[dict[str, object]]:
    return [
        {
            "id": "NIST-SAMPLE-SIZE",
            "authority": "official NIST/SEMATECH statistical handbook",
            "url": "https://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm",
            "supports": "two-sided normal-approximation sample-size equation",
        },
        {
            "id": "WILSON-1927",
            "authority": "original peer-reviewed article",
            "url": "https://doi.org/10.1080/01621459.1927.10502953",
            "supports": "score-interval construction underlying the reported Wilson-style bounds",
        },
        {
            "id": "BRADLEY-TERRY-1952",
            "authority": "original peer-reviewed article",
            "url": "https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091",
            "supports": "paired-comparison logistic model and the transitivity assumption",
        },
        {
            "id": "FIDE-RATING-REGULATIONS",
            "authority": "official federation regulation",
            "url": "https://handbook.fide.com/chapter/B022024",
            "supports": "expected-score/rating-difference reporting convention",
        },
        {
            "id": "WCDF-RULES",
            "authority": "primary game-rules publication",
            "url": "https://wcdf.net/rules/rules_of_checkers_english.pdf",
            "supports": "American Checkers legal play used by every arena game",
        },
    ]


def build_evaluation_report(  # noqa: PLR0913
    *,
    config: BaselineConfig,
    matches: tuple[MatchResult, ...],
    summaries: tuple[BaselineMatchSummary, ...],
    tactical: dict[str, object],
    git_commit: str,
    config_sha256: str,
    goal_sha256: str,
    raw_games_sha256: str,
    hardware: dict[str, object],
    dependencies: dict[str, object],
) -> dict[str, object]:
    """Assemble the source-backed, machine-readable Phase 5 baseline report.

    Args:
        config: Frozen experiment contract.
        matches: Complete replay results in any configured pair order.
        summaries: One digest-bound summary per match in the same order.
        tactical: Exact development-suite report.
        git_commit: Clean 40-character source commit.
        config_sha256: Exact configuration digest.
        goal_sha256: Read-only goal digest.
        raw_games_sha256: Saved replay archive digest.
        hardware: Non-secret runtime hardware metadata.
        dependencies: Exact relevant dependency versions.

    Returns:
        Complete JSON-compatible evidence and gate verdict document.

    Raises:
        TypeError: If containers or records have invalid runtime types.
        ValueError: If identities, pair coverage, or summaries disagree.
    """

    if not isinstance(config, BaselineConfig):
        raise TypeError("config must be a BaselineConfig")
    checked_matches = _validated_matches(matches)
    checked_summaries = _validate_summaries(checked_matches, summaries)
    if not isinstance(tactical, dict):
        raise TypeError("tactical must be a mapping")
    if GIT_SHA_PATTERN.fullmatch(_string(git_commit, "git_commit")) is None:
        raise ValueError("git_commit must be a lowercase 40-character Git SHA")
    checked_config_sha = _sha256(config_sha256, "config_sha256")
    checked_goal_sha = _sha256(goal_sha256, "goal_sha256")
    checked_raw_sha = _sha256(raw_games_sha256, "raw_games_sha256")
    if not isinstance(hardware, dict) or not isinstance(dependencies, dict):
        raise TypeError("hardware and dependencies must be mappings")
    expected_pairs = {frozenset(pair) for pair in config.comparisons}
    observed_pairs = {
        frozenset((match.first_agent, match.second_agent)) for match in checked_matches
    }
    if expected_pairs != observed_pairs or len(checked_matches) != len(config.comparisons):
        raise ValueError("matches must contain the complete configured comparison set")

    matrix = PayoffMatrix.from_matches(agents=BASELINE_AGENTS, matches=checked_matches)
    deeper_score = matrix.score("minimax(2)", "minimax(1)")
    tactical_pass = _tactical_gate(tactical, config.tactical_depths)
    plan = config.power_plan
    power_justified = (
        config.games_per_match == plan.balanced_games and plan.achieved_power >= plan.target_power
    )
    common_anchor_observations = [
        {
            "anchor": anchor,
            "minimax_1_score": matrix.score("minimax(1)", anchor),
            "minimax_2_score": matrix.score("minimax(2)", anchor),
            "deeper_point_estimate_is_lower": (
                matrix.score("minimax(2)", anchor) < matrix.score("minimax(1)", anchor)
            ),
        }
        for anchor in ("random", "greedy")
    ]
    tactical_depth_observations = _tactical_depth_observations(
        tactical,
        config.tactical_depths,
    )
    any_non_monotonic = (
        any(
            cast(bool, observation["deeper_point_estimate_is_lower"])
            for observation in common_anchor_observations
        )
        or deeper_score < NEUTRAL_SCORE
        or any(
            cast(bool, observation["deeper_point_estimate_is_lower"])
            for observation in tactical_depth_observations
        )
    )
    diagnosis = (
        "Observed tactical depth regression is consistent with horizon effects in the "
        "non-quiescent material evaluator; it is an evaluator finding, not evidence of a rules "
        "engine defect. The depth-1 to depth-3 gate remains the predeclared decision criterion."
        if any_non_monotonic
        else (
            "No point-estimate regression was observed, but monotonic search strength is not "
            "assumed."
        )
    )
    return {
        "schema_version": 1,
        "experiment_type": "baseline",
        "engineering_objective": (
            "Estimate every fixed baseline pair at the predeclared power and test whether "
            "minimax(2) avoids catastrophic inversion while improving exact tactics."
        ),
        "config": _config_record(config),
        "identity": {
            "git_commit": git_commit,
            "config_sha256": checked_config_sha,
            "goal_sha256": checked_goal_sha,
            "raw_games_sha256": checked_raw_sha,
        },
        "hardware": dict(hardware),
        "dependencies": dict(dependencies),
        "power_plan": _power_record(plan),
        "matches": [_summary_record(summary) for summary in checked_summaries],
        "population": build_population_report(checked_matches),
        "tactical": tactical,
        "search_depth_non_monotonicity": {
            "reported_not_assumed_away": True,
            "minimax_2_vs_minimax_1_score": deeper_score,
            "common_anchor_observations": common_anchor_observations,
            "tactical_depth_observations": tactical_depth_observations,
            "any_point_estimate_non_monotonicity": any_non_monotonic,
            "diagnosis": diagnosis,
            "interpretation": (
                "Depth-limited material minimax is not guaranteed to improve monotonically; "
                "these are descriptive point estimates, not six independent confirmatory tests."
            ),
        },
        "gate_5": {
            "power_justified": power_justified,
            "no_catastrophic_inversion": deeper_score >= INVERSION_FLOOR,
            "catastrophic_inversion_floor": INVERSION_FLOOR,
            "minimax_2_vs_minimax_1_score": deeper_score,
            "tactical_superset_or_substantially_more": tactical_pass,
            "technical_pass": power_justified and deeper_score >= INVERSION_FLOOR and tactical_pass,
            "sealed_suite_status": "NOT_EVALUATED",
        },
        "statistical_assumptions": [
            (
                "Each game receives distinct injectively derived pseudorandom seed streams and "
                "colours alternate exactly; statistical independence is an explicit modeling "
                "assumption, not a consequence proved by seed uniqueness."
            ),
            (
                "The game-count calculation treats bounded scores 0/0.5/1 as Bernoulli, which "
                "upper-bounds their variance at a fixed mean under the stated model."
            ),
            (
                "The NIST continuity correction is not applied; power_justified refers only to "
                "the explicitly declared uncorrected two-sided normal approximation."
            ),
            (
                "Wilson-style intervals apply the score formula to fractional draw points; "
                "their finite-sample coverage with draws is approximate, not exact."
            ),
            (
                "League Elo confidence intervals are delta-method projections conditional on "
                "approximate transitivity; residuals and directed cycles are reported separately."
            ),
            (
                "The six pair estimates are descriptive as a population; no family-wise "
                "multiplicity correction is claimed."
            ),
        ],
        "sources": _source_records(),
        "limitations": [
            (
                "The development tactical suite is programmatically verified, pending human "
                "review, and selected for depth-3 success; it is not an unbiased tactical sample."
            ),
            "No sealed-suite result, trained best response, or external engine anchor is claimed.",
            (
                "Match scores measure these fixed seeded implementations under declared "
                "engine variants."
            ),
            (
                "Every match starts from the standard initial position; no opening-ballot "
                "distribution was evaluated."
            ),
            (
                "The 100-Elo residual threshold for approximate transitivity is a declared "
                "project diagnostic, not an externally calibrated cutoff."
            ),
        ],
    }
