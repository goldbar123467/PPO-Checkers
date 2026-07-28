"""Fixed-anchor, tactical, and two-policy population evaluation for learned policies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import torch

from checkers.agents.greedy_agent import GreedyAgent
from checkers.agents.minimax_agent import MinimaxAgent
from checkers.agents.policy_agent import PolicyAgent, PolicyMode
from checkers.agents.random_agent import RandomAgent
from checkers.eval.arena import AgentSpec, GameRecord, MatchResult, play_balanced_match
from checkers.eval.population import PayoffMatrix, league_elo, three_cycles
from checkers.eval.suites import evaluate_tactical, load_dev_tactical_suite
from checkers.rl.determinism import derive_stream_seed
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import State

EXPLOITABILITY_STATUSES = frozenset({"MEASURED", "NOT_EVALUATED"})
NOT_EVALUATED_SENTINEL = -1.0
MIN_BALANCED_GAMES = 2


@dataclass(frozen=True, slots=True)
class ExploitabilityEvidence:
    """Measured best-response score or an unmistakable numeric missing-value sentinel."""

    status: str
    score: float
    training_steps: int | None

    def __post_init__(self) -> None:
        if self.status not in EXPLOITABILITY_STATUSES:
            raise ValueError("exploitability status must be MEASURED or NOT_EVALUATED")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("exploitability score must be numeric")
        checked_score = float(self.score)
        if not math.isfinite(checked_score):
            raise ValueError("exploitability score must be finite")
        if self.status == "NOT_EVALUATED":
            if checked_score != NOT_EVALUATED_SENTINEL or self.training_steps is not None:
                raise ValueError("NOT_EVALUATED exploitability must use the -1 sentinel")
            return
        if not 0.0 <= checked_score <= 1.0:
            raise ValueError("measured exploitability score must be in [0, 1]")
        if (
            isinstance(self.training_steps, bool)
            or not isinstance(self.training_steps, int)
            or self.training_steps < 1
        ):
            raise ValueError("measured exploitability requires positive training_steps")

    @classmethod
    def not_evaluated(cls) -> ExploitabilityEvidence:
        """Return the explicit pre-best-response sentinel record."""

        return cls(
            status="NOT_EVALUATED",
            score=NOT_EVALUATED_SENTINEL,
            training_steps=None,
        )

    @classmethod
    def measured(cls, *, score: float, training_steps: int) -> ExploitabilityEvidence:
        """Return a measured short-budget best-response record."""

        return cls(status="MEASURED", score=score, training_steps=training_steps)


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Scalar W&B metrics, payoff table rows, matches, and proxy status."""

    scalar_metrics: dict[str, float]
    payoff_rows: tuple[dict[str, object], ...]
    game_rows: tuple[dict[str, object], ...]
    anchor_matches: tuple[MatchResult, ...]
    sampled_random_match: MatchResult
    population_match: MatchResult
    exploitability_status: str


def _policy_spec(
    *,
    network: CheckersNetwork,
    mode: PolicyMode,
    name: str,
) -> AgentSpec:
    def factory(seed: int) -> PolicyAgent:
        return PolicyAgent(network=network, mode=mode, seed=seed, name=name)

    return AgentSpec(name=name, factory=factory)


def _anchor_specs(*, max_plies: int) -> tuple[tuple[str, AgentSpec], ...]:
    return (
        ("random", AgentSpec(name="random", factory=lambda seed: RandomAgent(seed=seed))),
        ("greedy", AgentSpec(name="greedy", factory=lambda seed: GreedyAgent(seed=seed))),
        (
            "minimax2",
            AgentSpec(
                name="minimax(2)",
                factory=lambda seed: MinimaxAgent(depth=2, seed=seed, max_plies=max_plies),
            ),
        ),
    )


def _match_metrics(prefix: str, match: MatchResult) -> dict[str, float]:
    return {
        prefix: match.score.score,
        f"{prefix}_ci_low": match.score.low,
        f"{prefix}_ci_high": match.score.high,
        f"{prefix}_games": float(match.score.games),
    }


def _payoff_rows(matrix: PayoffMatrix) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for row_agent in matrix.agents:
        for column_agent in matrix.agents:
            wdl = matrix.wdl(row_agent, column_agent)
            rows.append(
                {
                    "row_agent": row_agent,
                    "column_agent": column_agent,
                    "wins": wdl.wins,
                    "draws": wdl.draws,
                    "losses": wdl.losses,
                    "games": wdl.games,
                    "score": matrix.score(row_agent, column_agent),
                }
            )
    return tuple(rows)


def _perspective_result(record: GameRecord, perspective_agent: str) -> str:
    if record.outcome.winner is None:
        return "draw"
    winner_agent = record.red_agent if record.outcome.winner.name == "RED" else record.white_agent
    return "win" if winner_agent == perspective_agent else "loss"


def game_rows_from_matches(
    labelled_matches: tuple[tuple[str, MatchResult], ...],
) -> tuple[dict[str, object], ...]:
    """Render labelled arena matches as replayable ACF/action table rows."""

    rows: list[dict[str, object]] = []
    for label, match in labelled_matches:
        for game_index, record in enumerate(match.records):
            rows.append(
                {
                    "match": label,
                    "game_index": game_index,
                    "perspective_agent": match.first_agent,
                    "perspective_result": _perspective_result(record, match.first_agent),
                    "red_agent": record.red_agent,
                    "white_agent": record.white_agent,
                    "winner": (
                        "DRAW" if record.outcome.winner is None else record.outcome.winner.name
                    ),
                    "termination_reason": record.outcome.reason.value,
                    "steps": record.steps,
                    "completed_moves": record.completed_moves,
                    "moves": " ".join(record.moves),
                    "actions": " ".join(str(action) for action in record.actions),
                    "red_seed": record.red_seed,
                    "white_seed": record.white_seed,
                    "environment_seed": record.environment_seed,
                }
            )
    return tuple(rows)


def _clone_network(
    network: CheckersNetwork,
    model_state: Mapping[str, torch.Tensor],
) -> CheckersNetwork:
    clone = deepcopy(network)
    clone.load_state_dict(model_state, strict=True)
    return clone


def evaluate_development_policy(  # noqa: PLR0913, PLR0914
    *,
    network: CheckersNetwork,
    initial_model_state: Mapping[str, torch.Tensor],
    games: int,
    seed: int,
    max_plies: int,
    repetition_draws: bool,
    exploitability: ExploitabilityEvidence,
    initial_state: State | None = None,
) -> PolicyEvaluation:
    """Evaluate greedy current play against fixed anchors and the pinned initial policy."""

    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    if not isinstance(exploitability, ExploitabilityEvidence):
        raise TypeError("exploitability must be ExploitabilityEvidence")
    if isinstance(games, bool) or not isinstance(games, int):
        raise TypeError("games must be an integer")
    if games < MIN_BALANCED_GAMES or games % MIN_BALANCED_GAMES:
        raise ValueError("games must be a positive even colour-balanced count")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    if not isinstance(repetition_draws, bool):
        raise TypeError("repetition_draws must be bool")

    current = _policy_spec(network=network, mode="greedy", name="current")
    scalar_metrics: dict[str, float] = {}
    anchor_matches: list[MatchResult] = []
    for index, (metric_name, anchor) in enumerate(_anchor_specs(max_plies=max_plies)):
        match = play_balanced_match(
            first=current,
            second=anchor,
            games=games,
            seed=derive_stream_seed(seed, index),
            initial_state=initial_state,
            max_plies=max_plies,
            repetition_draws=repetition_draws,
        )
        anchor_matches.append(match)
        scalar_metrics.update(_match_metrics(f"eval/vs_{metric_name}", match))

    sampled_random = play_balanced_match(
        first=_policy_spec(network=network, mode="sampled", name="current-sampled"),
        second=AgentSpec(name="random", factory=lambda value: RandomAgent(seed=value)),
        games=games,
        seed=derive_stream_seed(seed, 3),
        initial_state=initial_state,
        max_plies=max_plies,
        repetition_draws=repetition_draws,
    )
    scalar_metrics["eval/greedy_vs_sampled_delta"] = (
        anchor_matches[0].score.score - sampled_random.score.score
    )

    initial_network = _clone_network(network, initial_model_state)
    population_match = play_balanced_match(
        first=current,
        second=_policy_spec(network=initial_network, mode="greedy", name="initial"),
        games=games,
        seed=derive_stream_seed(seed, 4),
        initial_state=initial_state,
        max_plies=max_plies,
        repetition_draws=repetition_draws,
    )
    matrix = PayoffMatrix.from_matches(
        agents=("current", "initial"),
        matches=(population_match,),
    )
    elo = league_elo(matrix).rating_for("current")
    cycles = three_cycles(matrix)
    scalar_metrics.update(
        {
            "eval/league_elo": elo.rating,
            "eval/league_elo_ci_low": elo.low,
            "eval/league_elo_ci_high": elo.high,
            "eval/three_cycle_count": float(cycles.count),
            "eval/exploitability_proxy": float(exploitability.score),
            "eval/dev_tactical_acc": evaluate_tactical(
                PolicyAgent(network=network, mode="greedy", seed=seed),
                load_dev_tactical_suite().cases,
            ).accuracy,
        }
    )
    return PolicyEvaluation(
        scalar_metrics=scalar_metrics,
        payoff_rows=_payoff_rows(matrix),
        game_rows=game_rows_from_matches(
            (
                ("vs_random", anchor_matches[0]),
                ("vs_greedy", anchor_matches[1]),
                ("vs_minimax2", anchor_matches[2]),
                ("sampled_vs_random", sampled_random),
                ("current_vs_initial", population_match),
            )
        ),
        anchor_matches=tuple(anchor_matches),
        sampled_random_match=sampled_random,
        population_match=population_match,
        exploitability_status=exploitability.status,
    )
