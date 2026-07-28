"""Versioned exact tactical suite, terminal-only solver, and depth gate."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import cast

from checkers.agents.base import Agent
from checkers.env.encoding import DEFAULT_MAX_PLIES
from checkers.env.masking import ACTION_COUNT, action_to_step, legal_action_map
from checkers.rules.moves import apply_step
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import terminal_outcome

DEV_TACTICAL_CASES = 50
DEV_TACTICAL_FILENAME = "dev_tactics_v1.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PAIR_SIZE = 2


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


def _optional_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name)


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    checked = tuple(_name(item, field_name) for item in value)
    if not checked:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(checked)) != len(checked):
        raise ValueError(f"{field_name} must contain unique values")
    return checked


def _action_tuple(value: object, field_name: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    for action in value:
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError(f"{field_name} must contain integers")
        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"{field_name} action must be in [0, {ACTION_COUNT - 1}]")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class TacticalSuiteManifest:
    """Provenance, licensing, review, schema, and content identity for the dev suite."""

    schema_version: int
    name: str
    case_count: int
    horizon_completed_moves: int
    generator_seed: int
    games_scanned: int
    depth1_misses: int
    depth3_solved: int
    provenance: str
    license: str
    author: str
    creation_method: str
    review_status: str
    grade_band: str
    safety_categories: tuple[str, ...]
    subject_categories: tuple[str, ...]
    difficulty: str
    split: str
    cases_sha256: str
    generator_source_sha256: str
    rules_source_sha256: str
    goal_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _name(self.name, "name")
        if self.case_count != DEV_TACTICAL_CASES:
            raise ValueError(f"case_count must be {DEV_TACTICAL_CASES}")
        _positive_integer(self.horizon_completed_moves, "horizon_completed_moves")
        _nonnegative_integer(self.generator_seed, "generator_seed")
        _positive_integer(self.games_scanned, "games_scanned")
        misses = _nonnegative_integer(self.depth1_misses, "depth1_misses")
        if misses > self.case_count:
            raise ValueError("depth1_misses cannot exceed case_count")
        if self.depth3_solved != self.case_count:
            raise ValueError("depth3_solved must equal case_count")
        _name(self.provenance, "provenance")
        _name(self.license, "license")
        _name(self.author, "author")
        _name(self.creation_method, "creation_method")
        _name(self.review_status, "review_status")
        _name(self.grade_band, "grade_band")
        _string_tuple(self.safety_categories, "safety_categories")
        _string_tuple(self.subject_categories, "subject_categories")
        _name(self.difficulty, "difficulty")
        _name(self.split, "split")
        _sha256(self.cases_sha256, "cases_sha256")
        _sha256(self.generator_source_sha256, "generator_source_sha256")
        _sha256(self.rules_source_sha256, "rules_source_sha256")
        _sha256(self.goal_sha256, "goal_sha256")


@dataclass(frozen=True, slots=True)
class TacticalCase:
    """One reachable nontrivial position and its exact forced-win root actions."""

    case_id: str
    state: State
    max_completed_moves: int
    winning_actions: tuple[int, ...]
    replay_actions: tuple[int, ...]
    rationale: str
    duplicate_group: str
    source_game: int
    source_step: int
    difficulty: str

    def __post_init__(self) -> None:
        _name(self.case_id, "case_id")
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        if self.state.capture_in_progress:
            raise ValueError("tactical state must be a completed-move boundary")
        if terminal_outcome(self.state) is not None:
            raise ValueError("tactical state must be nonterminal")
        _positive_integer(self.max_completed_moves, "max_completed_moves")
        winning = _action_tuple(self.winning_actions, "winning_actions", allow_empty=False)
        legal = tuple(legal_action_map(self.state))
        if not set(winning) < set(legal):
            raise ValueError("winning_actions must be a non-empty strict subset of legal actions")
        _action_tuple(self.replay_actions, "replay_actions", allow_empty=False)
        _name(self.rationale, "rationale")
        _sha256(self.duplicate_group, "duplicate_group")
        _nonnegative_integer(self.source_game, "source_game")
        _positive_integer(self.source_step, "source_step")
        _name(self.difficulty, "difficulty")


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


def tactical_case_record(case: TacticalCase) -> dict[str, object]:
    """Return the canonical JSON-compatible record for one tactical case.

    Args:
        case: Validated immutable tactical case.

    Returns:
        Stable mapping used by the generator and content digest.

    Raises:
        TypeError: If ``case`` is not a ``TacticalCase``.
    """

    if not isinstance(case, TacticalCase):
        raise TypeError("case must be a TacticalCase")
    return {
        "case_id": case.case_id,
        "state": _state_record(case.state),
        "max_completed_moves": case.max_completed_moves,
        "winning_actions": list(case.winning_actions),
        "replay_actions": list(case.replay_actions),
        "rationale": case.rationale,
        "duplicate_group": case.duplicate_group,
        "source_game": case.source_game,
        "source_step": case.source_step,
        "difficulty": case.difficulty,
    }


def tactical_cases_sha256(cases: tuple[TacticalCase, ...]) -> str:
    """Hash canonical case records in declared order.

    Args:
        cases: Ordered tactical cases.

    Returns:
        Lowercase SHA-256 digest.

    Raises:
        TypeError: If cases is not an exact tuple of ``TacticalCase`` values.
    """

    if not isinstance(cases, tuple) or not all(isinstance(case, TacticalCase) for case in cases):
        raise TypeError("cases must be a tuple of TacticalCase values")
    payload = json.dumps(
        [tactical_case_record(case) for case in cases],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_tactical_case(case: TacticalCase) -> State:
    """Replay a case's exact action prefix from standard initial play.

    Args:
        case: Tactical case containing a canonical action prefix.

    Returns:
        Exact reached state, including counters and ply.

    Raises:
        TypeError: If ``case`` is not a ``TacticalCase``.
        ValueError: If a replay action is illegal.
    """

    if not isinstance(case, TacticalCase):
        raise TypeError("case must be a TacticalCase")
    state = State.initial()
    for action in case.replay_actions:
        step = action_to_step(state, action)
        state = apply_step(state, step).after
    return state


def _can_force_win(
    state: State,
    *,
    root: PlayerId,
    remaining_moves: int,
    max_plies: int,
    cache: dict[tuple[State, int], bool],
) -> bool:
    key = (state, remaining_moves)
    if key in cache:
        return cache[key]
    outcome = terminal_outcome(state, max_plies=max_plies)
    if outcome is not None:
        result = outcome.winner is root
        cache[key] = result
        return result
    if remaining_moves == 0:
        cache[key] = False
        return False

    child_results: list[bool] = []
    for step in legal_action_map(state).values():
        transition = apply_step(state, step)
        child_results.append(
            _can_force_win(
                transition.after,
                root=root,
                remaining_moves=remaining_moves - int(transition.move_completed),
                max_plies=max_plies,
                cache=cache,
            )
        )
    result = any(child_results) if state.side_to_move is root else all(child_results)
    cache[key] = result
    return result


def forced_win_actions(
    state: State,
    *,
    max_completed_moves: int,
    max_plies: int = DEFAULT_MAX_PLIES,
) -> tuple[int, ...]:
    """Find actions that force the current player to win within a completed-move horizon.

    This exhaustive AND/OR solver uses terminal outcomes only. It has no material evaluator and
    shares no search implementation with ``MinimaxAgent``.

    Args:
        state: Complete state at any transition boundary.
        max_completed_moves: Positive horizon counting completed checkers moves, not jump steps.
        max_plies: Exact engine ply-cap used for terminal evaluation.

    Returns:
        Deterministically ordered canonical actions that force a terminal win.

    Raises:
        TypeError: If inputs have invalid runtime types.
        ValueError: If horizons are not positive.
    """

    if not isinstance(state, State):
        raise TypeError("state must be a State")
    horizon = _positive_integer(max_completed_moves, "max_completed_moves")
    checked_max_plies = _positive_integer(max_plies, "max_plies")
    if terminal_outcome(state, max_plies=checked_max_plies) is not None:
        return ()
    root = state.side_to_move
    cache: dict[tuple[State, int], bool] = {}
    winners: list[int] = []
    for action, step in legal_action_map(state).items():
        transition = apply_step(state, step)
        if _can_force_win(
            transition.after,
            root=root,
            remaining_moves=horizon - int(transition.move_completed),
            max_plies=checked_max_plies,
            cache=cache,
        ):
            winners.append(action)
    return tuple(winners)


@dataclass(frozen=True, slots=True)
class TacticalSuite:
    """Validated manifest and exact, reachable, deduplicated cases."""

    manifest: TacticalSuiteManifest
    cases: tuple[TacticalCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TacticalSuiteManifest):
            raise TypeError("manifest must be a TacticalSuiteManifest")
        if not isinstance(self.cases, tuple) or not all(
            isinstance(case, TacticalCase) for case in self.cases
        ):
            raise TypeError("cases must be a tuple of TacticalCase values")
        if len(self.cases) != self.manifest.case_count:
            raise ValueError("manifest case_count disagrees with tactical cases")
        if any(
            case.max_completed_moves != self.manifest.horizon_completed_moves for case in self.cases
        ):
            raise ValueError("case horizon disagrees with manifest")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("tactical case IDs must be unique")
        if len({case.state for case in self.cases}) != len(self.cases):
            raise ValueError("tactical states must be unique")
        if len({case.duplicate_group for case in self.cases}) != len(self.cases):
            raise ValueError("tactical duplicate groups must be unique")
        if tactical_cases_sha256(self.cases) != self.manifest.cases_sha256:
            raise ValueError("manifest cases digest disagrees with tactical cases")
        for case in self.cases:
            if replay_tactical_case(case) != case.state:
                raise ValueError(f"case {case.case_id} replay does not reach frozen state")
            if (
                forced_win_actions(
                    case.state,
                    max_completed_moves=case.max_completed_moves,
                )
                != case.winning_actions
            ):
                raise ValueError(f"case {case.case_id} frozen winning actions disagree with oracle")


def _mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a string-keyed object")
    return cast(dict[str, object], value)


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def _required(mapping: dict[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"missing required tactical field {key!r}")
    return mapping[key]


def _pair(value: object, field_name: str) -> tuple[int, int]:
    values = _list(value, field_name)
    if len(values) != PAIR_SIZE or any(
        isinstance(item, bool) or not isinstance(item, int) for item in values
    ):
        raise TypeError(f"{field_name} must be a two-integer list")
    return cast(int, values[0]), cast(int, values[1])


def _parse_state(value: object) -> State:
    mapping = _mapping(value, "state")
    side_value = _required(mapping, "side_to_move")
    if isinstance(side_value, bool) or not isinstance(side_value, int):
        raise TypeError("side_to_move must be an integer")
    try:
        side = PlayerId(side_value)
    except ValueError as error:
        raise ValueError("side_to_move must be 0 or 1") from error
    return State(
        men=_pair(_required(mapping, "men"), "men"),
        kings=_pair(_required(mapping, "kings"), "kings"),
        side_to_move=side,
        capture_in_progress=_boolean(
            _required(mapping, "capture_in_progress"),
            "capture_in_progress",
        ),
        moving_square=_optional_integer(_required(mapping, "moving_square"), "moving_square"),
        sequence_origin=_optional_integer(
            _required(mapping, "sequence_origin"),
            "sequence_origin",
        ),
        captured_pending=_integer(_required(mapping, "captured_pending"), "captured_pending"),
        no_progress=_pair(_required(mapping, "no_progress"), "no_progress"),
        ply=_integer(_required(mapping, "ply"), "ply"),
    )


def _tuple_from_json(value: object, field_name: str) -> tuple[object, ...]:
    return tuple(_list(value, field_name))


def _parse_manifest(value: object) -> TacticalSuiteManifest:
    mapping = _mapping(value, "manifest")
    return TacticalSuiteManifest(
        schema_version=_integer(_required(mapping, "schema_version"), "schema_version"),
        name=_name(_required(mapping, "name"), "name"),
        case_count=_integer(_required(mapping, "case_count"), "case_count"),
        horizon_completed_moves=_integer(
            _required(mapping, "horizon_completed_moves"),
            "horizon_completed_moves",
        ),
        generator_seed=_integer(_required(mapping, "generator_seed"), "generator_seed"),
        games_scanned=_integer(_required(mapping, "games_scanned"), "games_scanned"),
        depth1_misses=_integer(_required(mapping, "depth1_misses"), "depth1_misses"),
        depth3_solved=_integer(_required(mapping, "depth3_solved"), "depth3_solved"),
        provenance=_name(_required(mapping, "provenance"), "provenance"),
        license=_name(_required(mapping, "license"), "license"),
        author=_name(_required(mapping, "author"), "author"),
        creation_method=_name(_required(mapping, "creation_method"), "creation_method"),
        review_status=_name(_required(mapping, "review_status"), "review_status"),
        grade_band=_name(_required(mapping, "grade_band"), "grade_band"),
        safety_categories=_string_tuple(
            _tuple_from_json(_required(mapping, "safety_categories"), "safety_categories"),
            "safety_categories",
        ),
        subject_categories=_string_tuple(
            _tuple_from_json(
                _required(mapping, "subject_categories"),
                "subject_categories",
            ),
            "subject_categories",
        ),
        difficulty=_name(_required(mapping, "difficulty"), "difficulty"),
        split=_name(_required(mapping, "split"), "split"),
        cases_sha256=_sha256(_required(mapping, "cases_sha256"), "cases_sha256"),
        generator_source_sha256=_sha256(
            _required(mapping, "generator_source_sha256"),
            "generator_source_sha256",
        ),
        rules_source_sha256=_sha256(
            _required(mapping, "rules_source_sha256"),
            "rules_source_sha256",
        ),
        goal_sha256=_sha256(_required(mapping, "goal_sha256"), "goal_sha256"),
    )


def _parse_case(value: object) -> TacticalCase:
    mapping = _mapping(value, "case")
    return TacticalCase(
        case_id=_name(_required(mapping, "case_id"), "case_id"),
        state=_parse_state(_required(mapping, "state")),
        max_completed_moves=_integer(
            _required(mapping, "max_completed_moves"),
            "max_completed_moves",
        ),
        winning_actions=_action_tuple(
            _tuple_from_json(_required(mapping, "winning_actions"), "winning_actions"),
            "winning_actions",
            allow_empty=False,
        ),
        replay_actions=_action_tuple(
            _tuple_from_json(_required(mapping, "replay_actions"), "replay_actions"),
            "replay_actions",
            allow_empty=False,
        ),
        rationale=_name(_required(mapping, "rationale"), "rationale"),
        duplicate_group=_sha256(
            _required(mapping, "duplicate_group"),
            "duplicate_group",
        ),
        source_game=_integer(_required(mapping, "source_game"), "source_game"),
        source_step=_integer(_required(mapping, "source_step"), "source_step"),
        difficulty=_name(_required(mapping, "difficulty"), "difficulty"),
    )


def parse_tactical_suite(text: str) -> TacticalSuite:
    """Parse and fully verify a tactical-suite JSON document.

    Args:
        text: Versioned suite JSON.

    Returns:
        Manifest and cases after digest, replay, and exact-solution checks.

    Raises:
        TypeError: If text or schema fields have invalid runtime types.
        ValueError: If JSON, schema, digest, replay, or solutions are invalid.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid tactical-suite JSON") from error
    root = _mapping(loaded, "root")
    manifest = _parse_manifest(_required(root, "manifest"))
    cases = tuple(_parse_case(value) for value in _list(_required(root, "cases"), "cases"))
    return TacticalSuite(manifest=manifest, cases=cases)


def load_dev_tactical_suite() -> TacticalSuite:
    """Load and fully verify the packaged 50-case development tactical suite.

    Returns:
        Immutable verified development suite.

    Raises:
        ValueError: If packaged content fails any schema or exactness check.
    """

    resource = resources.files("checkers.eval").joinpath("data", DEV_TACTICAL_FILENAME)
    return parse_tactical_suite(resource.read_text(encoding="utf-8"))


class TacticalAgentError(RuntimeError):
    """Illegal tactical-suite action attributed to one policy and case."""

    def __init__(self, *, agent_name: str, case_id: str, action: object) -> None:
        """Record exact policy/case attribution for an illegal selection."""

        self.agent_name = agent_name
        self.case_id = case_id
        self.action = action
        super().__init__(f"agent {agent_name!r} returned illegal action {action!r} on {case_id}")


@dataclass(frozen=True, slots=True)
class TacticalCaseResult:
    """One policy selection and exact-solution verdict."""

    case_id: str
    selected_action: int
    solved: bool

    def __post_init__(self) -> None:
        _name(self.case_id, "case_id")
        _action_tuple((self.selected_action,), "selected_action", allow_empty=False)
        if not isinstance(self.solved, bool):
            raise TypeError("solved must be bool")


@dataclass(frozen=True, slots=True)
class TacticalEvaluation:
    """Per-case tactical selections and aggregate solved count."""

    agent_name: str
    results: tuple[TacticalCaseResult, ...]
    solved: int

    def __post_init__(self) -> None:
        _name(self.agent_name, "agent_name")
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, TacticalCaseResult) for result in self.results
        ):
            raise TypeError("results must be a tuple of TacticalCaseResult values")
        if not self.results:
            raise ValueError("results must not be empty")
        if len({result.case_id for result in self.results}) != len(self.results):
            raise ValueError("evaluation case IDs must be unique")
        checked_solved = _nonnegative_integer(self.solved, "solved")
        if checked_solved != sum(result.solved for result in self.results):
            raise ValueError("solved count disagrees with tactical case results")

    @classmethod
    def from_case_results(
        cls,
        *,
        agent_name: str,
        results: tuple[TacticalCaseResult, ...],
    ) -> TacticalEvaluation:
        """Construct an evaluation and derive its solved count.

        Args:
            agent_name: Stable policy label.
            results: Ordered non-empty case results.

        Returns:
            Validated aggregate evaluation.
        """

        if not isinstance(results, tuple):
            raise TypeError("results must be a tuple")
        return cls(
            agent_name=agent_name,
            results=results,
            solved=sum(result.solved for result in results),
        )

    @property
    def total(self) -> int:
        """Return evaluated case count."""

        return len(self.results)

    @property
    def accuracy(self) -> float:
        """Return exact tactical solve fraction."""

        return self.solved / self.total

    @property
    def solved_case_ids(self) -> tuple[str, ...]:
        """Return solved IDs in suite order."""

        return tuple(result.case_id for result in self.results if result.solved)


def evaluate_tactical(
    agent: Agent,
    cases: tuple[TacticalCase, ...],
) -> TacticalEvaluation:
    """Evaluate one policy against frozen exact winning-action sets.

    Args:
        agent: Runtime policy under evaluation.
        cases: Ordered non-empty tactical cases.

    Returns:
        Per-case selections and solved count.

    Raises:
        TypeError: If the policy or cases have invalid runtime types.
        ValueError: If cases is empty.
        TacticalAgentError: If the policy selects an illegal action.
    """

    if not isinstance(agent, Agent):
        raise TypeError("agent must implement Agent")
    if not isinstance(cases, tuple) or not all(isinstance(case, TacticalCase) for case in cases):
        raise TypeError("cases must be a tuple of TacticalCase values")
    if not cases:
        raise ValueError("cases must not be empty")
    results: list[TacticalCaseResult] = []
    for case in cases:
        action = agent.select_action(case.state)
        if action not in legal_action_map(case.state):
            raise TacticalAgentError(
                agent_name=agent.name,
                case_id=case.case_id,
                action=action,
            )
        results.append(
            TacticalCaseResult(
                case_id=case.case_id,
                selected_action=action,
                solved=action in case.winning_actions,
            )
        )
    return TacticalEvaluation.from_case_results(agent_name=agent.name, results=tuple(results))


@dataclass(frozen=True, slots=True)
class TacticalDepthComparison:
    """Exact superset/substantial-gain tactical gate evidence."""

    shallow_agent: str
    deep_agent: str
    case_ids: tuple[str, ...]
    gained_case_ids: tuple[str, ...]
    lost_case_ids: tuple[str, ...]
    substantial_gain: int
    deep_is_superset: bool
    net_gain: int
    substantially_more: bool
    passes_gate: bool

    def __post_init__(self) -> None:
        shallow = _name(self.shallow_agent, "shallow_agent")
        deep = _name(self.deep_agent, "deep_agent")
        if shallow == deep:
            raise ValueError("comparison agents must be distinct")
        cases = _string_tuple(self.case_ids, "case_ids")
        gained = self._subset(self.gained_case_ids, cases, "gained_case_ids")
        lost = self._subset(self.lost_case_ids, cases, "lost_case_ids")
        if set(gained) & set(lost):
            raise ValueError("comparison gained/lost cases must be disjoint")
        threshold = _positive_integer(self.substantial_gain, "substantial_gain")
        if not isinstance(self.deep_is_superset, bool):
            raise TypeError("deep_is_superset must be bool")
        if self.deep_is_superset != (not lost):
            raise ValueError("comparison deep_is_superset disagrees with lost cases")
        if isinstance(self.net_gain, bool) or not isinstance(self.net_gain, int):
            raise TypeError("net_gain must be an integer")
        if self.net_gain != len(gained) - len(lost):
            raise ValueError("comparison net_gain disagrees with gained/lost cases")
        if not isinstance(self.substantially_more, bool):
            raise TypeError("substantially_more must be bool")
        if self.substantially_more != (self.net_gain >= threshold):
            raise ValueError("comparison substantially_more disagrees with net gain")
        if not isinstance(self.passes_gate, bool):
            raise TypeError("passes_gate must be bool")
        if self.passes_gate != (self.deep_is_superset or self.substantially_more):
            raise ValueError("comparison passes_gate disagrees with gate criteria")

    @staticmethod
    def _subset(values: object, cases: tuple[str, ...], field_name: str) -> tuple[str, ...]:
        checked = _string_tuple(values, field_name) if values else ()
        if not set(checked) <= set(cases):
            raise ValueError(f"comparison {field_name} contains an unknown case")
        return checked

    @property
    def gained(self) -> int:
        """Return newly solved cases."""

        return len(self.gained_case_ids)


def compare_tactical(
    shallow: TacticalEvaluation,
    deep: TacticalEvaluation,
    *,
    substantial_gain: int,
) -> TacticalDepthComparison:
    """Compare exact solved sets under the corrected non-monotone search gate.

    Args:
        shallow: Shallower-search evaluation.
        deep: Deeper-search evaluation over identical ordered cases.
        substantial_gain: Minimum net solved-case gain for the alternative gate branch.

    Returns:
        Superset, gained/lost, net-gain, and pass/fail evidence.

    Raises:
        TypeError: If evaluations have invalid runtime types.
        ValueError: If agents/case IDs are not comparable or the threshold is invalid.
    """

    if not isinstance(shallow, TacticalEvaluation):
        raise TypeError("shallow must be a TacticalEvaluation")
    if not isinstance(deep, TacticalEvaluation):
        raise TypeError("deep must be a TacticalEvaluation")
    if shallow.agent_name == deep.agent_name:
        raise ValueError("comparison agents must be distinct")
    shallow_ids = tuple(result.case_id for result in shallow.results)
    deep_ids = tuple(result.case_id for result in deep.results)
    if shallow_ids != deep_ids:
        raise ValueError("comparison requires identical ordered case IDs")
    threshold = _positive_integer(substantial_gain, "substantial_gain")
    shallow_solved = set(shallow.solved_case_ids)
    deep_solved = set(deep.solved_case_ids)
    gained = tuple(case_id for case_id in shallow_ids if case_id in deep_solved - shallow_solved)
    lost = tuple(case_id for case_id in shallow_ids if case_id in shallow_solved - deep_solved)
    net_gain = len(gained) - len(lost)
    deep_is_superset = not lost
    substantially_more = net_gain >= threshold
    return TacticalDepthComparison(
        shallow_agent=shallow.agent_name,
        deep_agent=deep.agent_name,
        case_ids=shallow_ids,
        gained_case_ids=gained,
        lost_case_ids=lost,
        substantial_gain=threshold,
        deep_is_superset=deep_is_superset,
        net_gain=net_gain,
        substantially_more=substantially_more,
        passes_gate=deep_is_superset or substantially_more,
    )
