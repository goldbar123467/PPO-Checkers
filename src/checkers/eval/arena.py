"""Deterministic game execution and colour-balanced match scheduling."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from checkers.agents.base import Agent
from checkers.env.checkers_env import CheckersEnv, IllegalActionError
from checkers.env.encoding import DEFAULT_MAX_PLIES
from checkers.env.masking import ACTION_COUNT
from checkers.eval.power import MatchScore, score_interval
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome

UINT64_MASK = (1 << 64) - 1
STREAMS_PER_GAME = 3
SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB

AgentFactory = Callable[[int], Agent]


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _seed(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if not 0 <= value <= UINT64_MASK:
        raise ValueError(f"{field_name} must be an unsigned 64-bit integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 < checked < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    return checked


def _splitmix64(value: int) -> int:
    mixed = (value + SPLITMIX_INCREMENT) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * SPLITMIX_MULTIPLIER_1) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * SPLITMIX_MULTIPLIER_2) & UINT64_MASK
    return mixed ^ (mixed >> 31)


def _derived_seed(root_seed: int, game_index: int, stream_index: int) -> int:
    ordinal = game_index * STREAMS_PER_GAME + stream_index
    return _splitmix64((root_seed + ordinal) & UINT64_MASK)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """A stable policy label paired with a deterministic seeded factory."""

    name: str
    factory: AgentFactory = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _name(self.name, "name")
        if not callable(self.factory):
            raise TypeError("factory must be callable")

    def build(self, seed: int) -> Agent:
        """Build and validate a fresh policy instance.

        Args:
            seed: Unsigned 64-bit seed assigned by the schedule.

        Returns:
            A runtime-conforming agent with the declared name.

        Raises:
            TypeError: If the seed or returned policy has an invalid type.
            ValueError: If the seed is invalid or the runtime name differs.
        """

        checked_seed = _seed(seed, "seed")
        agent = self.factory(checked_seed)
        if not isinstance(agent, Agent):
            raise TypeError("factory must return an Agent")
        if agent.name != self.name:
            raise ValueError("factory Agent name must match declared name")
        return agent


class AgentActionError(RuntimeError):
    """An illegal arena action attributed to its exact policy and side."""

    def __init__(self, *, agent_name: str, side: PlayerId, action: object) -> None:
        """Record an illegal action with attribution.

        Args:
            agent_name: Declared policy name.
            side: Colour controlled when the violation occurred.
            action: Runtime value returned by the policy.
        """

        self.agent_name = agent_name
        self.side = side
        self.action = action
        super().__init__(
            f"agent {agent_name!r} playing {side.name} returned illegal action {action!r}"
        )


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Replayable actions, completed moves, seeds, and terminal outcome for one game."""

    red_agent: str
    white_agent: str
    red_seed: int
    white_seed: int
    environment_seed: int
    outcome: Outcome
    actions: tuple[int, ...]
    moves: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.red_agent, "red_agent")
        _name(self.white_agent, "white_agent")
        _seed(self.red_seed, "red_seed")
        _seed(self.white_seed, "white_seed")
        _seed(self.environment_seed, "environment_seed")
        if not isinstance(self.outcome, Outcome):
            raise TypeError("outcome must be an Outcome")
        if not isinstance(self.actions, tuple):
            raise TypeError("actions must be a tuple")
        for action in self.actions:
            if isinstance(action, bool) or not isinstance(action, int):
                raise TypeError("each action must be an integer")
            if not 0 <= action < ACTION_COUNT:
                raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")
        if not isinstance(self.moves, tuple):
            raise TypeError("moves must be a tuple")
        if any(not isinstance(move, str) or not move for move in self.moves):
            raise ValueError("each move must be a non-empty notation string")
        if len(self.moves) > len(self.actions):
            raise ValueError("moves cannot outnumber environment-step actions")

    @property
    def steps(self) -> int:
        """Return the number of one-jump/simple-move environment steps."""

        return len(self.actions)

    @property
    def completed_moves(self) -> int:
        """Return the number of completed checkers moves."""

        return len(self.moves)


def _first_agent_won(record: GameRecord, first_agent: str) -> bool:
    winner = record.outcome.winner
    return (winner is PlayerId.RED and record.red_agent == first_agent) or (
        winner is PlayerId.WHITE and record.white_agent == first_agent
    )


def _validate_scheduled_record(
    record: object,
    *,
    first: str,
    second: str,
    root_seed: int,
    game_index: int,
) -> GameRecord:
    if not isinstance(record, GameRecord):
        raise TypeError("records must contain only GameRecord values")
    first_is_red = game_index % 2 == 0
    expected_red = first if first_is_red else second
    expected_white = second if first_is_red else first
    first_seed = _derived_seed(root_seed, game_index, 0)
    second_seed = _derived_seed(root_seed, game_index, 1)
    expected_red_seed = first_seed if first_is_red else second_seed
    expected_white_seed = second_seed if first_is_red else first_seed
    expected_environment_seed = _derived_seed(root_seed, game_index, 2)
    if (
        record.red_agent != expected_red
        or record.white_agent != expected_white
        or record.red_seed != expected_red_seed
        or record.white_seed != expected_white_seed
        or record.environment_seed != expected_environment_seed
    ):
        raise ValueError("record does not match the declared colour/seed schedule")
    return record


def _result_counts(records: tuple[GameRecord, ...], first_agent: str) -> tuple[int, int, int]:
    wins = sum(_first_agent_won(record, first_agent) for record in records)
    draws = sum(record.outcome.winner is None for record in records)
    return wins, draws, len(records) - wins - draws


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Validated colour-balanced records and score from the first policy's view."""

    first_agent: str
    second_agent: str
    seed: int
    initial_state: State
    max_plies: int
    repetition_draws: bool
    confidence: float
    records: tuple[GameRecord, ...]
    score: MatchScore

    def __post_init__(self) -> None:
        first = _name(self.first_agent, "first_agent")
        second = _name(self.second_agent, "second_agent")
        if first == second:
            raise ValueError("match agent names must be distinct")
        root_seed = _seed(self.seed, "seed")
        if not isinstance(self.initial_state, State):
            raise TypeError("initial_state must be a State")
        if self.initial_state.capture_in_progress:
            raise ValueError("initial_state must be a completed-move boundary")
        _positive_integer(self.max_plies, "max_plies")
        if not isinstance(self.repetition_draws, bool):
            raise TypeError("repetition_draws must be bool")
        checked_confidence = _confidence(self.confidence)
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple")
        if not self.records or len(self.records) % 2:
            raise ValueError("records must contain a positive even schedule")
        if not isinstance(self.score, MatchScore):
            raise TypeError("score must be a MatchScore")

        checked_records = tuple(
            _validate_scheduled_record(
                record,
                first=first,
                second=second,
                root_seed=root_seed,
                game_index=game_index,
            )
            for game_index, record in enumerate(self.records)
        )
        wins, draws, losses = _result_counts(checked_records, first)

        expected_score = score_interval(
            wins=wins,
            draws=draws,
            losses=losses,
            confidence=checked_confidence,
        )
        if self.score != expected_score:
            raise ValueError("score does not match recorded outcomes")

    @property
    def games(self) -> int:
        """Return the number of scheduled games."""

        return len(self.records)

    @property
    def wins(self) -> int:
        """Return wins by the first policy."""

        return self.score.wins

    @property
    def draws(self) -> int:
        """Return draws."""

        return self.score.draws

    @property
    def losses(self) -> int:
        """Return losses by the first policy."""

        return self.score.losses

    @property
    def first_as_red_games(self) -> int:
        """Return games in which the first policy controlled Red."""

        return sum(record.red_agent == self.first_agent for record in self.records)


def play_game(  # noqa: PLR0913
    *,
    red: AgentSpec,
    white: AgentSpec,
    red_seed: int,
    white_seed: int,
    environment_seed: int,
    initial_state: State | None = None,
    max_plies: int = DEFAULT_MAX_PLIES,
    repetition_draws: bool = True,
) -> GameRecord:
    """Play one exact game and retain everything needed for transition replay.

    Args:
        red: Factory for the policy controlling Red.
        white: Factory for the policy controlling White.
        red_seed: Independent seed for the Red policy.
        white_seed: Independent seed for the White policy.
        environment_seed: Gymnasium environment seed recorded for replay.
        initial_state: Completed-move opening state; defaults to the standard initial state.
        max_plies: Maximum environment steps before an engine-variant draw.
        repetition_draws: Whether the arena-only threefold rule is enabled.

    Returns:
        Immutable game record with actions, notation, seeds, and outcome.

    Raises:
        TypeError: If specs or configuration values have invalid runtime types.
        ValueError: If configuration values or the initial state are invalid.
        AgentActionError: If a policy returns an illegal action.
    """

    if not isinstance(red, AgentSpec):
        raise TypeError("red must be an AgentSpec")
    if not isinstance(white, AgentSpec):
        raise TypeError("white must be an AgentSpec")
    checked_red_seed = _seed(red_seed, "red_seed")
    checked_white_seed = _seed(white_seed, "white_seed")
    checked_environment_seed = _seed(environment_seed, "environment_seed")
    configured_initial = State.initial() if initial_state is None else initial_state
    environment = CheckersEnv(
        max_plies=max_plies,
        repetition_draws=repetition_draws,
        initial_state=configured_initial,
    )
    red_agent = red.build(checked_red_seed)
    white_agent = white.build(checked_white_seed)
    agents = {PlayerId.RED: red_agent, PlayerId.WHITE: white_agent}
    actions: list[int] = []
    moves: list[str] = []
    try:
        environment.reset(seed=checked_environment_seed)
        while not environment.terminated:
            side = environment.state.side_to_move
            agent = agents[side]
            action = agent.select_action(environment.state)
            try:
                _observation, _reward, _terminated, truncated, info = environment.step(action)
            except IllegalActionError as error:
                raise AgentActionError(
                    agent_name=agent.name,
                    side=side,
                    action=action,
                ) from error
            if truncated:
                raise RuntimeError("checkers environment unexpectedly truncated a game")
            actions.append(action)
            notation = info["checkers_move_san"]
            if notation is not None:
                if not isinstance(notation, str):
                    raise RuntimeError("environment returned non-string move notation")
                moves.append(notation)
        outcome = environment.outcome
        if outcome is None:
            raise RuntimeError("terminated game is missing an outcome")
        return GameRecord(
            red_agent=red.name,
            white_agent=white.name,
            red_seed=checked_red_seed,
            white_seed=checked_white_seed,
            environment_seed=checked_environment_seed,
            outcome=outcome,
            actions=tuple(actions),
            moves=tuple(moves),
        )
    finally:
        environment.close()


def play_balanced_match(  # noqa: PLR0913
    *,
    first: AgentSpec,
    second: AgentSpec,
    games: int,
    seed: int,
    initial_state: State | None = None,
    max_plies: int = DEFAULT_MAX_PLIES,
    repetition_draws: bool = True,
    confidence: float = 0.95,
) -> MatchResult:
    """Play an even schedule with exact alternating colours and independent seeds.

    Args:
        first: Reference policy whose wins/losses define the returned score.
        second: Opposing policy.
        games: Positive even number of games.
        seed: Unsigned 64-bit root seed for disjoint deterministic streams.
        initial_state: Completed-move opening state; defaults to standard play.
        max_plies: Maximum environment steps per game.
        repetition_draws: Whether the arena-only threefold rule is enabled.
        confidence: Confidence level for the returned score interval.

    Returns:
        Validated match records and score from ``first``'s perspective.

    Raises:
        TypeError: If specs or configuration values have invalid runtime types.
        ValueError: If names, schedule size, seeds, or state are invalid.
        AgentActionError: If either policy returns an illegal action.
    """

    if not isinstance(first, AgentSpec):
        raise TypeError("first must be an AgentSpec")
    if not isinstance(second, AgentSpec):
        raise TypeError("second must be an AgentSpec")
    if first.name == second.name:
        raise ValueError("match agent names must be distinct")
    checked_games = _positive_integer(games, "games")
    if checked_games % 2:
        raise ValueError("games must be even for colour balance")
    if checked_games > (UINT64_MASK + 1) // STREAMS_PER_GAME:
        raise ValueError("games exceeds the unique seed-stream capacity")
    root_seed = _seed(seed, "seed")
    checked_confidence = _confidence(confidence)
    configured_initial = State.initial() if initial_state is None else initial_state

    records: list[GameRecord] = []
    for game_index in range(checked_games):
        first_is_red = game_index % 2 == 0
        first_seed = _derived_seed(root_seed, game_index, 0)
        second_seed = _derived_seed(root_seed, game_index, 1)
        records.append(
            play_game(
                red=first if first_is_red else second,
                white=second if first_is_red else first,
                red_seed=first_seed if first_is_red else second_seed,
                white_seed=second_seed if first_is_red else first_seed,
                environment_seed=_derived_seed(root_seed, game_index, 2),
                initial_state=configured_initial,
                max_plies=max_plies,
                repetition_draws=repetition_draws,
            )
        )

    wins = sum(_first_agent_won(record, first.name) for record in records)
    draws = sum(record.outcome.winner is None for record in records)
    losses = checked_games - wins - draws
    score = score_interval(
        wins=wins,
        draws=draws,
        losses=losses,
        confidence=checked_confidence,
    )
    return MatchResult(
        first_agent=first.name,
        second_agent=second.name,
        seed=root_seed,
        initial_state=configured_initial,
        max_plies=max_plies,
        repetition_draws=repetition_draws,
        confidence=checked_confidence,
        records=tuple(records),
        score=score,
    )
