"""Deterministic colour-balanced arena scheduling and replay records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from checkers.agents.base import Agent
from checkers.env.checkers_env import CheckersEnv, StepResult
from checkers.env.encoding import DEFAULT_MAX_PLIES
from checkers.env.masking import legal_action_map, step_to_action
from checkers.eval.arena import (
    AgentActionError,
    AgentSpec,
    GameRecord,
    MatchResult,
    play_balanced_match,
    play_game,
)
from checkers.eval.power import MatchScore, score_interval
from checkers.rules.moves import Step
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome, TerminationReason

FORCED_CAPTURE_ACTION = 34
MATCH_GAMES = 4
MAX_SEED = (1 << 64) - 1
TWO_JUMP_STEPS = 2
COLOUR_BALANCED_HALF = MATCH_GAMES // 2
TOO_MANY_GAMES = ((1 << 64) // 3) + 1


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


class FirstLegalAgent:
    """Tiny deterministic test policy that exposes its call count."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def select_action(self, state: State) -> int:
        self.calls += 1
        return next(iter(legal_action_map(state)))


class IllegalAgent:
    """Test double that always violates the action contract."""

    name = "illegal"

    def select_action(self, state: State) -> int:
        del state
        return -1


def _first_spec(name: str) -> AgentSpec:
    return AgentSpec(name=name, factory=lambda _seed: FirstLegalAgent(name))


def _forced_red_win_state() -> State:
    return State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _forced_two_jump_win_state() -> State:
    return State(
        men=(_mask(9), _mask(14, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _terminal_state() -> State:
    return State(
        men=(_mask(9), 0),
        kings=(0, 0),
        side_to_move=PlayerId.WHITE,
    )


def _test_match(
    *,
    games: int = 2,
    seed: int = 0,
    confidence: float = 0.95,
    max_plies: int = DEFAULT_MAX_PLIES,
    repetition_draws: bool = True,
) -> MatchResult:
    return play_balanced_match(
        first=_first_spec("first"),
        second=_first_spec("second"),
        games=games,
        seed=seed,
        confidence=confidence,
        max_plies=max_plies,
        repetition_draws=repetition_draws,
        initial_state=_forced_red_win_state(),
    )


def _play_with_specs(red: AgentSpec, white: AgentSpec) -> GameRecord:
    return play_game(
        red=red,
        white=white,
        red_seed=1,
        white_seed=2,
        environment_seed=3,
        initial_state=_forced_red_win_state(),
    )


def _match_with_specs(first: AgentSpec, second: AgentSpec) -> MatchResult:
    return play_balanced_match(
        first=first,
        second=second,
        games=2,
        seed=0,
        initial_state=_forced_red_win_state(),
    )


def _valid_game_record() -> GameRecord:
    return GameRecord(
        red_agent="first",
        white_agent="second",
        red_seed=1,
        white_seed=2,
        environment_seed=3,
        outcome=Outcome(
            winner=PlayerId.RED,
            reason=TerminationReason.NO_PIECES,
        ),
        actions=(FORCED_CAPTURE_ACTION,),
        moves=("9x18",),
    )


def test_agent_spec_passes_exact_seed_and_checks_runtime_name() -> None:
    seen: list[int] = []

    def factory(seed: int) -> Agent:
        seen.append(seed)
        return FirstLegalAgent("first")

    spec = AgentSpec(name="first", factory=factory)
    agent = spec.build(42)

    assert agent.name == "first"
    assert seen == [42]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: AgentSpec(name="", factory=lambda _seed: FirstLegalAgent("x")), "name"),
        (
            lambda: AgentSpec(
                name=cast(str, 1),
                factory=lambda _seed: FirstLegalAgent("x"),
            ),
            "name",
        ),
        (lambda: AgentSpec(name="x", factory=cast(Callable[[int], Agent], 1)), "factory"),
        (
            lambda: AgentSpec(
                name="x",
                factory=lambda _seed: cast(Agent, object()),
            ).build(0),
            "Agent",
        ),
        (
            lambda: AgentSpec(
                name="declared",
                factory=lambda _seed: FirstLegalAgent("actual"),
            ).build(0),
            "name",
        ),
        (lambda: _first_spec("x").build(cast(int, True)), "seed"),
        (lambda: _first_spec("x").build(-1), "seed"),
        (lambda: _first_spec("x").build(MAX_SEED + 1), "seed"),
    ],
)
def test_agent_spec_rejects_ambiguous_or_invalid_factories(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_play_game_records_forced_capture_and_terminal_reason() -> None:
    record = play_game(
        red=_first_spec("red-policy"),
        white=_first_spec("white-policy"),
        red_seed=11,
        white_seed=12,
        environment_seed=13,
        initial_state=_forced_red_win_state(),
    )

    expected_action = step_to_action(
        _forced_red_win_state(),
        Step(origin=8, destination=17, captured=13),
    )
    assert record.outcome == Outcome(
        winner=PlayerId.RED,
        reason=TerminationReason.NO_PIECES,
    )
    assert record.actions == (expected_action,)
    assert record.moves == ("9x18",)
    assert record.steps == 1
    assert record.completed_moves == 1
    assert record.red_agent == "red-policy"
    assert record.white_agent == "white-policy"


def test_play_game_keeps_multi_jump_as_one_completed_move() -> None:
    record = play_game(
        red=_first_spec("red-policy"),
        white=_first_spec("white-policy"),
        red_seed=1,
        white_seed=2,
        environment_seed=3,
        initial_state=_forced_two_jump_win_state(),
    )

    assert record.outcome.winner is PlayerId.RED
    assert record.actions == (
        step_to_action(
            _forced_two_jump_win_state(),
            Step(origin=8, destination=17, captured=13),
        ),
        step_to_action(
            State(
                men=(_mask(18), _mask(14, 22)),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                capture_in_progress=True,
                moving_square=17,
                sequence_origin=8,
                captured_pending=_mask(14),
                ply=1,
            ),
            Step(origin=17, destination=24, captured=21),
        ),
    )
    assert record.moves == ("9x18x25",)
    assert record.steps == TWO_JUMP_STEPS
    assert record.completed_moves == 1


def test_play_game_handles_already_terminal_initial_state_without_action() -> None:
    created: list[FirstLegalAgent] = []

    def factory(seed: int) -> Agent:
        del seed
        agent = FirstLegalAgent("policy")
        created.append(agent)
        return agent

    spec = AgentSpec(name="policy", factory=factory)
    record = play_game(
        red=spec,
        white=spec,
        red_seed=1,
        white_seed=2,
        environment_seed=3,
        initial_state=_terminal_state(),
    )

    assert record.outcome.winner is PlayerId.RED
    assert record.actions == ()
    assert record.moves == ()
    assert [agent.calls for agent in created] == [0, 0]


def test_play_game_records_ply_cap_draw_exactly() -> None:
    record = play_game(
        red=_first_spec("red-policy"),
        white=_first_spec("white-policy"),
        red_seed=1,
        white_seed=2,
        environment_seed=3,
        initial_state=State.initial(),
        max_plies=1,
    )

    assert record.outcome == Outcome(
        winner=None,
        reason=TerminationReason.PLY_CAP,
    )
    assert record.steps == 1


def test_play_game_attributes_an_illegal_action_to_the_policy() -> None:
    with pytest.raises(AgentActionError, match="illegal.*-1") as caught:
        play_game(
            red=AgentSpec(name="illegal", factory=lambda _seed: IllegalAgent()),
            white=_first_spec("white-policy"),
            red_seed=1,
            white_seed=2,
            environment_seed=3,
            initial_state=_forced_red_win_state(),
        )

    assert caught.value.agent_name == "illegal"
    assert caught.value.action == -1
    assert caught.value.side is PlayerId.RED


def test_balanced_match_alternates_colours_and_scores_for_first_agent() -> None:
    result = play_balanced_match(
        first=_first_spec("first"),
        second=_first_spec("second"),
        games=MATCH_GAMES,
        seed=20260728,
        initial_state=_forced_red_win_state(),
    )

    assert result.games == MATCH_GAMES
    assert [record.red_agent for record in result.records] == [
        "first",
        "second",
        "first",
        "second",
    ]
    assert result.score == score_interval(
        wins=COLOUR_BALANCED_HALF,
        draws=0,
        losses=COLOUR_BALANCED_HALF,
    )
    assert result.wins == COLOUR_BALANCED_HALF
    assert result.draws == 0
    assert result.losses == COLOUR_BALANCED_HALF
    assert result.first_as_red_games == COLOUR_BALANCED_HALF


def test_balanced_match_uses_independent_reproducible_seed_streams() -> None:
    first = _test_match(games=MATCH_GAMES, seed=99)
    replay = _test_match(games=MATCH_GAMES, seed=99)

    assert first == replay
    all_seeds = {
        seed
        for record in first.records
        for seed in (record.red_seed, record.white_seed, record.environment_seed)
    }
    assert len(all_seeds) == MATCH_GAMES * 3
    assert _test_match(games=MATCH_GAMES, seed=100) != first


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: _test_match(games=0), "games"),
        (lambda: _test_match(games=3), "even"),
        (lambda: _test_match(games=cast(int, True)), "games"),
        (lambda: _test_match(seed=-1), "seed"),
        (lambda: _test_match(seed=cast(int, True)), "seed"),
        (lambda: _test_match(confidence=1.0), "confidence"),
        (lambda: _test_match(max_plies=0), "max_plies"),
        (
            lambda: _test_match(repetition_draws=cast(bool, 1)),
            "repetition_draws",
        ),
    ],
)
def test_balanced_match_rejects_invalid_schedule_values(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_balanced_match_requires_distinct_declared_agent_names() -> None:
    with pytest.raises(ValueError, match="distinct"):
        play_balanced_match(
            first=_first_spec("same"),
            second=_first_spec("same"),
            games=2,
            seed=0,
            initial_state=_forced_red_win_state(),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(_valid_game_record(), red_agent=""), "red_agent"),
        (lambda: replace(_valid_game_record(), red_seed=-1), "red_seed"),
        (
            lambda: replace(
                _valid_game_record(),
                outcome=cast(Outcome, "red wins"),
            ),
            "outcome",
        ),
        (lambda: replace(_valid_game_record(), actions=cast(tuple[int, ...], [])), "actions"),
        (
            lambda: replace(
                _valid_game_record(),
                actions=(cast(int, True),),
            ),
            "action",
        ),
        (lambda: replace(_valid_game_record(), actions=(128,)), "action"),
        (
            lambda: replace(
                _valid_game_record(),
                moves=cast(tuple[str, ...], ["9x18"]),
            ),
            "moves",
        ),
        (lambda: replace(_valid_game_record(), moves=("",)), "move"),
        (lambda: replace(_valid_game_record(), moves=("9-13", "13-17")), "moves"),
    ],
)
def test_game_records_reject_corruption(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_match_result_rejects_aggregate_corruption() -> None:
    valid = play_balanced_match(
        first=_first_spec("first"),
        second=_first_spec("second"),
        games=2,
        seed=0,
        initial_state=_forced_red_win_state(),
    )
    wrong_score = MatchScore(
        wins=0,
        draws=2,
        losses=0,
        score=0.5,
        low=0.1,
        high=0.9,
        confidence=0.95,
    )

    with pytest.raises(ValueError, match="score"):
        replace(valid, score=wrong_score)
    with pytest.raises(ValueError, match="schedule"):
        replace(valid, records=tuple(reversed(valid.records)))
    with pytest.raises(TypeError, match="records"):
        replace(valid, records=cast(tuple[GameRecord, ...], list(valid.records)))
    with pytest.raises(ValueError, match="distinct"):
        replace(valid, second_agent="first")
    with pytest.raises(TypeError, match="initial_state"):
        replace(valid, initial_state=cast(State, "state"))
    with pytest.raises(ValueError, match="boundary"):
        replace(
            valid,
            initial_state=State(
                men=(_mask(18), _mask(14, 22)),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                capture_in_progress=True,
                moving_square=17,
                sequence_origin=8,
                captured_pending=_mask(14),
                ply=1,
            ),
        )
    with pytest.raises(TypeError, match="repetition_draws"):
        replace(valid, repetition_draws=cast(bool, 1))
    with pytest.raises(ValueError, match="records"):
        replace(valid, records=())
    with pytest.raises(TypeError, match="GameRecord"):
        replace(
            valid,
            records=(cast(GameRecord, "record"), valid.records[1]),
        )
    with pytest.raises(TypeError, match="MatchScore"):
        replace(valid, score=cast(MatchScore, "score"))


def test_match_result_public_type_is_immutable_record() -> None:
    result = play_balanced_match(
        first=_first_spec("first"),
        second=_first_spec("second"),
        games=2,
        seed=0,
        initial_state=_forced_red_win_state(),
    )

    assert isinstance(result, MatchResult)
    with pytest.raises((AttributeError, TypeError)):
        result.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _play_with_specs(
                cast(AgentSpec, "red"),
                _first_spec("white"),
            ),
            "red",
        ),
        (
            lambda: _play_with_specs(
                _first_spec("red"),
                cast(AgentSpec, "white"),
            ),
            "white",
        ),
        (
            lambda: _match_with_specs(
                cast(AgentSpec, "first"),
                _first_spec("second"),
            ),
            "first",
        ),
        (
            lambda: _match_with_specs(
                _first_spec("first"),
                cast(AgentSpec, "second"),
            ),
            "second",
        ),
        (lambda: _test_match(confidence=cast(float, "confidence")), "confidence"),
        (lambda: _test_match(games=TOO_MANY_GAMES), "capacity"),
    ],
)
def test_arena_public_functions_reject_invalid_runtime_types(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_play_game_rejects_unexpected_environment_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_step = CheckersEnv.step

    def truncated_step(self: CheckersEnv, action: int) -> StepResult:
        observation, reward, terminated, _truncated, info = original_step(self, action)
        return observation, reward, terminated, True, info

    monkeypatch.setattr(CheckersEnv, "step", truncated_step)

    with pytest.raises(RuntimeError, match="truncated"):
        _play_with_specs(_first_spec("red"), _first_spec("white"))


def test_play_game_rejects_non_string_environment_notation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_step = CheckersEnv.step

    def malformed_step(self: CheckersEnv, action: int) -> StepResult:
        observation, reward, terminated, truncated, info = original_step(self, action)
        malformed_info: dict[str, Any] = dict(info)
        malformed_info["checkers_move_san"] = 9
        return observation, reward, terminated, truncated, malformed_info

    monkeypatch.setattr(CheckersEnv, "step", malformed_step)

    with pytest.raises(RuntimeError, match="notation"):
        _play_with_specs(_first_spec("red"), _first_spec("white"))


def test_play_game_rejects_terminated_environment_without_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CheckersEnv, "outcome", property(lambda _self: None))

    with pytest.raises(RuntimeError, match="missing an outcome"):
        _play_with_specs(_first_spec("red"), _first_spec("white"))
