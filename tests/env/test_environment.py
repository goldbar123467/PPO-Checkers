"""Gymnasium lifecycle, reward, terminal, and rendering contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.passive_env_checker import (
    env_reset_passive_checker,
    env_step_passive_checker,
)

from checkers.env.checkers_env import CheckersEnv, IllegalActionError
from checkers.env.encoding import BOARD_SIZE, OBSERVATION_PLANES
from checkers.env.masking import ACTION_COUNT, step_to_action
from checkers.rules.moves import Step, legal_steps
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome, TerminationReason
from checkers.rules.zobrist import position_key, state_key

INFO_KEYS = {
    "legal_mask",
    "actor",
    "move_completed",
    "checkers_move_san",
    "outcome",
}
MAX_PLIES = 512
JUMP_SEQUENCE_PLIES = 2


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def _action(env: CheckersEnv, step: Step) -> int:
    return step_to_action(env.state, step)


def _two_jump_state(*, ply: int = 0) -> State:
    return State(
        men=(_mask(9, 11), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        no_progress=(17, 39),
        ply=ply,
    )


def _cycle_state() -> State:
    return State(
        men=(0, 0),
        kings=(_mask(14), _mask(24)),
        side_to_move=PlayerId.RED,
    )


def test_reset_exposes_exact_gymnasium_spaces_and_info_contract() -> None:
    env = CheckersEnv()

    observation, info = env.reset(seed=20260728)

    assert isinstance(env, gym.Env)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == ACTION_COUNT
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert env.observation_space.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert set(info) == INFO_KEYS
    assert info["actor"] is PlayerId.RED
    assert info["move_completed"] is False
    assert info["checkers_move_san"] is None
    assert info["outcome"] is None
    assert np.array_equal(info["legal_mask"], env.legal_mask())
    assert info["legal_mask"] is not env.legal_mask()


def test_default_environment_passes_gymnasium_passive_api_checks() -> None:
    env = CheckersEnv()
    reset_checker = cast(Callable[[object], object], env_reset_passive_checker)
    step_checker = cast(Callable[[object, object], object], env_step_passive_checker)
    reset_checker(env)
    legal_action = int(np.flatnonzero(env.legal_mask())[0])
    step_checker(env, legal_action)


def test_reset_restarts_configured_boundary_state_and_rejects_options() -> None:
    initial = _cycle_state()
    env = CheckersEnv(initial_state=initial)
    action = _action(env, _step(14, 17))
    env.step(action)

    observation, info = env.reset(seed=7)

    assert env.state == initial
    assert observation[7, 0, 0] == 0.0
    assert info["actor"] is PlayerId.RED
    with pytest.raises(ValueError, match="options"):
        env.reset(options={"state": "unsupported"})
    with pytest.raises(TypeError, match="options"):
        env.reset(options=cast(dict[str, Any], []))


def test_simple_step_returns_actor_completed_move_and_acf_notation() -> None:
    env = CheckersEnv()
    before_key = env.state_key()
    step = legal_steps(env.state)[0]

    observation, reward, terminated, truncated, info = env.step(_action(env, step))

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["actor"] is PlayerId.RED
    assert info["move_completed"] is True
    assert info["checkers_move_san"] == f"{step.origin + 1}-{step.destination + 1}"
    assert info["outcome"] is None
    assert int(env.state.side_to_move) == int(PlayerId.WHITE)
    assert env.state.ply == 1
    assert env.state_key() == state_key(env.state)
    assert env.state_key() != before_key
    assert env.position_key() == position_key(env.state)
    assert env.observation_space.contains(observation)
    assert np.array_equal(info["legal_mask"], env.legal_mask())


def test_multijump_keeps_actor_and_emits_notation_only_at_move_boundary() -> None:
    env = CheckersEnv(initial_state=_two_jump_state())

    first = env.step(_action(env, _step(9, 18, 14)))

    assert first[1:4] == (0.0, False, False)
    assert first[4]["actor"] is PlayerId.RED
    assert first[4]["move_completed"] is False
    assert first[4]["checkers_move_san"] is None
    assert env.state.side_to_move is PlayerId.RED
    assert env.state.capture_in_progress is True
    assert env.state.no_progress == (17, 39)
    assert env.state.ply == 1
    with pytest.raises(ValueError, match="move boundary"):
        env.position_key()

    second = env.step(_action(env, _step(18, 25, 22)))

    assert second[1:4] == (0.0, False, False)
    assert second[4]["actor"] is PlayerId.RED
    assert second[4]["move_completed"] is True
    assert second[4]["checkers_move_san"] == "9x18x25"
    assert int(env.state.side_to_move) == int(PlayerId.WHITE)
    assert env.state.capture_in_progress is False
    assert env.state.no_progress == (0, 39)
    assert env.state.ply == JUMP_SEQUENCE_PLIES


def test_promotion_ends_capture_sequence_immediately() -> None:
    state = State(
        men=(_mask(21), _mask(25, 26)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    env = CheckersEnv(initial_state=state)

    result = env.step(_action(env, _step(21, 30, 25)))

    assert result[4]["move_completed"] is True
    assert result[4]["checkers_move_san"] == "21x30"
    assert env.state.kings[PlayerId.RED] & _mask(30)
    assert env.state.side_to_move is PlayerId.WHITE


def test_capture_of_last_piece_terminates_with_actor_win_reward() -> None:
    state = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    env = CheckersEnv(initial_state=state)

    observation, reward, terminated, truncated, info = env.step(_action(env, _step(9, 18, 14)))

    expected = Outcome(winner=PlayerId.RED, reason=TerminationReason.NO_PIECES)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == expected
    assert info["actor"] is PlayerId.RED
    assert not info["legal_mask"].any()
    assert not env.legal_mask().any()
    assert env.outcome == expected
    assert env.terminated is True
    assert env.observation_space.contains(observation)
    with pytest.raises(IllegalActionError, match="terminated"):
        env.step(0)


@pytest.mark.parametrize(
    ("state", "step", "reason"),
    [
        (
            State(
                men=(0, 0),
                kings=(_mask(14), _mask(24)),
                side_to_move=PlayerId.WHITE,
                no_progress=(40, 39),
            ),
            _step(24, 20),
            TerminationReason.NO_PROGRESS,
        ),
        (
            State(
                men=(0, 0),
                kings=(_mask(14), _mask(24)),
                side_to_move=PlayerId.RED,
                ply=MAX_PLIES - 1,
            ),
            _step(14, 17),
            TerminationReason.PLY_CAP,
        ),
    ],
)
def test_game_rule_draws_are_terminated_never_truncated(
    state: State,
    step: Step,
    reason: TerminationReason,
) -> None:
    env = CheckersEnv(initial_state=state)

    _observation, reward, terminated, truncated, info = env.step(_action(env, step))

    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == Outcome(winner=None, reason=reason)


def test_optional_repetition_counts_only_completed_move_boundaries() -> None:
    env = CheckersEnv(initial_state=_cycle_state(), repetition_draws=True)
    cycle = ((_step(14, 17)), (_step(24, 20)), (_step(17, 14)), (_step(20, 24)))

    for index, step in enumerate(cycle * 2):
        _observation, reward, terminated, truncated, info = env.step(_action(env, step))
        assert truncated is False
        if index < len(cycle * 2) - 1:
            assert terminated is False
            assert reward == 0.0
            assert info["outcome"] is None

    assert terminated is True
    assert reward == 0.0
    assert info["outcome"] == Outcome(winner=None, reason=TerminationReason.REPETITION)


def test_repetition_cycle_does_not_terminate_when_arena_rule_is_disabled() -> None:
    env = CheckersEnv(initial_state=_cycle_state(), repetition_draws=False)
    cycle = (_step(14, 17), _step(24, 20), _step(17, 14), _step(20, 24))

    for step in cycle * 2:
        _observation, _reward, terminated, truncated, info = env.step(_action(env, step))

    assert terminated is False
    assert truncated is False
    assert info["outcome"] is None


@pytest.mark.parametrize("action", [-1, ACTION_COUNT, True, "0", None])
def test_illegal_action_always_raises_environment_exception(action: object) -> None:
    env = CheckersEnv()
    with pytest.raises(IllegalActionError, match="illegal action"):
        env.step(cast(int, action))


def test_in_range_masked_action_raises_without_mutating_state() -> None:
    env = CheckersEnv()
    before = env.state
    illegal = int(np.flatnonzero(~env.legal_mask())[0])

    with pytest.raises(IllegalActionError, match="illegal action"):
        env.step(illegal)

    assert env.state == before
    assert env.state_key() == state_key(before)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CheckersEnv(max_plies=0), "max_plies"),
        (lambda: CheckersEnv(max_plies=cast(int, True)), "max_plies"),
        (
            lambda: CheckersEnv(repetition_draws=cast(bool, 1)),
            "repetition_draws",
        ),
        (lambda: CheckersEnv(initial_state=cast(State, "state")), "initial_state"),
        (
            lambda: CheckersEnv(initial_state=_midsequence_state()),
            "move boundary",
        ),
        (lambda: CheckersEnv(render_mode="rgb_array"), "render_mode"),
    ],
)
def test_constructor_rejects_invalid_configuration(
    factory: Callable[[], CheckersEnv],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_render_contains_every_acf_number_and_piece_legend() -> None:
    env = CheckersEnv(render_mode="ansi")
    rendered = env.render()

    assert isinstance(rendered, str)
    assert "actor=RED" in rendered
    assert "r/R=Red man/king" in rendered
    for acf_square in range(1, 33):
        assert f"{acf_square:02d}:" in rendered
    assert env.render("ansi") == rendered
    with pytest.raises(ValueError, match="ansi"):
        env.render("rgb_array")


def test_render_marks_pending_capture_and_forced_piece() -> None:
    env = CheckersEnv(initial_state=_two_jump_state())
    env.step(_action(env, _step(9, 18, 14)))

    rendered = env.render("ansi")

    assert "14:w*" in rendered
    assert "18:r@" in rendered


def test_render_distinguishes_both_colours_of_king() -> None:
    state = State(
        men=(0, 0),
        kings=(_mask(14), _mask(24)),
        side_to_move=PlayerId.RED,
    )
    rendered = CheckersEnv(initial_state=state).render("ansi")

    assert "14:R " in rendered
    assert "24:W " in rendered


def _midsequence_state() -> State:
    env = CheckersEnv(initial_state=_two_jump_state())
    env.step(_action(env, _step(9, 18, 14)))
    return env.state
