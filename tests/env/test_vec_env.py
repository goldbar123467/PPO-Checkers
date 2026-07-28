"""Synchronous one-environment-step vectorization and checkpoint tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from checkers.env.checkers_env import IllegalActionError
from checkers.env.masking import ACTION_COUNT, step_to_action
from checkers.env.vec_env import VECTOR_ENVIRONMENT_SCHEMA, CheckersVectorEnv
from checkers.rules.moves import Step, legal_steps
from checkers.rules.state import PlayerId, State

VECTOR_SIZE = 2


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def _two_jump_state() -> State:
    return State(
        men=(_mask(9, 11), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _mixed_vector() -> CheckersVectorEnv:
    return CheckersVectorEnv(
        VECTOR_SIZE,
        initial_states=(_two_jump_state(), State.initial()),
    )


def _legal_action(vector: CheckersVectorEnv, index: int, step: Step | None = None) -> int:
    environment = vector.envs[index]
    selected = legal_steps(environment.state)[0] if step is None else step
    return step_to_action(environment.state, selected)


def test_reset_returns_dense_batches_and_per_lane_info() -> None:
    vector = _mixed_vector()

    observations, infos = vector.reset(seed=20260728)

    assert vector.num_envs == VECTOR_SIZE
    assert observations.shape == (VECTOR_SIZE, 8, 8, 8)
    assert observations.dtype == np.float32
    assert len(infos) == VECTOR_SIZE
    assert vector.legal_masks().shape == (VECTOR_SIZE, ACTION_COUNT)
    assert vector.legal_masks().dtype == np.bool_
    assert vector.legal_masks().all(axis=1).tolist() == [False, False]
    assert vector.legal_masks().any(axis=1).tolist() == [True, True]
    assert vector.state_keys().shape == (VECTOR_SIZE,)
    assert vector.state_keys().dtype == np.uint64
    assert all(info["actor"] is PlayerId.RED for info in infos)


def test_default_initial_states_create_standard_games() -> None:
    vector = CheckersVectorEnv(1)

    assert vector.envs[0].state == State.initial()


def test_reset_rejects_noninteger_seed() -> None:
    with pytest.raises(TypeError, match="seed"):
        _mixed_vector().reset(seed=cast(int, True))


def test_vector_step_is_lockstep_by_environment_step_not_checkers_move() -> None:
    vector = _mixed_vector()
    actions = (
        _legal_action(vector, 0, _step(9, 18, 14)),
        _legal_action(vector, 1),
    )

    observations, rewards, terminated, truncated, infos = vector.step(actions)

    assert observations.shape == (VECTOR_SIZE, 8, 8, 8)
    assert rewards.dtype == np.float32
    assert terminated.dtype == np.bool_
    assert truncated.dtype == np.bool_
    assert rewards.tolist() == [0.0, 0.0]
    assert terminated.tolist() == [False, False]
    assert truncated.tolist() == [False, False]
    assert vector.envs[0].state.ply == vector.envs[1].state.ply == 1
    assert vector.envs[0].state.side_to_move is PlayerId.RED
    assert vector.envs[0].state.capture_in_progress is True
    assert vector.envs[1].state.side_to_move is PlayerId.WHITE
    assert infos[0]["move_completed"] is False
    assert infos[1]["move_completed"] is True


def test_numpy_action_batch_is_supported() -> None:
    vector = _mixed_vector()
    actions = np.asarray(
        [_legal_action(vector, 0), _legal_action(vector, 1)],
        dtype=np.int64,
    )

    vector.step(actions)

    assert vector.state_keys().shape == (VECTOR_SIZE,)


def test_bad_action_prevalidation_keeps_every_lane_unmodified() -> None:
    vector = _mixed_vector()
    before_keys = vector.state_keys().copy()
    illegal_second = int(np.flatnonzero(~vector.envs[1].legal_mask())[0])

    with pytest.raises(IllegalActionError, match="lane 1"):
        vector.step((_legal_action(vector, 0), illegal_second))

    assert np.array_equal(vector.state_keys(), before_keys)
    assert all(environment.state.ply == 0 for environment in vector.envs)


def test_terminated_lane_prevalidation_keeps_live_lane_unmodified() -> None:
    terminal_next = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    vector = CheckersVectorEnv(
        VECTOR_SIZE,
        initial_states=(terminal_next, State.initial()),
    )
    vector.step((_legal_action(vector, 0), _legal_action(vector, 1)))
    live_key = vector.envs[1].state_key()

    with pytest.raises(IllegalActionError, match="lane 0"):
        vector.step((0, _legal_action(vector, 1)))

    assert vector.envs[1].state_key() == live_key


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ((0,), "exactly 2"),
        ((0, 1, 2), "exactly 2"),
        (cast(tuple[int, ...], "actions"), "sequence"),
        (np.zeros((2, 1), dtype=np.int64), "one-dimensional"),
    ],
)
def test_vector_step_rejects_invalid_batch_shape(actions: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _mixed_vector().step(cast(tuple[int, ...], actions))


def test_reset_restores_all_lanes_after_different_step_semantics() -> None:
    vector = _mixed_vector()
    initial_keys = vector.state_keys().copy()
    vector.step((_legal_action(vector, 0), _legal_action(vector, 1)))

    observations, infos = vector.reset(seed=9)

    assert np.array_equal(vector.state_keys(), initial_keys)
    assert observations.shape == (VECTOR_SIZE, 8, 8, 8)
    assert all(info["move_completed"] is False for info in infos)


def test_vector_snapshot_round_trips_a_mixed_midsequence_batch() -> None:
    vector = _mixed_vector()
    vector.step(
        (
            _legal_action(vector, 0, _step(9, 18, 14)),
            _legal_action(vector, 1),
        )
    )
    encoded = vector.serialize()

    restored = CheckersVectorEnv.from_serialized(encoded)

    assert restored.serialize() == encoded
    assert np.array_equal(restored.observations(), vector.observations())
    assert np.array_equal(restored.legal_masks(), vector.legal_masks())
    assert np.array_equal(restored.state_keys(), vector.state_keys())
    assert json.loads(encoded)["schema"] == VECTOR_ENVIRONMENT_SCHEMA


def test_vector_snapshot_restores_in_place_when_lane_configs_match() -> None:
    source = _mixed_vector()
    source.step((_legal_action(source, 0), _legal_action(source, 1)))
    target = _mixed_vector()

    target.restore(source.serialize())

    assert target.serialize() == source.serialize()


def test_restored_vector_finishes_partial_capture_with_full_notation() -> None:
    vector = _mixed_vector()
    vector.step(
        (
            _legal_action(vector, 0, _step(9, 18, 14)),
            _legal_action(vector, 1),
        )
    )
    restored = CheckersVectorEnv.from_serialized(vector.serialize())
    actions = (
        _legal_action(restored, 0, _step(18, 25, 22)),
        _legal_action(restored, 1),
    )

    result = restored.step(actions)

    assert result[4][0]["checkers_move_san"] == "9x18x25"
    assert result[4][0]["move_completed"] is True


def test_vector_restore_is_atomic_on_one_bad_lane() -> None:
    vector = _mixed_vector()
    before = vector.serialize()
    payload = json.loads(before)
    payload["environments"][1]["state"] = "invalid"

    with pytest.raises(ValueError, match="lane 1"):
        vector.restore(json.dumps(payload))

    assert vector.serialize() == before


def test_vector_restore_rejects_lane_count_without_mutation() -> None:
    vector = _mixed_vector()
    before = vector.serialize()
    one_lane = CheckersVectorEnv(1).serialize()

    with pytest.raises(ValueError, match="exactly 2"):
        vector.restore(one_lane)

    assert vector.serialize() == before


def test_vector_restore_rejects_valid_but_incompatible_lane_configuration() -> None:
    vector = _mixed_vector()
    before = vector.serialize()
    payload = json.loads(before)
    payload["environments"][1]["max_plies"] = 128

    with pytest.raises(ValueError, match="lane 1.*configuration"):
        vector.restore(json.dumps(payload))

    assert vector.serialize() == before


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CheckersVectorEnv(0), "num_envs"),
        (lambda: CheckersVectorEnv(cast(int, True)), "num_envs"),
        (
            lambda: CheckersVectorEnv(2, initial_states=(State.initial(),)),
            "initial_states",
        ),
        (
            lambda: CheckersVectorEnv(
                1,
                initial_states=cast(tuple[State, ...], [State.initial()]),
            ),
            "tuple",
        ),
        (
            lambda: CheckersVectorEnv(
                1,
                initial_states=(cast(State, "state"),),
            ),
            "initial_states",
        ),
    ],
)
def test_vector_constructor_rejects_invalid_configuration(
    factory: Callable[[], CheckersVectorEnv],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload | {"schema": "CHECKERS_VECTOR_ENV_999"}, "schema"),
        (lambda payload: payload | {"unknown": 1}, "fields"),
        (lambda payload: payload | {"environments": {}}, "list"),
        (lambda payload: payload | {"environments": []}, "at least one"),
        (lambda payload: payload | {"environments": ["invalid"]}, "lane 0"),
    ],
)
def test_vector_snapshot_parser_rejects_invalid_payloads(
    mutation: Callable[[dict[str, object]], dict[str, object]],
    message: str,
) -> None:
    payload = json.loads(_mixed_vector().serialize())
    with pytest.raises(ValueError, match=message):
        CheckersVectorEnv.from_serialized(json.dumps(mutation(payload)))


@pytest.mark.parametrize(("text", "error"), [("{", "JSON"), ("[]", "object")])
def test_vector_snapshot_rejects_bad_top_level_json(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        CheckersVectorEnv.from_serialized(text)


def test_vector_snapshot_requires_text() -> None:
    with pytest.raises(TypeError, match="text"):
        CheckersVectorEnv.from_serialized(cast(str, 7))
