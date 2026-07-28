"""Full environment snapshot and mid-sequence restore tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from checkers.env.checkers_env import CheckersEnv
from checkers.env.masking import step_to_action
from checkers.env.serialize import (
    ENVIRONMENT_SCHEMA,
    EnvironmentSnapshot,
    parse_environment_snapshot,
    serialize_environment_snapshot,
)
from checkers.rules.moves import Step
from checkers.rules.notation import serialize_state
from checkers.rules.state import PlayerId, State
from checkers.rules.zobrist import position_key


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


def _midsequence_environment() -> CheckersEnv:
    env = CheckersEnv(initial_state=_two_jump_state())
    env.step(step_to_action(env.state, _step(9, 18, 14)))
    return env


def _boundary_snapshot() -> EnvironmentSnapshot:
    state = State.initial()
    return EnvironmentSnapshot(
        state=state,
        initial_state=state,
        max_plies=512,
        repetition_draws=False,
        position_counts=((position_key(state), 1),),
        active_move_squares=(),
    )


def test_snapshot_codec_is_canonical_and_round_trips_all_fields() -> None:
    initial = State.initial()
    snapshot = EnvironmentSnapshot(
        state=initial,
        initial_state=initial,
        max_plies=512,
        repetition_draws=True,
        position_counts=((position_key(initial), 2),),
        active_move_squares=(),
    )

    encoded = serialize_environment_snapshot(snapshot)
    decoded = parse_environment_snapshot(encoded)

    assert decoded == snapshot
    assert serialize_environment_snapshot(decoded) == encoded
    assert json.loads(encoded)["schema"] == ENVIRONMENT_SCHEMA
    assert serialize_state(initial) in encoded


def test_midsequence_restore_reproduces_observation_mask_key_and_render() -> None:
    original = _midsequence_environment()
    encoded = original.serialize()
    restored = CheckersEnv()

    restored.restore(encoded)

    assert restored.state == original.state
    assert restored.state_key() == original.state_key()
    assert np.array_equal(restored.legal_mask(), original.legal_mask())
    assert np.array_equal(restored.observe(), original.observe())
    assert restored.render("ansi") == original.render("ansi")
    assert restored.serialize() == encoded


def test_restored_midsequence_finishes_with_complete_original_move_notation() -> None:
    original = _midsequence_environment()
    restored = CheckersEnv.from_serialized(original.serialize())
    action = step_to_action(original.state, _step(18, 25, 22))

    original_result = original.step(action)
    restored_result = restored.step(action)

    assert original_result[1:4] == restored_result[1:4]
    assert original_result[4]["actor"] == restored_result[4]["actor"]
    assert original_result[4]["move_completed"] == restored_result[4]["move_completed"]
    assert original_result[4]["checkers_move_san"] == restored_result[4]["checkers_move_san"]
    assert original_result[4]["outcome"] == restored_result[4]["outcome"]
    assert np.array_equal(
        original_result[4]["legal_mask"],
        restored_result[4]["legal_mask"],
    )
    assert np.array_equal(original_result[0], restored_result[0])
    assert restored_result[4]["checkers_move_san"] == "9x18x25"


def test_restore_is_atomic_when_snapshot_is_invalid() -> None:
    env = CheckersEnv()
    before = env.serialize()
    malformed = json.loads(before)
    malformed["state"] = "not-a-state"

    with pytest.raises(ValueError, match="snapshot"):
        env.restore(json.dumps(malformed))

    assert env.serialize() == before


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload | {"schema": "CHECKERS_ENV_999"}, "schema"),
        (lambda payload: payload | {"unknown": 1}, "fields"),
        (lambda payload: payload | {"max_plies": 0}, "max_plies"),
        (lambda payload: payload | {"repetition_draws": 1}, "repetition_draws"),
        (lambda payload: payload | {"position_counts": []}, "position"),
        (lambda payload: payload | {"active_move_squares": [1]}, "active move"),
    ],
)
def test_snapshot_parser_rejects_invalid_records(
    mutator: Callable[[dict[str, object]], dict[str, object]],
    message: str,
) -> None:
    payload = json.loads(CheckersEnv().serialize())
    changed = mutator(payload)
    with pytest.raises(ValueError, match=message):
        parse_environment_snapshot(json.dumps(changed))


def test_snapshot_rejects_capture_state_without_matching_active_path() -> None:
    env = _midsequence_environment()
    snapshot = parse_environment_snapshot(env.serialize())

    with pytest.raises(ValueError, match="active move"):
        replace(snapshot, active_move_squares=(snapshot.active_move_squares[0],))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda snapshot: replace(snapshot, state=cast(State, "state")), "snapshot state"),
        (
            lambda snapshot: replace(snapshot, initial_state=cast(State, "state")),
            "initial_state",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                initial_state=_midsequence_environment().state,
            ),
            "move boundary",
        ),
        (lambda snapshot: replace(snapshot, max_plies=cast(int, True)), "max_plies"),
        (
            lambda snapshot: replace(
                snapshot,
                position_counts=cast(tuple[tuple[int, int], ...], []),
            ),
            "tuple",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                position_counts=cast(tuple[tuple[int, int], ...], ((1,),)),
            ),
            "pair",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                position_counts=((cast(int, "key"), 1),),
            ),
            "position key",
        ),
        (
            lambda snapshot: replace(snapshot, position_counts=((-1, 1),)),
            "uint64",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                position_counts=((snapshot.position_counts[0][0], 0),),
            ),
            "positive",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                position_counts=(snapshot.position_counts[0], snapshot.position_counts[0]),
            ),
            "unique",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                position_counts=((snapshot.position_counts[0][0] ^ 1, 1),),
            ),
            "current position",
        ),
        (
            lambda snapshot: replace(
                snapshot,
                active_move_squares=cast(tuple[int, ...], []),
            ),
            "tuple",
        ),
        (
            lambda snapshot: replace(snapshot, active_move_squares=(cast(int, True),)),
            "integers",
        ),
        (
            lambda snapshot: replace(snapshot, active_move_squares=(32,)),
            r"\[0, 31\]",
        ),
        (
            lambda snapshot: replace(snapshot, active_move_squares=(0, 1)),
            "move boundary",
        ),
    ],
)
def test_snapshot_constructor_rejects_invalid_cross_field_state(
    mutation: Callable[[EnvironmentSnapshot], EnvironmentSnapshot],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        mutation(_boundary_snapshot())


@pytest.mark.parametrize(
    ("active_path", "message"),
    [
        ((7, 17), "origin"),
        ((8, 16), "landing"),
    ],
)
def test_midsequence_snapshot_path_must_match_state_endpoints(
    active_path: tuple[int, ...],
    message: str,
) -> None:
    snapshot = parse_environment_snapshot(_midsequence_environment().serialize())
    with pytest.raises(ValueError, match=message):
        replace(snapshot, active_move_squares=active_path)


def test_snapshot_public_functions_reject_wrong_runtime_types() -> None:
    with pytest.raises(TypeError, match="EnvironmentSnapshot"):
        serialize_environment_snapshot(cast(EnvironmentSnapshot, "snapshot"))
    with pytest.raises(TypeError, match="text"):
        parse_environment_snapshot(cast(str, 7))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "JSON"),
        ("[]", "JSON object"),
    ],
)
def test_snapshot_parser_rejects_malformed_or_nonobject_json(
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_environment_snapshot(text)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("position_counts", {}, "list"),
        ("position_counts", [{}], "fields"),
        (
            "position_counts",
            [{"key": "NOT-A-KEY", "count": 1}],
            "hexadecimal",
        ),
        (
            "position_counts",
            [{"key": "0000000000000000", "count": 0}],
            "positive",
        ),
        ("active_move_squares", {}, "list"),
    ],
)
def test_snapshot_parser_rejects_malformed_nested_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(CheckersEnv().serialize())
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        parse_environment_snapshot(json.dumps(payload))


def test_restore_rejects_snapshot_configuration_mismatch() -> None:
    encoded = CheckersEnv(max_plies=128, repetition_draws=True).serialize()
    env = CheckersEnv(max_plies=512, repetition_draws=False)

    with pytest.raises(ValueError, match="configuration"):
        env.restore(encoded)
