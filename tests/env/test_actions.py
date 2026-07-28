"""Canonical 128-action encoding and legal-mask tests."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from checkers.env import masking
from checkers.env.masking import (
    ACTION_COUNT,
    DIRECTIONS_PER_SQUARE,
    ActionEncodingError,
    action_to_step,
    decode_action_id,
    encode_action_id,
    legal_action_map,
    legal_action_mask,
    step_to_action,
)
from checkers.rules.board import rotate_square
from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.state import PlayerId, State

MAX_PLIES = 512
PROMOTION_SQUARE = 29


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def test_action_id_bijection_exhaustively_covers_128_ids() -> None:
    encoded = {
        encode_action_id(square, direction)
        for square in range(32)
        for direction in range(DIRECTIONS_PER_SQUARE)
    }
    assert encoded == set(range(ACTION_COUNT))
    for action in range(ACTION_COUNT):
        square, direction = decode_action_id(action)
        assert encode_action_id(square, direction) == action


@pytest.mark.parametrize(
    ("factory", "error_type"),
    [
        (lambda: encode_action_id(-1, 0), ValueError),
        (lambda: encode_action_id(32, 0), ValueError),
        (lambda: encode_action_id(0, -1), ValueError),
        (lambda: encode_action_id(0, 4), ValueError),
        (lambda: decode_action_id(-1), ValueError),
        (lambda: decode_action_id(ACTION_COUNT), ValueError),
        (lambda: decode_action_id(True), TypeError),
        (lambda: encode_action_id(cast(int, "0"), 0), TypeError),
        (lambda: encode_action_id(0, cast(int, np.bool_(True))), TypeError),
    ],
)
def test_action_components_reject_invalid_runtime_values(
    factory: Callable[[], object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        factory()


def test_numpy_integer_action_is_accepted_for_gymnasium_interoperability() -> None:
    assert decode_action_id(cast(int, np.int64(127))) == (31, 3)


@pytest.mark.parametrize(
    "state",
    [
        State.initial(),
        State(
            men=(_mask(9), _mask(14, 22)),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        ),
        State(
            men=(0, _mask(10, 18)),
            kings=(_mask(14), 0),
            side_to_move=PlayerId.RED,
        ),
    ],
)
def test_every_legal_step_round_trips_without_action_collision(state: State) -> None:
    steps = legal_steps(state)
    actions = tuple(step_to_action(state, step) for step in steps)
    assert len(actions) == len(set(actions))
    assert tuple(action_to_step(state, action) for action in actions) == steps
    assert legal_action_map(state) == dict(zip(actions, steps, strict=True))


def test_white_canonical_rotation_maps_origin_and_direction_round_trip() -> None:
    red = State(men=(_mask(14), 0), kings=(0, 0), side_to_move=PlayerId.RED)
    white = State(men=(0, _mask(19)), kings=(0, 0), side_to_move=PlayerId.WHITE)

    red_actions = {decode_action_id(step_to_action(red, step)) for step in legal_steps(red)}
    white_actions = {decode_action_id(step_to_action(white, step)) for step in legal_steps(white)}

    assert red_actions == white_actions
    for step in legal_steps(white):
        canonical_square, _direction = decode_action_id(step_to_action(white, step))
        assert canonical_square == rotate_square(step.origin)


def test_forced_continuation_mask_exposes_only_the_moving_piece() -> None:
    state = State(
        men=(_mask(9, 11), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    continuation = apply_step(state, _step(9, 18, 14)).after
    action_map = legal_action_map(continuation)

    assert continuation.moving_square is not None
    assert action_map
    assert {step.origin for step in action_map.values()} == {continuation.moving_square}
    assert legal_action_mask(continuation).sum() == len(legal_steps(continuation))


def test_promotion_ending_jump_has_no_same_piece_continuation_action() -> None:
    state = State(
        men=(_mask(21), _mask(25, 26)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    transition = apply_step(state, _step(21, 30, 25))

    assert transition.move_completed is True
    assert transition.after.capture_in_progress is False
    assert all(
        step.origin != PROMOTION_SQUARE for step in legal_action_map(transition.after).values()
    )


def test_legal_mask_matches_generator_over_seeded_reachable_states() -> None:
    rng = random.Random(20260728)
    state = State.initial()
    for _ in range(1_000):
        steps = legal_steps(state)
        if not steps or state.ply >= MAX_PLIES:
            state = State.initial()
            steps = legal_steps(state)
        mask = legal_action_mask(state)
        assert mask.dtype == np.bool_
        assert mask.shape == (ACTION_COUNT,)
        assert int(mask.sum()) == len(steps)
        assert mask.any()
        state = apply_step(state, steps[rng.randrange(len(steps))]).after


def test_action_to_step_rejects_an_in_range_but_illegal_action() -> None:
    state = State.initial()
    illegal = next(action for action in range(ACTION_COUNT) if not legal_action_mask(state)[action])
    with pytest.raises(ActionEncodingError, match="legal"):
        action_to_step(state, illegal)


def test_step_encoding_rejects_invalid_objects_and_illegal_steps() -> None:
    state = State.initial()
    legal_step = legal_steps(state)[0]
    with pytest.raises(TypeError, match="State"):
        step_to_action(cast(State, "state"), legal_step)
    with pytest.raises(TypeError, match="Step"):
        step_to_action(state, cast(Step, "step"))
    with pytest.raises(ActionEncodingError, match="legal"):
        step_to_action(state, Step(origin=0, destination=4))
    with pytest.raises(TypeError, match="State"):
        legal_action_mask(cast(State, "state"))


def test_action_map_rejects_non_diagonal_generator_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(masking, "legal_steps", lambda _state: (Step(0, 1),))
    with pytest.raises(ActionEncodingError, match="diagonal"):
        legal_action_map(State.initial())


def test_action_map_detects_any_future_generator_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    colliding_steps = (Step(0, 5), Step(0, 9, captured=5))
    monkeypatch.setattr(masking, "legal_steps", lambda _state: colliding_steps)
    with pytest.raises(RuntimeError, match="collided"):
        legal_action_map(State.initial())
