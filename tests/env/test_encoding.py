"""Canonical eight-plane observation tests for GOAL.md §6.2."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from checkers.env.encoding import (
    BOARD_SIZE,
    OBSERVATION_PLANES,
    encode_observation,
)
from checkers.rules.board import coord, is_playable_coord, rotate_square
from checkers.rules.state import PlayerId, State


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _plane_squares(observation: np.ndarray, plane: int) -> set[int]:
    squares: set[int] = set()
    for square in range(32):
        row, column = coord(square)
        if observation[plane, row, column] == 1.0:
            squares.add(square)
    return squares


def test_initial_observation_has_exact_shape_dtype_and_actor_order() -> None:
    observation = encode_observation(State.initial())

    assert observation.shape == (OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert observation.dtype == np.float32
    assert _plane_squares(observation, 0) == set(range(12))
    assert _plane_squares(observation, 1) == set()
    assert _plane_squares(observation, 2) == set(range(20, 32))
    assert _plane_squares(observation, 3) == set()
    assert not observation[4].any()
    assert not observation[5].any()
    assert not observation[6].any()
    assert not observation[7].any()


def test_white_observation_rotates_180_and_swaps_actor_planes() -> None:
    red_view = State(
        men=(_mask(9), _mask(25)),
        kings=(_mask(14), _mask(20)),
        side_to_move=PlayerId.RED,
    )
    white_view = State(
        men=red_view.men,
        kings=red_view.kings,
        side_to_move=PlayerId.WHITE,
    )

    red_observation = encode_observation(red_view)
    white_observation = encode_observation(white_view)

    assert _plane_squares(white_observation, 0) == {rotate_square(24)}
    assert _plane_squares(white_observation, 1) == {rotate_square(19)}
    assert _plane_squares(white_observation, 2) == {rotate_square(8)}
    assert _plane_squares(white_observation, 3) == {rotate_square(13)}
    assert not np.array_equal(red_observation[:4], white_observation[:4])


def test_n7_pending_and_forced_planes_prevent_observation_aliasing() -> None:
    mid_sequence = State(
        men=(_mask(18), _mask(14, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        capture_in_progress=True,
        moving_square=17,
        sequence_origin=8,
        captured_pending=_mask(14),
        no_progress=(17, 39),
        ply=5,
    )
    same_placement_boundary = State(
        men=mid_sequence.men,
        kings=mid_sequence.kings,
        side_to_move=mid_sequence.side_to_move,
        no_progress=mid_sequence.no_progress,
        ply=mid_sequence.ply,
    )

    mid_observation = encode_observation(mid_sequence)
    boundary_observation = encode_observation(same_placement_boundary)

    assert _plane_squares(mid_observation, 2) == {13, 21}
    assert _plane_squares(mid_observation, 4) == {13}
    assert _plane_squares(mid_observation, 5) == {17}
    assert not boundary_observation[4].any()
    assert not boundary_observation[5].any()
    assert not np.array_equal(mid_observation, boundary_observation)


def test_counter_and_ply_planes_are_broadcast_normalized_constants() -> None:
    state = State(
        men=(_mask(9), _mask(24)),
        kings=(0, 0),
        side_to_move=PlayerId.WHITE,
        no_progress=(7, 20),
        ply=128,
    )
    observation = encode_observation(state, max_plies=512)

    assert np.all(observation[6] == np.float32(0.5))
    assert np.all(observation[7] == np.float32(0.25))


def test_light_squares_are_zero_in_all_piece_planes() -> None:
    state = State(
        men=(_mask(1, 9), _mask(24, 32)),
        kings=(_mask(14), _mask(19)),
        side_to_move=PlayerId.WHITE,
    )
    observation = encode_observation(state)

    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            if not is_playable_coord(row, column):
                assert not observation[:5, row, column].any()


def test_baseline_eight_planes_deliberately_omit_opponent_counter() -> None:
    first = State(
        men=(_mask(9), _mask(24)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        no_progress=(17, 1),
    )
    second = State(
        men=first.men,
        kings=first.kings,
        side_to_move=first.side_to_move,
        no_progress=(17, 39),
    )

    assert np.array_equal(encode_observation(first), encode_observation(second))


@pytest.mark.parametrize("max_plies", [0, -1, True])
def test_observation_rejects_invalid_max_plies(max_plies: int) -> None:
    with pytest.raises((TypeError, ValueError), match="max_plies"):
        encode_observation(State.initial(), max_plies=max_plies)


def test_observation_requires_a_state() -> None:
    with pytest.raises(TypeError, match="State"):
        encode_observation(cast(State, "state"))
