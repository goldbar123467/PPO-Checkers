"""Turn structure and simple-move tests for WCDF R2 and R3."""

from __future__ import annotations

from typing import cast

import pytest

from checkers.rules.board import acf_number, coord, is_playable_coord
from checkers.rules.moves import (
    DIRECTION_DELTAS,
    GEOMETRY,
    IllegalStepError,
    Step,
    Transition,
    apply_step,
    legal_steps,
    undo_step,
)
from checkers.rules.state import PlayerId, State

TWO_ENVIRONMENT_STEPS = 2


def test_precomputed_geometry_matches_frozen_coordinate_derivation() -> None:
    for square, directions in enumerate(GEOMETRY):
        row, column = coord(square)
        for (row_delta, column_delta), actual in zip(
            DIRECTION_DELTAS,
            directions,
            strict=True,
        ):
            adjacent_coord = (row + row_delta, column + column_delta)
            landing_coord = (row + 2 * row_delta, column + 2 * column_delta)
            expected_adjacent = (
                acf_number(*adjacent_coord) - 1 if is_playable_coord(*adjacent_coord) else None
            )
            expected_landing = (
                acf_number(*landing_coord) - 1 if is_playable_coord(*landing_coord) else None
            )
            assert actual == (expected_adjacent, expected_landing)


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def test_r2_1_completed_moves_alternate_players() -> None:
    state = State(
        men=(_mask(9), _mask(24)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    red_transition = apply_step(state, _step(9, 13))
    assert red_transition.move_completed
    assert red_transition.after.side_to_move is PlayerId.WHITE

    white_transition = apply_step(red_transition.after, _step(24, 20))
    assert white_transition.move_completed
    assert white_transition.after.side_to_move is PlayerId.RED


def test_r2_2_multijump_is_one_move_and_many_environment_steps() -> None:
    state = State(
        men=(_mask(9), _mask(14, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        no_progress=(7, 9),
    )

    first = apply_step(state, _step(9, 18, 14))
    assert not first.move_completed
    assert first.after.side_to_move is PlayerId.RED
    assert first.after.no_progress == (7, 9)
    assert first.after.ply == 1

    second = apply_step(first.after, _step(18, 25, 22))
    assert second.move_completed
    assert second.after.side_to_move is PlayerId.WHITE
    assert second.after.no_progress == (0, 9)
    assert second.after.ply == TWO_ENVIRONMENT_STEPS


def test_r3_1_man_simple_moves_are_forward_only() -> None:
    red = State(men=(_mask(14), 0), kings=(0, 0), side_to_move=PlayerId.RED)
    white = State(men=(0, _mask(19)), kings=(0, 0), side_to_move=PlayerId.WHITE)

    assert set(legal_steps(red)) == {_step(14, 17), _step(14, 18)}
    assert set(legal_steps(white)) == {_step(19, 15), _step(19, 16)}


def test_r3_2_king_simple_moves_forward_and_backward() -> None:
    state = State(men=(0, 0), kings=(_mask(14), 0), side_to_move=PlayerId.RED)

    assert set(legal_steps(state)) == {
        _step(14, 9),
        _step(14, 10),
        _step(14, 17),
        _step(14, 18),
    }


def test_r3_3_no_flying_king_or_occupied_destination() -> None:
    state = State(
        men=(_mask(17, 18), 0),
        kings=(_mask(14), 0),
        side_to_move=PlayerId.RED,
    )

    steps = set(legal_steps(state))
    assert {_step(14, 9), _step(14, 10)} <= steps
    assert _step(14, 17) not in steps
    assert _step(14, 18) not in steps
    assert _step(14, 23) not in steps


def test_r3_3_illegal_step_raises_and_undo_restores_exact_state() -> None:
    state = State(men=(_mask(14), 0), kings=(0, 0), side_to_move=PlayerId.RED)
    with pytest.raises(IllegalStepError, match="legal"):
        apply_step(state, _step(14, 10))

    transition = apply_step(state, _step(14, 17))
    assert undo_step(transition) == state


@pytest.mark.parametrize(
    ("origin", "destination", "captured", "message"),
    [
        (0, 0, None, "differ"),
        (0, 4, 0, "captured square"),
        (0, 4, 4, "captured square"),
    ],
)
def test_step_rejects_aliased_square_fields(
    origin: int,
    destination: int,
    captured: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Step(origin=origin, destination=destination, captured=captured)


def test_public_transition_functions_reject_wrong_object_types() -> None:
    state = State.initial()
    with pytest.raises(TypeError, match="Step"):
        apply_step(state, cast(Step, 0))
    with pytest.raises(TypeError, match="Transition"):
        undo_step(cast(Transition, state))
