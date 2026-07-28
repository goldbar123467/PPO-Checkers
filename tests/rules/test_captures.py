"""Mandatory capture and delayed-removal tests for WCDF R4."""

from __future__ import annotations

from checkers.rules.board import acf_number, coord, is_playable_coord
from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.state import PlayerId, State

INTERNAL_SQUARE_18 = 17


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def test_r4_1_jump_geometry_and_landing_occupancy() -> None:
    open_landing = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    blocked_landing = State(
        men=(_mask(9, 18), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    assert legal_steps(open_landing) == (_step(9, 18, 14),)
    assert _step(9, 18, 14) not in legal_steps(blocked_landing)


def test_r4_2_capture_is_mandatory_across_the_whole_player() -> None:
    state = State(
        men=(_mask(9, 11), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    assert legal_steps(state) == (_step(9, 18, 14),)


def test_r4_3_1_king_jumps_forward_and_backward() -> None:
    state = State(
        men=(0, _mask(10, 18)),
        kings=(_mask(14), 0),
        side_to_move=PlayerId.RED,
    )

    assert set(legal_steps(state)) == {_step(14, 7, 10), _step(14, 23, 18)}


def test_r4_3_2_man_never_jumps_backward() -> None:
    state = State(
        men=(_mask(14), _mask(10, 18)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    assert legal_steps(state) == (_step(14, 23, 18),)


def test_r4_4_continuation_is_mandatory_for_the_same_piece() -> None:
    state = State(
        men=(_mask(9, 11), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    first = apply_step(state, _step(9, 18, 14))

    assert not first.move_completed
    assert first.after.moving_square == INTERNAL_SQUARE_18
    assert legal_steps(first.after) == (_step(18, 25, 22),)


def test_r4_5_marked_piece_remains_occupied_and_cannot_be_jumped_twice() -> None:
    state = State(
        men=(0, _mask(6, 15)),
        kings=(_mask(1), 0),
        side_to_move=PlayerId.RED,
    )

    first = apply_step(state, _step(1, 10, 6))

    assert not first.move_completed
    assert first.after.captured_pending == _mask(6)
    assert first.after.occupied & _mask(6)
    assert legal_steps(first.after) == (_step(10, 19, 15),)
    assert _step(10, 1, 6) not in legal_steps(first.after)


def test_r4_5_landing_on_a_pending_square_is_geometrically_impossible() -> None:
    """Prove the short-jump parity fact behind BLOCK-002 exhaustively."""

    for origin in range(32):
        origin_row, origin_column = coord(origin)
        for row_delta in (-1, 1):
            for column_delta in (-1, 1):
                middle = (origin_row + row_delta, origin_column + column_delta)
                landing = (origin_row + 2 * row_delta, origin_column + 2 * column_delta)
                if not is_playable_coord(*landing):
                    continue
                middle_square = acf_number(*middle) - 1
                landing_square = acf_number(*landing) - 1
                middle_row, middle_column = coord(middle_square)
                landing_row, landing_column = coord(landing_square)
                assert (landing_row % 2, landing_column % 2) == (
                    origin_row % 2,
                    origin_column % 2,
                )
                assert (middle_row % 2, middle_column % 2) != (
                    origin_row % 2,
                    origin_column % 2,
                )


def test_r4_6_no_majority_capture_rule() -> None:
    state = State(
        men=(0, _mask(14, 15, 23)),
        kings=(_mask(10), 0),
        side_to_move=PlayerId.RED,
    )

    assert set(legal_steps(state)) == {_step(10, 17, 14), _step(10, 19, 15)}
    assert apply_step(state, _step(10, 17, 14)).move_completed
    assert not apply_step(state, _step(10, 19, 15)).move_completed
