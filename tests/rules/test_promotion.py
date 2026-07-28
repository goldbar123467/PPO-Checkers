"""Promotion tests for WCDF R5."""

from __future__ import annotations

from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.state import PlayerId, State


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def test_r5_1_man_promotes_at_a_completed_move() -> None:
    state = State(men=(_mask(25), 0), kings=(0, 0), side_to_move=PlayerId.RED)

    transition = apply_step(state, _step(25, 29))

    assert transition.move_completed
    assert transition.after.men[PlayerId.RED] == 0
    assert transition.after.kings[PlayerId.RED] == _mask(29)


def test_r5_2_promotion_ends_jump_sequence_before_a_new_king_jump() -> None:
    state = State(
        men=(_mask(21), _mask(25, 26)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    transition = apply_step(state, _step(21, 30, 25))

    assert transition.move_completed
    assert transition.after.side_to_move is PlayerId.WHITE
    assert transition.after.kings[PlayerId.RED] == _mask(30)
    assert transition.after.men[PlayerId.WHITE] == _mask(26)
    assert transition.after.captured_pending == 0
    assert _step(30, 23, 26) not in legal_steps(transition.after)


def test_r5_3_king_is_never_demoted() -> None:
    state = State(men=(0, 0), kings=(_mask(25), 0), side_to_move=PlayerId.RED)

    transition = apply_step(state, _step(25, 21))

    assert transition.after.men[PlayerId.RED] == 0
    assert transition.after.kings[PlayerId.RED] == _mask(21)
