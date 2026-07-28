"""Terminal-condition boundaries for WCDF losses and declared engine variants."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import cast

import pytest

from checkers.rules.moves import Step, apply_step
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import (
    DEFAULT_MAX_PLIES,
    Outcome,
    TerminationReason,
    terminal_outcome,
)


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def _mobile_state(
    *,
    no_progress: tuple[int, int] = (0, 0),
    ply: int = 0,
) -> State:
    return State(
        men=(0, 0),
        kings=(_mask(14), _mask(24)),
        side_to_move=PlayerId.RED,
        no_progress=no_progress,
        ply=ply,
    )


def test_r6_1_no_pieces_loses() -> None:
    state = State(
        men=(_mask(9), 0),
        kings=(0, 0),
        side_to_move=PlayerId.WHITE,
    )

    outcome = terminal_outcome(state)

    assert outcome == Outcome(winner=PlayerId.RED, reason=TerminationReason.NO_PIECES)
    assert outcome is not None
    assert outcome.is_draw is False
    assert outcome.score_for(PlayerId.RED) == 1
    assert outcome.score_for(PlayerId.WHITE) == -1


def test_r6_2_stalemate_is_loss() -> None:
    state = State(
        men=(0x0000FFFF, 0xFFFF0000),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )

    assert terminal_outcome(state) == Outcome(
        winner=PlayerId.WHITE,
        reason=TerminationReason.NO_LEGAL_MOVE,
    )


def test_loss_precedes_engine_variant_draws_when_boundaries_coincide() -> None:
    state = State(
        men=(_mask(9), 0),
        kings=(0, 0),
        side_to_move=PlayerId.WHITE,
        no_progress=(40, 40),
        ply=512,
    )

    assert terminal_outcome(state) == Outcome(
        winner=PlayerId.RED,
        reason=TerminationReason.NO_PIECES,
    )


def test_r6_3_per_player_40_move_boundary() -> None:
    below = _mobile_state(no_progress=(40, 39))
    at_boundary = _mobile_state(no_progress=(40, 40))

    assert terminal_outcome(below) is None
    assert terminal_outcome(at_boundary) == Outcome(
        winner=None,
        reason=TerminationReason.NO_PROGRESS,
    )


def test_r6_3_completed_king_move_reaches_both_counters_once() -> None:
    state = State(
        men=(0, 0),
        kings=(_mask(14), _mask(24)),
        side_to_move=PlayerId.WHITE,
        no_progress=(40, 39),
    )

    after = apply_step(state, _step(24, 20)).after

    assert after.no_progress == (40, 40)
    assert terminal_outcome(after) == Outcome(
        winner=None,
        reason=TerminationReason.NO_PROGRESS,
    )


def test_r6_4_repetition_only_at_move_boundaries() -> None:
    boundary = _mobile_state()
    capture = State(
        men=(0, _mask(6, 15)),
        kings=(_mask(1), 0),
        side_to_move=PlayerId.RED,
    )
    mid_sequence = apply_step(capture, _step(1, 10, 6)).after

    assert terminal_outcome(boundary, repetition_draws=False, repetition_count=3) is None
    assert terminal_outcome(boundary, repetition_draws=True, repetition_count=2) is None
    assert terminal_outcome(boundary, repetition_draws=True, repetition_count=3) == Outcome(
        winner=None,
        reason=TerminationReason.REPETITION,
    )
    assert terminal_outcome(mid_sequence, repetition_draws=True, repetition_count=3) is None


def test_r6_5_511_vs_512_step_boundary() -> None:
    below = _mobile_state(ply=511)

    assert terminal_outcome(below) is None
    at_boundary = apply_step(below, _step(14, 17)).after
    assert at_boundary.ply == DEFAULT_MAX_PLIES
    assert terminal_outcome(at_boundary) == Outcome(
        winner=None,
        reason=TerminationReason.PLY_CAP,
    )


def test_r6_6_no_draw_by_agreement_api() -> None:
    assert "draw_agreed" not in inspect.signature(terminal_outcome).parameters
    with pytest.raises(TypeError, match="unexpected"):
        terminal_outcome(_mobile_state(), **{"draw_agreed": True})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_plies": 0}, "max_plies"),
        ({"max_plies": cast(int, True)}, "max_plies"),
        ({"repetition_count": -1}, "repetition_count"),
        ({"repetition_count": cast(int, True)}, "repetition_count"),
        ({"repetition_draws": cast(bool, 1)}, "repetition_draws"),
    ],
)
def test_terminal_outcome_rejects_invalid_runtime_options(
    kwargs: dict[str, int | bool],
    message: str,
) -> None:
    call = cast(Callable[..., Outcome | None], terminal_outcome)
    with pytest.raises((TypeError, ValueError), match=message):
        call(_mobile_state(), **kwargs)


def test_outcome_score_requires_explicit_player_identity() -> None:
    outcome = Outcome(winner=None, reason=TerminationReason.PLY_CAP)
    assert outcome.score_for(PlayerId.RED) == 0
    with pytest.raises(TypeError, match="PlayerId"):
        outcome.score_for(cast(PlayerId, 0))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Outcome(
            winner=cast(PlayerId, 0),
            reason=TerminationReason.NO_PIECES,
        ),
        lambda: Outcome(
            winner=None,
            reason=cast(TerminationReason, "ply_cap"),
        ),
        lambda: Outcome(winner=None, reason=TerminationReason.NO_PIECES),
        lambda: Outcome(winner=PlayerId.RED, reason=TerminationReason.PLY_CAP),
    ],
)
def test_outcome_rejects_inconsistent_runtime_values(factory: Callable[[], Outcome]) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_terminal_outcome_requires_a_state() -> None:
    with pytest.raises(TypeError, match="State"):
        terminal_outcome(cast(State, "state"))
