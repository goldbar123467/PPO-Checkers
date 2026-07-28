"""Immutable complete-state tests for GOAL.md §5.1 and WCDF R1.4-R1.5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from checkers.rules.state import PlayerId, State


def test_r1_4_initial_position_is_exact_and_contains_only_men() -> None:
    state = State.initial()
    assert state.men == ((1 << 12) - 1, ((1 << 12) - 1) << 20)
    assert state.kings == (0, 0)
    assert state.occupied == ((1 << 12) - 1) | (((1 << 12) - 1) << 20)
    assert state.capture_in_progress is False
    assert state.moving_square is None
    assert state.sequence_origin is None
    assert state.captured_pending == 0
    assert state.no_progress == (0, 0)
    assert state.ply == 0


def test_r1_5_red_is_explicit_first_player() -> None:
    state = State.initial()
    assert state.side_to_move is PlayerId.RED
    assert PlayerId.RED.opponent is PlayerId.WHITE
    assert PlayerId.WHITE.opponent is PlayerId.RED


def test_state_is_frozen() -> None:
    state = State.initial()
    field_name = "ply"
    with pytest.raises(FrozenInstanceError):
        setattr(state, field_name, 1)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: State(
                men=cast(tuple[int, int], [0, 0]),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
            ),
            "pair",
        ),
        (
            lambda: State(
                men=(cast(int, True), 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
            ),
            "uint32",
        ),
        (lambda: State(men=(-1, 0), kings=(0, 0), side_to_move=PlayerId.RED), "uint32"),
        (lambda: State(men=(1 << 32, 0), kings=(0, 0), side_to_move=PlayerId.RED), "uint32"),
        (lambda: State(men=(1, 1), kings=(0, 0), side_to_move=PlayerId.RED), "disjoint"),
        (lambda: State(men=(1, 0), kings=(1, 0), side_to_move=PlayerId.RED), "disjoint"),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=cast(PlayerId, 0)),
            "PlayerId",
        ),
        (
            lambda: State(
                men=(0, 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                capture_in_progress=cast(bool, 1),
            ),
            "must be bool",
        ),
        (
            lambda: State(
                men=(0, 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                moving_square=cast(int, True),
            ),
            "integer square",
        ),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED, moving_square=32),
            "square",
        ),
        (
            lambda: State(
                men=(0, 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                capture_in_progress=True,
            ),
            "moving_square",
        ),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED, moving_square=0),
            "capture_in_progress",
        ),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED, sequence_origin=0),
            "capture_in_progress",
        ),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED, captured_pending=1),
            "capture_in_progress",
        ),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED, no_progress=(-1, 0)),
            "non-negative",
        ),
        (
            lambda: State(
                men=(0, 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                no_progress=cast(tuple[int, int], [0, 0]),
            ),
            "pair of integers",
        ),
        (
            lambda: State(
                men=(0, 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                no_progress=(cast(int, True), 0),
            ),
            "pair of integers",
        ),
        (
            lambda: State(
                men=(0, 0),
                kings=(0, 0),
                side_to_move=PlayerId.RED,
                ply=cast(int, True),
            ),
            "ply must be an integer",
        ),
        (
            lambda: State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED, ply=-1),
            "non-negative",
        ),
    ],
)
def test_state_rejects_invalid_invariants(factory: Callable[[], State], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_capture_state_requires_actor_piece_and_pending_opponent_subset() -> None:
    valid = State(
        men=(1 << 9, 1 << 13),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        capture_in_progress=True,
        moving_square=9,
        sequence_origin=5,
        captured_pending=1 << 13,
    )
    assert valid.capture_in_progress

    with pytest.raises(ValueError, match="actor"):
        State(
            men=(1 << 9, 1 << 13),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
            capture_in_progress=True,
            moving_square=8,
            sequence_origin=5,
            captured_pending=1 << 13,
        )
    with pytest.raises(ValueError, match="opponent"):
        State(
            men=(1 << 9, 1 << 13),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
            capture_in_progress=True,
            moving_square=9,
            sequence_origin=5,
            captured_pending=1 << 12,
        )
