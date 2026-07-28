"""Independent object-grid oracle corroboration for the bitboard move generator."""

from __future__ import annotations

import pytest

from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.oracle import oracle_legal_steps
from checkers.rules.state import PlayerId, State

BFS_DEPTH = 5
MINIMUM_NONVACUOUS_COMPARISONS = 1_000


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


@pytest.mark.parametrize(
    "state",
    [
        State.initial(),
        State(men=(_mask(14), 0), kings=(0, 0), side_to_move=PlayerId.RED),
        State(men=(0, _mask(19)), kings=(0, 0), side_to_move=PlayerId.WHITE),
        State(
            men=(_mask(9, 11), _mask(14)),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        ),
        State(
            men=(0, _mask(6, 15)),
            kings=(_mask(10), 0),
            side_to_move=PlayerId.RED,
            capture_in_progress=True,
            moving_square=9,
            sequence_origin=0,
            captured_pending=_mask(6),
            ply=1,
        ),
        State(men=(0, 0), kings=(0, 0), side_to_move=PlayerId.RED),
    ],
)
def test_independent_oracle_matches_hand_constructed_rule_positions(state: State) -> None:
    assert oracle_legal_steps(state) == legal_steps(state)


def test_independent_oracle_has_expected_initial_legal_steps() -> None:
    expected = {
        _step(9, 13),
        _step(9, 14),
        _step(10, 14),
        _step(10, 15),
        _step(11, 15),
        _step(11, 16),
        _step(12, 16),
    }
    assert set(oracle_legal_steps(State.initial())) == expected


def test_differential_bfs_agrees_through_fixed_depth() -> None:
    frontier = {State.initial()}
    seen = set(frontier)
    comparisons = 0

    for _ in range(BFS_DEPTH + 1):
        next_frontier: set[State] = set()
        for state in frontier:
            fast = legal_steps(state)
            assert oracle_legal_steps(state) == fast
            comparisons += 1
            for step in fast:
                child = apply_step(state, step).after
                if child not in seen:
                    seen.add(child)
                    next_frontier.add(child)
        frontier = next_frontier

    assert comparisons >= MINIMUM_NONVACUOUS_COMPARISONS
