"""Metamorphic checks for the valid rank-preserving American Checkers symmetry."""

from __future__ import annotations

from collections.abc import Callable

from checkers.rules.board import PLAYABLE_SQUARES, bit, coord, rotate_square
from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.state import State

BFS_DEPTH = 4


def _rotate_mask(mask: int) -> int:
    rotated = 0
    for square in range(PLAYABLE_SQUARES):
        if mask & bit(square):
            rotated |= bit(rotate_square(square))
    return rotated


def _rotate_optional(square: int | None) -> int | None:
    return None if square is None else rotate_square(square)


def _rotate_and_swap_state(state: State) -> State:
    return State(
        men=(_rotate_mask(state.men[1]), _rotate_mask(state.men[0])),
        kings=(_rotate_mask(state.kings[1]), _rotate_mask(state.kings[0])),
        side_to_move=state.side_to_move.opponent,
        capture_in_progress=state.capture_in_progress,
        moving_square=_rotate_optional(state.moving_square),
        sequence_origin=_rotate_optional(state.sequence_origin),
        captured_pending=_rotate_mask(state.captured_pending),
        no_progress=(state.no_progress[1], state.no_progress[0]),
        ply=state.ply,
    )


def _rotate_step(step: Step) -> Step:
    return Step(
        origin=rotate_square(step.origin),
        destination=rotate_square(step.destination),
        captured=_rotate_optional(step.captured),
    )


def test_colour_swap_with_180_rotation_is_an_exact_transition_symmetry() -> None:
    frontier = {State.initial()}
    seen = set(frontier)
    for _ in range(BFS_DEPTH + 1):
        next_frontier: set[State] = set()
        for state in frontier:
            transformed_state = _rotate_and_swap_state(state)
            assert _rotate_and_swap_state(transformed_state) == state
            expected_steps = {_rotate_step(step) for step in legal_steps(state)}
            assert set(legal_steps(transformed_state)) == expected_steps

            for step in legal_steps(state):
                transition = apply_step(state, step)
                transformed_transition = apply_step(transformed_state, _rotate_step(step))
                assert transformed_transition.after == _rotate_and_swap_state(transition.after)
                assert transformed_transition.move_completed == transition.move_completed
                child = transition.after
                if child not in seen:
                    seen.add(child)
                    next_frontier.add(child)
        frontier = next_frontier


def test_no_separate_rank_preserving_geometric_mirror_exists() -> None:
    """Audit BLOCK-003 over every D4 map that preserves playable-square colour."""

    transforms: tuple[Callable[[int, int], tuple[int, int]], ...] = (
        lambda row, column: (row, column),
        lambda row, column: (7 - row, 7 - column),
        lambda row, column: (column, row),
        lambda row, column: (7 - column, 7 - row),
    )
    row_preserving: list[int] = []
    row_reversing: list[int] = []
    for index, transform in enumerate(transforms):
        mapped_rows = [transform(*coord(square))[0] for square in range(PLAYABLE_SQUARES)]
        original_rows = [coord(square)[0] for square in range(PLAYABLE_SQUARES)]
        if mapped_rows == original_rows:
            row_preserving.append(index)
        if mapped_rows == [7 - row for row in original_rows]:
            row_reversing.append(index)

    assert row_preserving == [0]
    assert row_reversing == [1]
