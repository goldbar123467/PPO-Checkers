"""Deliberately simple object-grid oracle for legal-step differential checks.

The production generator uses bitboards and a precomputed geometry table. This module instead
materializes an 8×8 object grid and performs direct coordinate arithmetic on every call. It shares
only the public `State`, `Step`, and frozen board-coordinate contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from checkers.rules.board import BOARD_SIZE, PLAYABLE_SQUARES, acf_number, coord, is_playable_coord
from checkers.rules.moves import Step
from checkers.rules.state import PlayerId, State

ALL_DIAGONALS = ((1, -1), (1, 1), (-1, -1), (-1, 1))


@dataclass(frozen=True, slots=True)
class _Piece:
    owner: PlayerId
    king: bool


Grid = list[list[_Piece | None]]


def _grid_from_state(state: State) -> Grid:
    grid: Grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for square in range(PLAYABLE_SQUARES):
        row, column = coord(square)
        square_bit = 1 << square
        for player in PlayerId:
            player_index = int(player)
            if state.men[player_index] & square_bit:
                grid[row][column] = _Piece(owner=player, king=False)
            elif state.kings[player_index] & square_bit:
                grid[row][column] = _Piece(owner=player, king=True)
    return grid


def _directions(piece: _Piece) -> tuple[tuple[int, int], ...]:
    if piece.king:
        return ALL_DIAGONALS
    return ((1, -1), (1, 1)) if piece.owner is PlayerId.RED else ((-1, -1), (-1, 1))


def _piece_at(grid: Grid, square: int) -> _Piece:
    row, column = coord(square)
    return cast(_Piece, grid[row][column])


def _oracle_captures_from(
    grid: Grid,
    origin: int,
    captured_pending: set[int],
) -> tuple[Step, ...]:
    piece = _piece_at(grid, origin)
    row, column = coord(origin)
    captures: list[Step] = []
    for row_delta, column_delta in _directions(piece):
        middle_coord = (row + row_delta, column + column_delta)
        landing_coord = (row + 2 * row_delta, column + 2 * column_delta)
        if not is_playable_coord(*middle_coord) or not is_playable_coord(*landing_coord):
            continue
        middle_piece = grid[middle_coord[0]][middle_coord[1]]
        if middle_piece is None or middle_piece.owner is piece.owner:
            continue
        middle = acf_number(*middle_coord) - 1
        if middle in captured_pending:
            continue
        if grid[landing_coord[0]][landing_coord[1]] is not None:
            continue
        captures.append(
            Step(
                origin=origin,
                destination=acf_number(*landing_coord) - 1,
                captured=middle,
            )
        )
    return tuple(captures)


def _oracle_simple_from(grid: Grid, origin: int) -> tuple[Step, ...]:
    piece = _piece_at(grid, origin)
    row, column = coord(origin)
    moves: list[Step] = []
    for row_delta, column_delta in _directions(piece):
        destination_coord = (row + row_delta, column + column_delta)
        if not is_playable_coord(*destination_coord):
            continue
        if grid[destination_coord[0]][destination_coord[1]] is None:
            moves.append(Step(origin=origin, destination=acf_number(*destination_coord) - 1))
    return tuple(moves)


def _sort_key(step: Step) -> tuple[int, int, int]:
    return step.origin, step.destination, -1 if step.captured is None else step.captured


def oracle_legal_steps(state: State) -> tuple[Step, ...]:
    """Generate legal steps using the independent object-grid implementation.

    Args:
        state: Complete immutable state to adjudicate.

    Returns:
        Legal steps in the same deterministic public ordering as `legal_steps`.

    """

    grid = _grid_from_state(state)
    pending = {
        square for square in range(PLAYABLE_SQUARES) if state.captured_pending & (1 << square)
    }
    origins: tuple[int, ...]
    if state.capture_in_progress:
        origins = (cast(int, state.moving_square),)
    else:
        origins = tuple(
            square
            for square in range(PLAYABLE_SQUARES)
            if (piece := grid[coord(square)[0]][coord(square)[1]]) is not None
            and piece.owner is state.side_to_move
        )

    captures = [step for origin in origins for step in _oracle_captures_from(grid, origin, pending)]
    if captures or state.capture_in_progress:
        return tuple(sorted(captures, key=_sort_key))

    moves = [step for origin in origins for step in _oracle_simple_from(grid, origin)]
    return tuple(sorted(moves, key=_sort_key))
