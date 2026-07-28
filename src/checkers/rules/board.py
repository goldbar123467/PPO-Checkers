"""Frozen ACF 1-32 board geometry for American Checkers."""

from __future__ import annotations

from checkers.rules.state import PlayerId

BOARD_SIZE = 8
PLAYABLE_SQUARES = 32
FULL_BOARD_MASK = (1 << PLAYABLE_SQUARES) - 1


def _validate_square(square: int) -> None:
    if isinstance(square, bool) or not isinstance(square, int):
        raise TypeError("square must be an integer")
    if not 0 <= square < PLAYABLE_SQUARES:
        raise ValueError("square must be in [0, 31]")


def is_playable_coord(row: int, column: int) -> bool:
    """Return whether a coordinate is an in-bounds playable dark square.

    Args:
        row: Zero-based row from Red's home row toward White.
        column: Zero-based column from Red's left to right.

    Returns:
        True exactly for one of the board's 32 playable squares.
    """

    return 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE and (row + column) % 2 == 0


def coord(square: int) -> tuple[int, int]:
    """Convert a zero-based ACF square index to a board coordinate.

    Args:
        square: Internal square index in `[0, 31]`; ACF number is `square + 1`.

    Returns:
        `(row, column)` viewed from Red's side.

    Raises:
        TypeError: If `square` is not an integer.
        ValueError: If `square` is outside `[0, 31]`.
    """

    _validate_square(square)
    row, offset = divmod(square, 4)
    column = 6 + row % 2 - 2 * offset
    return row, column


def acf_number(row: int, column: int) -> int:
    """Convert a playable coordinate to its one-based ACF number.

    Args:
        row: Zero-based row from Red's home row toward White.
        column: Zero-based column from Red's left to right.

    Returns:
        The official number in `[1, 32]`.

    Raises:
        ValueError: If the coordinate is out of bounds or not playable.
    """

    if not is_playable_coord(row, column):
        raise ValueError("coordinate must be an in-bounds playable dark square")
    offset = (6 + row % 2 - column) // 2
    return row * 4 + offset + 1


def bit(square: int) -> int:
    """Return the uint32 bit for an internal square index.

    Args:
        square: Internal square index in `[0, 31]`.

    Returns:
        A Python integer with exactly one of its low 32 bits set.

    Raises:
        TypeError: If `square` is not an integer.
        ValueError: If `square` is outside `[0, 31]`.
    """

    _validate_square(square)
    return 1 << square


def rotate_square(square: int) -> int:
    """Rotate a square 180 degrees into the opponent's canonical frame.

    Args:
        square: Internal square index in `[0, 31]`.

    Returns:
        The rotated internal square index.

    Raises:
        TypeError: If `square` is not an integer.
        ValueError: If `square` is outside `[0, 31]`.
    """

    _validate_square(square)
    return PLAYABLE_SQUARES - 1 - square


def double_corner(player: PlayerId) -> tuple[int, int]:
    """Return a player's two double-corner squares.

    Args:
        player: Player whose right-hand double corner is requested.

    Returns:
        Two internal square indices in ascending ACF order.

    Raises:
        TypeError: If `player` is not a `PlayerId`.
    """

    if not isinstance(player, PlayerId):
        raise TypeError("player must be a PlayerId")
    return (0, 4) if player is PlayerId.RED else (27, 31)
