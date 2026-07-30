"""Canonical eight-plane observations for the checkers environment."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from checkers.rules.board import BOARD_SIZE as _BOARD_SIZE
from checkers.rules.board import coord, rotate_square
from checkers.rules.state import State

BOARD_SIZE = _BOARD_SIZE
OBSERVATION_PLANES = 8
NO_PROGRESS_LIMIT = 40
DEFAULT_MAX_PLIES = 512

Float32Array = NDArray[np.float32]


def _iter_squares(mask: int) -> Iterator[int]:
    remaining = mask
    while remaining:
        least_significant = remaining & -remaining
        yield least_significant.bit_length() - 1
        remaining ^= least_significant


def _canonical_square(state: State, square: int) -> int:
    return square if int(state.side_to_move) == 0 else rotate_square(square)


def _write_mask(observation: Float32Array, plane: int, state: State, mask: int) -> None:
    for square in _iter_squares(mask):
        row, column = coord(_canonical_square(state, square))
        observation[plane, row, column] = np.float32(1.0)


def encode_observation(
    state: State,
    *,
    max_plies: int = DEFAULT_MAX_PLIES,
) -> Float32Array:
    """Encode a state from its actor's canonical perspective.

    Args:
        state: Complete immutable state to encode.
        max_plies: Declared R6.5 terminal-ply limit used to normalize plane 7.

    Returns:
        A float32 array with shape ``(8, 8, 8)`` following the public experiment contract.

    Raises:
        TypeError: If ``state`` is not a ``State`` or ``max_plies`` is not an integer.
        ValueError: If ``max_plies`` is less than one.
    """

    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies < 1:
        raise ValueError("max_plies must be at least one")

    observation = np.zeros(
        (OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    actor = int(state.side_to_move)
    opponent = int(state.side_to_move.opponent)
    _write_mask(observation, 0, state, state.men[actor])
    _write_mask(observation, 1, state, state.kings[actor])
    _write_mask(observation, 2, state, state.men[opponent])
    _write_mask(observation, 3, state, state.kings[opponent])
    _write_mask(observation, 4, state, state.captured_pending)
    if state.moving_square is not None:
        _write_mask(observation, 5, state, 1 << state.moving_square)
    observation[6].fill(np.float32(state.no_progress[actor] / NO_PROGRESS_LIMIT))
    observation[7].fill(np.float32(state.ply / max_plies))
    return observation
