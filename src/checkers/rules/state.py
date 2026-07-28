"""Immutable complete game state for American Checkers transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

UINT32_MAX = (1 << 32) - 1
PAIR_SIZE = 2
PLAYABLE_SQUARES = 32


class PlayerId(IntEnum):
    """Stable player identity, independent of canonical observations."""

    RED = 0
    WHITE = 1

    @property
    def opponent(self) -> PlayerId:
        """Return the other player."""

        return PlayerId.WHITE if self is PlayerId.RED else PlayerId.RED


def _validate_uint32(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a uint32 integer")
    if not 0 <= value <= UINT32_MAX:
        raise ValueError(f"{field} must fit uint32")
    return value


def _validate_pair(values: object, field: str) -> tuple[int, int]:
    if not isinstance(values, tuple) or len(values) != PAIR_SIZE:
        raise TypeError(f"{field} must be a pair")
    return (
        _validate_uint32(values[0], f"{field}[0]"),
        _validate_uint32(values[1], f"{field}[1]"),
    )


def _validate_square_or_none(value: int | None, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer square or None")
    if not 0 <= value < PLAYABLE_SQUARES:
        raise ValueError(f"{field} square must be in [0, 31]")


@dataclass(frozen=True, slots=True)
class State:
    """Complete Markov state, including any active capture sequence.

    Bitboards are Python integers validated to the uint32 range. This preserves exact uint32
    semantics without NumPy scalar promotion or overflow surprises.
    """

    men: tuple[int, int]
    kings: tuple[int, int]
    side_to_move: PlayerId
    capture_in_progress: bool = False
    moving_square: int | None = None
    sequence_origin: int | None = None
    captured_pending: int = 0
    no_progress: tuple[int, int] = (0, 0)
    ply: int = 0

    def __post_init__(self) -> None:
        men = _validate_pair(self.men, "men")
        kings = _validate_pair(self.kings, "kings")
        if not isinstance(self.side_to_move, PlayerId):
            raise TypeError("side_to_move must be a PlayerId")
        if not isinstance(self.capture_in_progress, bool):
            raise TypeError("capture_in_progress must be bool")
        _validate_square_or_none(self.moving_square, "moving_square")
        _validate_square_or_none(self.sequence_origin, "sequence_origin")
        captured_pending = _validate_uint32(self.captured_pending, "captured_pending")
        self._validate_disjoint(men, kings)
        self._validate_capture_state(men, kings, captured_pending)
        self._validate_counters()

    @staticmethod
    def _validate_disjoint(men: tuple[int, int], kings: tuple[int, int]) -> None:
        occupied = 0
        for board in (*men, *kings):
            if occupied & board:
                raise ValueError("men and kings bitboards must be pairwise disjoint")
            occupied |= board

    def _validate_capture_state(
        self,
        men: tuple[int, int],
        kings: tuple[int, int],
        captured_pending: int,
    ) -> None:
        if self.capture_in_progress != (self.moving_square is not None):
            raise ValueError("capture_in_progress must match moving_square presence")
        if self.capture_in_progress != (self.sequence_origin is not None):
            raise ValueError("capture_in_progress must match sequence_origin presence")
        if captured_pending and not self.capture_in_progress:
            raise ValueError("captured_pending requires capture_in_progress")

        if self.capture_in_progress:
            actor = int(self.side_to_move)
            opponent = int(self.side_to_move.opponent)
            moving_bit = 1 << self.moving_square if self.moving_square is not None else 0
            if not moving_bit & (men[actor] | kings[actor]):
                raise ValueError("moving_square must contain the actor's piece")
            if captured_pending & ~(men[opponent] | kings[opponent]):
                raise ValueError("captured_pending must be a subset of opponent occupancy")

    def _validate_counters(self) -> None:
        if (
            not isinstance(self.no_progress, tuple)
            or len(self.no_progress) != PAIR_SIZE
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in self.no_progress
            )
        ):
            raise TypeError("no_progress must be a pair of integers")
        if any(value < 0 for value in self.no_progress):
            raise ValueError("no_progress counters must be non-negative")
        if isinstance(self.ply, bool) or not isinstance(self.ply, int):
            raise TypeError("ply must be an integer")
        if self.ply < 0:
            raise ValueError("ply must be non-negative")

    @classmethod
    def initial(cls) -> State:
        """Return the WCDF initial position with Red to move."""

        twelve_men = (1 << 12) - 1
        return cls(
            men=(twelve_men, twelve_men << 20),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        )

    @property
    def occupied(self) -> int:
        """Return the union of all men and king bitboards."""

        return self.men[0] | self.men[1] | self.kings[0] | self.kings[1]
