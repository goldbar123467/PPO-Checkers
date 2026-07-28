"""Bitboard-driven legal steps and reversible immutable state transitions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from checkers.rules.board import bit, coord
from checkers.rules.state import PlayerId, State

DIRECTION_DELTAS = ((1, -1), (1, 1), (-1, -1), (-1, 1))
RED_FORWARD_DIRECTIONS = (0, 1)
WHITE_FORWARD_DIRECTIONS = (2, 3)
KING_DIRECTIONS = (0, 1, 2, 3)

Geometry = tuple[int | None, int | None]


class IllegalStepError(ValueError):
    """Raised when a requested step is not legal in the supplied state."""


@dataclass(frozen=True, slots=True, init=False)
class Step:
    """One environment step: an adjacent move or one jump of a capture sequence.

    Args:
        origin: Zero-based ACF origin square.
        destination: Zero-based ACF destination square.
        captured: Zero-based ACF jumped square, or None for a simple move.

    Raises:
        TypeError: If a square is not an integer.
        ValueError: If a square is out of range or fields alias each other.
    """

    origin: int
    destination: int
    captured: int | None = None

    def __init__(self, origin: int, destination: int, captured: int | None = None) -> None:
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "captured", captured)
        self.__post_init__()

    def __post_init__(self) -> None:
        bit(self.origin)
        bit(self.destination)
        if self.captured is not None:
            bit(self.captured)
        if self.origin == self.destination:
            raise ValueError("step origin and destination must differ")
        if self.captured in (self.origin, self.destination):
            raise ValueError("captured square must differ from origin and destination")

    @property
    def is_capture(self) -> bool:
        """Return whether this step jumps an opposing piece."""

        return self.captured is not None


@dataclass(frozen=True, slots=True)
class Transition:
    """An applied step together with the exact state needed for O(1) undo.

    Args:
        before: Immutable state before the step.
        after: Immutable state after the step.
        step: Legal step that was applied.
        move_completed: Whether this environment step completed the checkers move.
    """

    before: State
    after: State
    step: Step
    move_completed: bool


GEOMETRY: tuple[tuple[Geometry, ...], ...] = (
    ((5, 9), (4, None), (None, None), (None, None)),
    ((6, 10), (5, 8), (None, None), (None, None)),
    ((7, 11), (6, 9), (None, None), (None, None)),
    ((None, None), (7, 10), (None, None), (None, None)),
    ((8, 13), (None, None), (0, None), (None, None)),
    ((9, 14), (8, 12), (1, None), (0, None)),
    ((10, 15), (9, 13), (2, None), (1, None)),
    ((11, None), (10, 14), (3, None), (2, None)),
    ((13, 17), (12, None), (5, 1), (4, None)),
    ((14, 18), (13, 16), (6, 2), (5, 0)),
    ((15, 19), (14, 17), (7, 3), (6, 1)),
    ((None, None), (15, 18), (None, None), (7, 2)),
    ((16, 21), (None, None), (8, 5), (None, None)),
    ((17, 22), (16, 20), (9, 6), (8, 4)),
    ((18, 23), (17, 21), (10, 7), (9, 5)),
    ((19, None), (18, 22), (11, None), (10, 6)),
    ((21, 25), (20, None), (13, 9), (12, None)),
    ((22, 26), (21, 24), (14, 10), (13, 8)),
    ((23, 27), (22, 25), (15, 11), (14, 9)),
    ((None, None), (23, 26), (None, None), (15, 10)),
    ((24, 29), (None, None), (16, 13), (None, None)),
    ((25, 30), (24, 28), (17, 14), (16, 12)),
    ((26, 31), (25, 29), (18, 15), (17, 13)),
    ((27, None), (26, 30), (19, None), (18, 14)),
    ((29, None), (28, None), (21, 17), (20, None)),
    ((30, None), (29, None), (22, 18), (21, 16)),
    ((31, None), (30, None), (23, 19), (22, 17)),
    ((None, None), (31, None), (None, None), (23, 18)),
    ((None, None), (None, None), (24, 21), (None, None)),
    ((None, None), (None, None), (25, 22), (24, 20)),
    ((None, None), (None, None), (26, 23), (25, 21)),
    ((None, None), (None, None), (27, None), (26, 22)),
)


def _iter_squares(mask: int) -> Iterator[int]:
    remaining = mask
    while remaining:
        least_significant = remaining & -remaining
        yield least_significant.bit_length() - 1
        remaining ^= least_significant


def _directions_for(state: State, square: int) -> tuple[int, ...]:
    actor = int(state.side_to_move)
    if state.kings[actor] & bit(square):
        return KING_DIRECTIONS
    return (
        RED_FORWARD_DIRECTIONS if state.side_to_move is PlayerId.RED else WHITE_FORWARD_DIRECTIONS
    )


def _capture_steps_from(state: State, origin: int) -> tuple[Step, ...]:
    opponent = int(state.side_to_move.opponent)
    opponent_occupied = state.men[opponent] | state.kings[opponent]
    captures: list[Step] = []
    for direction in _directions_for(state, origin):
        adjacent, landing = GEOMETRY[origin][direction]
        if adjacent is None or landing is None:
            continue
        adjacent_bit = bit(adjacent)
        if not adjacent_bit & opponent_occupied:
            continue
        if adjacent_bit & state.captured_pending:
            continue
        if bit(landing) & state.occupied:
            continue
        captures.append(Step(origin=origin, destination=landing, captured=adjacent))
    return tuple(captures)


def _simple_steps_from(state: State, origin: int) -> tuple[Step, ...]:
    moves: list[Step] = []
    for direction in _directions_for(state, origin):
        adjacent, _ = GEOMETRY[origin][direction]
        if adjacent is not None and not bit(adjacent) & state.occupied:
            moves.append(Step(origin=origin, destination=adjacent))
    return tuple(moves)


def _step_sort_key(step: Step) -> tuple[int, int, int]:
    captured = -1 if step.captured is None else step.captured
    return step.origin, step.destination, captured


def legal_steps(state: State) -> tuple[Step, ...]:
    """Return every legal environment step under R2–R5 in deterministic order.

    During a capture sequence, only the forced piece is considered. Otherwise, mandatory capture
    is enforced across all pieces before any simple step is returned.

    Args:
        state: Complete immutable game state.

    Returns:
        A tuple of unique legal steps sorted by origin, destination, and captured square.
    """

    actor = int(state.side_to_move)
    if state.capture_in_progress:
        moving_square = cast(int, state.moving_square)
        return tuple(sorted(_capture_steps_from(state, moving_square), key=_step_sort_key))

    origins = state.men[actor] | state.kings[actor]
    captures = [
        step for origin in _iter_squares(origins) for step in _capture_steps_from(state, origin)
    ]
    if captures:
        return tuple(sorted(captures, key=_step_sort_key))

    moves = [
        step for origin in _iter_squares(origins) for step in _simple_steps_from(state, origin)
    ]
    return tuple(sorted(moves, key=_step_sort_key))


def _moved_boards(state: State, step: Step) -> tuple[tuple[int, int], tuple[int, int], bool]:
    actor = int(state.side_to_move)
    origin_bit = bit(step.origin)
    destination_bit = bit(step.destination)
    men = [state.men[0], state.men[1]]
    kings = [state.kings[0], state.kings[1]]
    was_man = bool(men[actor] & origin_bit)
    if was_man:
        men[actor] = (men[actor] ^ origin_bit) | destination_bit
    else:
        kings[actor] = (kings[actor] ^ origin_bit) | destination_bit
    return (men[0], men[1]), (kings[0], kings[1]), was_man


def _is_king_row(player: PlayerId, square: int) -> bool:
    row, _ = coord(square)
    return row == (7 if player is PlayerId.RED else 0)


def _finish_capture(intermediate: State, was_man: bool) -> State:
    actor = int(intermediate.side_to_move)
    opponent = int(intermediate.side_to_move.opponent)
    men = [intermediate.men[0], intermediate.men[1]]
    kings = [intermediate.kings[0], intermediate.kings[1]]
    men[opponent] &= ~intermediate.captured_pending
    kings[opponent] &= ~intermediate.captured_pending

    destination = cast(int, intermediate.moving_square)
    if was_man and _is_king_row(intermediate.side_to_move, destination):
        destination_bit = bit(destination)
        men[actor] ^= destination_bit
        kings[actor] |= destination_bit

    counters = [intermediate.no_progress[0], intermediate.no_progress[1]]
    counters[actor] = 0
    return State(
        men=(men[0], men[1]),
        kings=(kings[0], kings[1]),
        side_to_move=intermediate.side_to_move.opponent,
        no_progress=(counters[0], counters[1]),
        ply=intermediate.ply,
    )


def _apply_capture(state: State, step: Step) -> Transition:
    captured = cast(int, step.captured)
    men, kings, was_man = _moved_boards(state, step)
    sequence_origin = state.sequence_origin if state.capture_in_progress else step.origin
    intermediate = State(
        men=men,
        kings=kings,
        side_to_move=state.side_to_move,
        capture_in_progress=True,
        moving_square=step.destination,
        sequence_origin=sequence_origin,
        captured_pending=state.captured_pending | bit(captured),
        no_progress=state.no_progress,
        ply=state.ply + 1,
    )

    promotion_ends_move = was_man and _is_king_row(state.side_to_move, step.destination)
    if not promotion_ends_move and _capture_steps_from(intermediate, step.destination):
        return Transition(before=state, after=intermediate, step=step, move_completed=False)
    return Transition(
        before=state,
        after=_finish_capture(intermediate, was_man),
        step=step,
        move_completed=True,
    )


def _apply_simple(state: State, step: Step) -> Transition:
    men, kings, was_man = _moved_boards(state, step)
    actor = int(state.side_to_move)
    if was_man and _is_king_row(state.side_to_move, step.destination):
        destination_bit = bit(step.destination)
        mutable_men = [men[0], men[1]]
        mutable_kings = [kings[0], kings[1]]
        mutable_men[actor] ^= destination_bit
        mutable_kings[actor] |= destination_bit
        men = (mutable_men[0], mutable_men[1])
        kings = (mutable_kings[0], mutable_kings[1])

    counters = [state.no_progress[0], state.no_progress[1]]
    counters[actor] = 0 if was_man else counters[actor] + 1
    after = State(
        men=men,
        kings=kings,
        side_to_move=state.side_to_move.opponent,
        no_progress=(counters[0], counters[1]),
        ply=state.ply + 1,
    )
    return Transition(before=state, after=after, step=step, move_completed=True)


def apply_step(state: State, step: Step) -> Transition:
    """Apply one legal step and retain an exact immutable undo record.

    Args:
        state: Complete state before the step.
        step: Requested simple move or jump.

    Returns:
        The before/after transition and completed-move flag.

    Raises:
        TypeError: If `step` is not a `Step`.
        IllegalStepError: If `step` is not legal in `state`.
    """

    if not isinstance(step, Step):
        raise TypeError("step must be a Step")
    if step not in legal_steps(state):
        raise IllegalStepError("step is not legal in this state")
    return _apply_capture(state, step) if step.is_capture else _apply_simple(state, step)


def undo_step(transition: Transition) -> State:
    """Restore the exact state preceding an immutable transition.

    Args:
        transition: Transition returned by `apply_step`.

    Returns:
        The exact prior `State` value.

    Raises:
        TypeError: If `transition` is not a `Transition`.
    """

    if not isinstance(transition, Transition):
        raise TypeError("transition must be a Transition")
    return transition.before
