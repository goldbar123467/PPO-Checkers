"""Canonical action IDs and legal-action masks for the checkers environment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from checkers.rules.board import PLAYABLE_SQUARES, coord, rotate_square
from checkers.rules.moves import DIRECTION_DELTAS, Step, legal_steps
from checkers.rules.state import PlayerId, State

DIRECTIONS_PER_SQUARE = 4
ACTION_COUNT = PLAYABLE_SQUARES * DIRECTIONS_PER_SQUARE

BoolArray = NDArray[np.bool_]


class ActionEncodingError(ValueError):
    """Raised when an action ID cannot identify a legal step in a state."""


def _bounded_integer(value: object, name: str, upper_bound: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    checked = int(value)
    if not 0 <= checked < upper_bound:
        raise ValueError(f"{name} must be in [0, {upper_bound - 1}]")
    return checked


def _require_state(state: object) -> State:
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    return state


def encode_action_id(square: int, direction: int) -> int:
    """Encode canonical origin and direction components as one action ID.

    Args:
        square: Canonical zero-based ACF origin in ``[0, 31]``.
        direction: Canonical direction index in ``[0, 3]``.

    Returns:
        The corresponding action ID in ``[0, 127]``.

    Raises:
        TypeError: If a component is not an integer.
        ValueError: If a component is outside its valid range.
    """

    checked_square = _bounded_integer(square, "square", PLAYABLE_SQUARES)
    checked_direction = _bounded_integer(
        direction,
        "direction",
        DIRECTIONS_PER_SQUARE,
    )
    return checked_square * DIRECTIONS_PER_SQUARE + checked_direction


def decode_action_id(action: int) -> tuple[int, int]:
    """Decode an action ID into canonical origin and direction components.

    Args:
        action: Action ID in ``[0, 127]``.

    Returns:
        ``(canonical_square, canonical_direction)``.

    Raises:
        TypeError: If ``action`` is not an integer.
        ValueError: If ``action`` is outside ``[0, 127]``.
    """

    checked_action = _bounded_integer(action, "action", ACTION_COUNT)
    return divmod(checked_action, DIRECTIONS_PER_SQUARE)


def _world_direction(step: Step) -> int:
    origin_row, origin_column = coord(step.origin)
    destination_row, destination_column = coord(step.destination)
    row_delta = destination_row - origin_row
    column_delta = destination_column - origin_column
    distance = 2 if step.is_capture else 1
    if abs(row_delta) != distance or abs(column_delta) != distance:
        raise ActionEncodingError("step is not a short diagonal move or jump")
    normalized = (row_delta // distance, column_delta // distance)
    return DIRECTION_DELTAS.index(normalized)


def _encode_step_geometry(state: State, step: Step) -> int:
    direction = _world_direction(step)
    if state.side_to_move is PlayerId.WHITE:
        return encode_action_id(
            rotate_square(step.origin),
            DIRECTIONS_PER_SQUARE - 1 - direction,
        )
    return encode_action_id(step.origin, direction)


def step_to_action(state: State, step: Step) -> int:
    """Encode one legal step in the acting player's canonical frame.

    Args:
        state: State in which the step must be legal.
        step: One legal simple move or short jump.

    Returns:
        The unique canonical action ID for ``step``.

    Raises:
        TypeError: If either argument has the wrong runtime type.
        ActionEncodingError: If ``step`` is not legal in ``state``.
    """

    checked_state = _require_state(state)
    if not isinstance(step, Step):
        raise TypeError("step must be a Step")
    if step not in legal_steps(checked_state):
        raise ActionEncodingError("step must be legal in state")
    return _encode_step_geometry(checked_state, step)


def legal_action_map(state: State) -> dict[int, Step]:
    """Return the state's deterministic mapping from legal action IDs to steps.

    Args:
        state: Complete immutable game state.

    Returns:
        A dictionary in legal-step order. It is empty exactly when there are no legal steps.

    Raises:
        TypeError: If ``state`` is not a ``State``.
        RuntimeError: If distinct legal steps ever collide under the frozen encoding.
    """

    checked_state = _require_state(state)
    action_map: dict[int, Step] = {}
    for step in legal_steps(checked_state):
        action = _encode_step_geometry(checked_state, step)
        if action in action_map:
            raise RuntimeError("distinct legal steps collided under action encoding")
        action_map[action] = step
    return action_map


def action_to_step(state: State, action: int) -> Step:
    """Resolve a canonical action ID to its legal step in ``state``.

    Args:
        state: State in which the action must be legal.
        action: Canonical action ID in ``[0, 127]``.

    Returns:
        The single legal step identified by ``action``.

    Raises:
        TypeError: If ``state`` or ``action`` has the wrong runtime type.
        ValueError: If ``action`` is outside ``[0, 127]``.
        ActionEncodingError: If the in-range action is not legal in ``state``.
    """

    checked_action = _bounded_integer(action, "action", ACTION_COUNT)
    action_map = legal_action_map(state)
    try:
        return action_map[checked_action]
    except KeyError as error:
        raise ActionEncodingError("action does not identify a legal step in state") from error


def legal_action_mask(state: State) -> BoolArray:
    """Encode all legal steps as a fixed-width Boolean mask.

    Args:
        state: Complete immutable game state.

    Returns:
        A fresh Boolean NumPy array of shape ``(128,)``.

    Raises:
        TypeError: If ``state`` is not a ``State``.
    """

    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    for action in legal_action_map(state):
        mask[action] = True
    return mask
