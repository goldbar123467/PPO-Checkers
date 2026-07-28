"""Typed interface and shared validation for checkers agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from checkers.rules.state import State


class NoLegalActionError(ValueError):
    """Raised when an agent is asked to act in a state with no legal action."""


@runtime_checkable
class Agent(Protocol):
    """Minimal policy interface consumed by arenas and tactical suites."""

    name: str

    def select_action(self, state: State) -> int:
        """Return one legal canonical action for ``state``."""


def validate_state(state: State) -> State:
    """Require an exact immutable checkers state.

    Args:
        state: Candidate runtime value.

    Returns:
        The validated state unchanged.

    Raises:
        TypeError: If ``state`` is not a ``State``.
    """

    if not isinstance(state, State):
        raise TypeError("state must be a State")
    return state


def validate_seed(seed: int | None) -> int | None:
    """Validate an optional deterministic PRNG seed.

    Args:
        seed: Integer seed or ``None`` for system initialization.

    Returns:
        The validated seed unchanged.

    Raises:
        TypeError: If ``seed`` is neither an integer nor ``None``.
    """

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    return seed
