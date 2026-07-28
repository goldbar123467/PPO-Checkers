"""Uniform seeded legal-action baseline."""

from __future__ import annotations

import random

from checkers.agents.base import NoLegalActionError, validate_seed, validate_state
from checkers.env.masking import legal_action_map
from checkers.rules.state import State


class RandomAgent:
    """Sample uniformly from the exact legal action set with a private PRNG."""

    name = "random"

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the private deterministic random stream.

        Args:
            seed: Optional integer seed.

        Raises:
            TypeError: If ``seed`` is not an integer or ``None``.
        """

        self._rng = random.Random(validate_seed(seed))

    def select_action(self, state: State) -> int:
        """Sample one legal action uniformly.

        Args:
            state: Nonterminal complete rules state.

        Returns:
            One canonical legal action ID.

        Raises:
            TypeError: If ``state`` is not a ``State``.
            NoLegalActionError: If no legal action exists.
        """

        actions = tuple(legal_action_map(validate_state(state)))
        if not actions:
            raise NoLegalActionError("state has no legal action")
        return self._rng.choice(actions)
