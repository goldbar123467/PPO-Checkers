"""Seeded one-step material-greedy baseline."""

from __future__ import annotations

import random

from checkers.agents.base import NoLegalActionError, validate_seed, validate_state
from checkers.env.masking import legal_action_map
from checkers.rules.moves import apply_step
from checkers.rules.state import PlayerId, State

MAN_VALUE = 1
KING_VALUE = 2


def _effective_piece_count(state: State, player: PlayerId) -> tuple[int, int]:
    pending = state.captured_pending if player is state.side_to_move.opponent else 0
    index = int(player)
    return (
        (state.men[index] & ~pending).bit_count(),
        (state.kings[index] & ~pending).bit_count(),
    )


def material_score(state: State, player: PlayerId) -> int:
    """Evaluate material from an explicit player's perspective.

    Pending captures count as removed material even though R4.5 retains their occupied squares
    until the sequence ends.

    Args:
        state: Complete rules state, including any capture continuation.
        player: Perspective used for the signed score.

    Returns:
        ``own men + 2*own kings - opponent men - 2*opponent kings``.

    Raises:
        TypeError: If ``state`` or ``player`` has the wrong runtime type.
    """

    checked_state = validate_state(state)
    if not isinstance(player, PlayerId):
        raise TypeError("player must be a PlayerId")
    own_men, own_kings = _effective_piece_count(checked_state, player)
    opponent_men, opponent_kings = _effective_piece_count(
        checked_state,
        player.opponent,
    )
    return MAN_VALUE * (own_men - opponent_men) + KING_VALUE * (own_kings - opponent_kings)


class GreedyAgent:
    """Choose a successor with maximum immediate material and seeded tie breaks."""

    name = "greedy"

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the private tie-break stream.

        Args:
            seed: Optional integer seed.

        Raises:
            TypeError: If ``seed`` is not an integer or ``None``.
        """

        self._rng = random.Random(validate_seed(seed))

    def select_action(self, state: State) -> int:
        """Return a legal action maximizing immediate material.

        Args:
            state: Nonterminal complete rules state.

        Returns:
            A maximum-scoring legal action with seeded uniform tie-breaking.

        Raises:
            TypeError: If ``state`` is not a ``State``.
            NoLegalActionError: If no legal action exists.
        """

        checked_state = validate_state(state)
        action_map = legal_action_map(checked_state)
        if not action_map:
            raise NoLegalActionError("state has no legal action")
        actor = checked_state.side_to_move
        scored_actions = tuple(
            (
                material_score(apply_step(checked_state, step).after, actor),
                action,
            )
            for action, step in action_map.items()
        )
        best_score = max(score for score, _action in scored_actions)
        best_actions = tuple(action for score, action in scored_actions if score == best_score)
        return self._rng.choice(best_actions)
