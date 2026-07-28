"""Terminal outcomes for American Checkers and declared engine variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from checkers.rules.moves import legal_steps
from checkers.rules.state import PlayerId, State

DEFAULT_MAX_PLIES = 512
NO_PROGRESS_LIMIT = 40
REPETITION_LIMIT = 3


class TerminationReason(StrEnum):
    """Stable reason labels for terminal metrics and serialized reports."""

    NO_PIECES = "no_pieces"
    NO_LEGAL_MOVE = "no_legal_move"
    NO_PROGRESS = "no_progress"
    REPETITION = "repetition"
    PLY_CAP = "ply_cap"


DRAW_REASONS = frozenset(
    {
        TerminationReason.NO_PROGRESS,
        TerminationReason.REPETITION,
        TerminationReason.PLY_CAP,
    }
)


@dataclass(frozen=True, slots=True)
class Outcome:
    """A terminal winner (or draw) paired with its exact rule reason.

    Args:
        winner: Winning player, or None for a draw.
        reason: Rule that made the state terminal.

    Raises:
        TypeError: If runtime values do not use the public enum types.
        ValueError: If winner and reason disagree about whether this is a draw.
    """

    winner: PlayerId | None
    reason: TerminationReason

    def __post_init__(self) -> None:
        if self.winner is not None and not isinstance(self.winner, PlayerId):
            raise TypeError("winner must be a PlayerId or None")
        if not isinstance(self.reason, TerminationReason):
            raise TypeError("reason must be a TerminationReason")
        if (self.winner is None) != (self.reason in DRAW_REASONS):
            raise ValueError("winner and termination reason disagree about draw status")

    @property
    def is_draw(self) -> bool:
        """Return whether neither player won."""

        return self.winner is None

    def score_for(self, player: PlayerId) -> int:
        """Return this outcome as +1, 0, or -1 from one player's perspective.

        Args:
            player: Explicit player whose perspective is requested.

        Returns:
            One for a win, zero for a draw, and minus one for a loss.

        Raises:
            TypeError: If player is not a PlayerId.
        """

        if not isinstance(player, PlayerId):
            raise TypeError("player must be a PlayerId")
        if self.winner is None:
            return 0
        return 1 if self.winner is player else -1


def _validate_options(max_plies: int, repetition_draws: bool, repetition_count: int) -> None:
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    if not isinstance(repetition_draws, bool):
        raise TypeError("repetition_draws must be bool")
    if isinstance(repetition_count, bool) or not isinstance(repetition_count, int):
        raise TypeError("repetition_count must be an integer")
    if repetition_count < 0:
        raise ValueError("repetition_count must be non-negative")


def terminal_outcome(
    state: State,
    *,
    max_plies: int = DEFAULT_MAX_PLIES,
    repetition_draws: bool = False,
    repetition_count: int = 0,
) -> Outcome | None:
    """Return the state's terminal outcome, or None while play can continue.

    Losses are evaluated before engine-variant draws when boundaries coincide. Repetition is an
    arena-only option and is ignored during capture substates, because official positions are
    counted only at completed-move boundaries. No draw-by-agreement input exists: R6.6 is an
    explicit autonomous-engine departure from WCDF 1.32.

    Args:
        state: Complete immutable game state.
        max_plies: R6.5 environment-step draw boundary.
        repetition_draws: Enable the optional R6.4 arena rule.
        repetition_count: Visits to this completed-move official position, including this visit.

    Returns:
        A terminal Outcome, or None when the state is non-terminal.

    Raises:
        TypeError: If state or an option has the wrong runtime type.
        ValueError: If a numeric option is outside its valid range.
    """

    if not isinstance(state, State):
        raise TypeError("state must be a State")
    _validate_options(max_plies, repetition_draws, repetition_count)

    actor = int(state.side_to_move)
    if not (state.men[actor] | state.kings[actor]):
        return Outcome(winner=state.side_to_move.opponent, reason=TerminationReason.NO_PIECES)
    if not legal_steps(state):
        return Outcome(winner=state.side_to_move.opponent, reason=TerminationReason.NO_LEGAL_MOVE)
    if all(counter >= NO_PROGRESS_LIMIT for counter in state.no_progress):
        return Outcome(winner=None, reason=TerminationReason.NO_PROGRESS)
    if repetition_draws and not state.capture_in_progress and repetition_count >= REPETITION_LIMIT:
        return Outcome(winner=None, reason=TerminationReason.REPETITION)
    if state.ply >= max_plies:
        return Outcome(winner=None, reason=TerminationReason.PLY_CAP)
    return None
