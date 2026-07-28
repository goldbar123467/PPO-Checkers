"""Transparent Elo transforms for head-to-head score estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from checkers.eval.power import MatchScore

ELO_SCALE = 400.0


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 < checked < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    return checked


@dataclass(frozen=True, slots=True)
class EloEstimate:
    """Elo difference and transformed score-confidence bounds."""

    difference: float
    low: float
    high: float
    confidence: float

    def __post_init__(self) -> None:
        for name, value in (
            ("difference", self.difference),
            ("low", self.low),
            ("high", self.high),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if math.isnan(float(value)):
                raise ValueError(f"{name} must not be NaN")
        _confidence(self.confidence)
        if not self.low <= self.difference <= self.high:
            raise ValueError("Elo bounds must be ordered around the difference")


def elo_difference(score: float) -> float:
    """Convert expected score to an Elo rating difference.

    Args:
        score: Expected score in ``[0, 1]``.

    Returns:
        Rating difference, including negative/positive infinity at zero/one.

    Raises:
        TypeError: If ``score`` is not numeric.
        ValueError: If ``score`` is NaN or outside ``[0, 1]``.
    """

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    checked = float(score)
    if math.isnan(checked) or not 0.0 <= checked <= 1.0:
        raise ValueError("score must be between zero and one")
    if checked == 0.0:
        return -math.inf
    if checked == 1.0:
        return math.inf
    return ELO_SCALE * math.log10(checked / (1.0 - checked))


def expected_score(rating_difference: float) -> float:
    """Convert an Elo rating difference to expected score.

    Args:
        rating_difference: Reference rating minus opponent rating.

    Returns:
        Logistic expected score in ``[0, 1]``.

    Raises:
        TypeError: If the difference is not numeric.
        ValueError: If the difference is NaN.
    """

    if isinstance(rating_difference, bool) or not isinstance(
        rating_difference,
        (int, float),
    ):
        raise TypeError("rating_difference must be numeric")
    checked = float(rating_difference)
    if math.isnan(checked):
        raise ValueError("rating_difference must not be NaN")
    if checked == math.inf:
        return 1.0
    if checked == -math.inf:
        return 0.0
    return 1.0 / (1.0 + math.pow(10.0, -checked / ELO_SCALE))


def elo_estimate(score: MatchScore) -> EloEstimate:
    """Transform a match-score interval monotonically into Elo units.

    Args:
        score: Validated match score and confidence interval.

    Returns:
        Elo difference with identically transformed endpoints.

    Raises:
        TypeError: If ``score`` is not a ``MatchScore``.
    """

    if not isinstance(score, MatchScore):
        raise TypeError("score must be a MatchScore")
    return EloEstimate(
        difference=elo_difference(score.score),
        low=elo_difference(score.low),
        high=elo_difference(score.high),
        confidence=score.confidence,
    )
