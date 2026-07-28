"""Score confidence intervals and explicit normal-approximation power plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

STANDARD_NORMAL = NormalDist()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    checked = _integer(value, name)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _open_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 < checked < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one")
    return checked


def _closed_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return checked


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Wins/draws/losses, scalar score, and a two-sided confidence interval."""

    wins: int
    draws: int
    losses: int
    score: float
    low: float
    high: float
    confidence: float

    def __post_init__(self) -> None:
        wins = _nonnegative_integer(self.wins, "wins")
        draws = _nonnegative_integer(self.draws, "draws")
        losses = _nonnegative_integer(self.losses, "losses")
        games = wins + draws + losses
        if games < 1:
            raise ValueError("at least one game is required")
        score = _closed_probability(self.score, "score")
        low = _closed_probability(self.low, "interval low")
        high = _closed_probability(self.high, "interval high")
        _open_probability(self.confidence, "confidence")
        expected_score = (wins + 0.5 * draws) / games
        if not math.isclose(score, expected_score, abs_tol=1e-12):
            raise ValueError("score disagrees with wins plus half draws")
        if not low <= score <= high:
            raise ValueError("confidence interval must contain score in ordered bounds")

    @property
    def games(self) -> int:
        """Return the total number of games."""

        return self.wins + self.draws + self.losses


@dataclass(frozen=True, slots=True)
class PowerPlan:
    """Normal-approximation game-count plan for one head-to-head score."""

    null_score: float
    alternative_score: float
    alpha: float
    target_power: float
    raw_games: int
    balanced_games: int
    achieved_power: float

    def __post_init__(self) -> None:
        null_score = _open_probability(self.null_score, "null_score")
        alternative_score = _open_probability(
            self.alternative_score,
            "alternative_score",
        )
        if null_score == alternative_score:
            raise ValueError("null and alternative scores must differ")
        _open_probability(self.alpha, "alpha")
        _open_probability(self.target_power, "target_power")
        raw_games = _integer(self.raw_games, "raw_games")
        balanced_games = _integer(self.balanced_games, "balanced_games")
        if raw_games < 1:
            raise ValueError("raw_games must be positive")
        if balanced_games < raw_games or balanced_games % 2 != 0:
            raise ValueError("balanced_games must be even and at least raw_games")
        _closed_probability(self.achieved_power, "achieved_power")


def achieved_normal_power(
    *,
    games: int,
    null_score: float,
    alternative_score: float,
    alpha: float,
) -> float:
    """Return approximate two-sided one-sample score-test power.

    The calculation conservatively treats each bounded game score as a Bernoulli outcome. Draws
    usually reduce variance relative to that assumption.

    Args:
        games: Positive number of independent games.
        null_score: Score under the null hypothesis.
        alternative_score: Smallest score worth detecting.
        alpha: Two-sided Type-I error probability.

    Returns:
        Normal-approximation detection probability in ``[0, 1]``.

    Raises:
        TypeError: If an input has the wrong runtime type.
        ValueError: If counts or probabilities are outside their valid ranges.
    """

    checked_games = _integer(games, "games")
    if checked_games < 1:
        raise ValueError("games must be positive")
    null = _open_probability(null_score, "null_score")
    alternative = _open_probability(alternative_score, "alternative_score")
    if null == alternative:
        raise ValueError("null and alternative scores must differ")
    checked_alpha = _open_probability(alpha, "alpha")
    critical = STANDARD_NORMAL.inv_cdf(1.0 - checked_alpha / 2.0)
    standardized = (
        abs(alternative - null) * math.sqrt(checked_games)
        - critical * math.sqrt(null * (1.0 - null))
    ) / math.sqrt(alternative * (1.0 - alternative))
    return STANDARD_NORMAL.cdf(standardized)


def plan_score_test(
    *,
    null_score: float = 0.50,
    alternative_score: float = 0.55,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> PowerPlan:
    """Compute and colour-balance a one-sample score-test game budget.

    Args:
        null_score: Head-to-head score under no difference.
        alternative_score: Smallest effect worth detecting.
        alpha: Two-sided Type-I error probability.
        target_power: Required detection probability at the alternative.

    Returns:
        Raw ceiling and next even game count with achieved approximate power.

    Raises:
        TypeError: If an input has the wrong runtime type.
        ValueError: If probabilities are invalid or null and alternative are equal.
    """

    null = _open_probability(null_score, "null_score")
    alternative = _open_probability(alternative_score, "alternative_score")
    if null == alternative:
        raise ValueError("null and alternative scores must differ")
    checked_alpha = _open_probability(alpha, "alpha")
    checked_power = _open_probability(target_power, "target_power")
    critical = STANDARD_NORMAL.inv_cdf(1.0 - checked_alpha / 2.0)
    power_quantile = STANDARD_NORMAL.inv_cdf(checked_power)
    numerator = critical * math.sqrt(null * (1.0 - null)) + power_quantile * math.sqrt(
        alternative * (1.0 - alternative)
    )
    raw_games = math.ceil((numerator / abs(alternative - null)) ** 2)
    balanced_games = raw_games + raw_games % 2
    return PowerPlan(
        null_score=null,
        alternative_score=alternative,
        alpha=checked_alpha,
        target_power=checked_power,
        raw_games=raw_games,
        balanced_games=balanced_games,
        achieved_power=achieved_normal_power(
            games=balanced_games,
            null_score=null,
            alternative_score=alternative,
            alpha=checked_alpha,
        ),
    )


def score_interval(
    *,
    wins: int,
    draws: int,
    losses: int,
    confidence: float = 0.95,
) -> MatchScore:
    """Compute score and a Wilson-style interval with each draw worth half.

    Args:
        wins: Non-negative wins for the reference agent.
        draws: Non-negative draws.
        losses: Non-negative losses for the reference agent.
        confidence: Two-sided confidence level.

    Returns:
        Validated counts, score, and interval.

    Raises:
        TypeError: If counts are not integers or confidence is not numeric.
        ValueError: If counts are negative/empty or confidence is invalid.
    """

    checked_wins = _nonnegative_integer(wins, "wins")
    checked_draws = _nonnegative_integer(draws, "draws")
    checked_losses = _nonnegative_integer(losses, "losses")
    games = checked_wins + checked_draws + checked_losses
    if games < 1:
        raise ValueError("at least one game is required")
    checked_confidence = _open_probability(confidence, "confidence")
    score = (checked_wins + 0.5 * checked_draws) / games
    z_value = STANDARD_NORMAL.inv_cdf((1.0 + checked_confidence) / 2.0)
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / games
    center = (score + z_squared / (2.0 * games)) / denominator
    half_width = (
        z_value
        / denominator
        * math.sqrt(score * (1.0 - score) / games + z_squared / (4.0 * games * games))
    )
    return MatchScore(
        wins=checked_wins,
        draws=checked_draws,
        losses=checked_losses,
        score=score,
        low=max(0.0, center - half_width),
        high=min(1.0, center + half_width),
        confidence=checked_confidence,
    )
