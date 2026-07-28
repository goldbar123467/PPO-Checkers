"""Full population payoffs, approximate league Elo, cycles, and fixed anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
from typing import Self

import numpy as np
from numpy.typing import NDArray

from checkers.eval.arena import MatchResult
from checkers.eval.elo import ELO_SCALE, elo_difference
from checkers.eval.power import MatchScore

APPROXIMATE_TRANSITIVITY_ASSUMPTION = (
    "normal/delta-method league-Elo CI is valid only under approximate transitivity"
)
STANDARD_NORMAL = NormalDist()
ZERO_SUM_TOLERANCE = 1e-8
MIN_POPULATION_SIZE = 2
CYCLE_SIZE = 3
NEUTRAL_SCORE = 0.5

FloatArray = NDArray[np.float64]


def _name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    checked = _nonnegative_integer(value, field_name)
    if checked < 1:
        raise ValueError(f"{field_name} must be positive")
    return checked


def _open_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 < checked < 1.0:
        raise ValueError(f"{field_name} must be strictly between zero and one")
    return checked


def _nonnegative_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return checked


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{field_name} must be finite")
    return checked


def _positive_finite(value: object, field_name: str) -> float:
    checked = _nonnegative_finite(value, field_name)
    if checked == 0.0:
        raise ValueError(f"{field_name} must be positive")
    return checked


@dataclass(frozen=True, slots=True)
class WDL:
    """Win/draw/loss counts from one explicitly oriented policy perspective."""

    wins: int
    draws: int
    losses: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.wins, "wins")
        _nonnegative_integer(self.draws, "draws")
        _nonnegative_integer(self.losses, "losses")

    @property
    def games(self) -> int:
        """Return total games represented by the cell."""

        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        """Return wins plus half of draws."""

        return self.wins + 0.5 * self.draws

    @property
    def score(self) -> float:
        """Return points divided by games.

        Returns:
            Oriented average game score.

        Raises:
            ValueError: If the W/D/L cell contains no games.
        """

        if self.games < 1:
            raise ValueError("at least one game is required for a score")
        return self.points / self.games

    def reversed(self) -> WDL:
        """Return the same results from the opponent's perspective."""

        return WDL(wins=self.losses, draws=self.draws, losses=self.wins)


@dataclass(frozen=True, slots=True)
class PairResult:
    """One unordered population edge stored from ``first``'s perspective."""

    first: str
    second: str
    wdl: WDL

    def __post_init__(self) -> None:
        first = _name(self.first, "first")
        second = _name(self.second, "second")
        if first == second:
            raise ValueError("pair agents must be distinct")
        if not isinstance(self.wdl, WDL):
            raise TypeError("wdl must be a WDL")
        if self.wdl.games < 1:
            raise ValueError("pair must contain at least one game")


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


@dataclass(frozen=True, slots=True)
class PayoffMatrix:
    """Complete W/D/L matrix with exactly one stored edge per agent pair."""

    agents: tuple[str, ...]
    pairs: tuple[PairResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.agents, tuple):
            raise TypeError("agents must be a tuple")
        if len(self.agents) < MIN_POPULATION_SIZE:
            raise ValueError("payoff matrix requires at least two agents")
        checked_agents = tuple(_name(agent, "agent") for agent in self.agents)
        if len(set(checked_agents)) != len(checked_agents):
            raise ValueError("payoff matrix agents must be unique")
        if not isinstance(self.pairs, tuple):
            raise TypeError("pairs must be a tuple")

        seen: set[tuple[str, str]] = set()
        for pair in self.pairs:
            if not isinstance(pair, PairResult):
                raise TypeError("pairs must contain PairResult values")
            if pair.first not in checked_agents or pair.second not in checked_agents:
                raise ValueError("pair agent is not in the payoff matrix")
            key = _pair_key(pair.first, pair.second)
            if key in seen:
                raise ValueError("payoff matrix contains a duplicate pair")
            seen.add(key)

        expected = {_pair_key(first, second) for first, second in combinations(checked_agents, 2)}
        if seen != expected:
            raise ValueError("payoff matrix must contain one complete result for every pair")

    @classmethod
    def from_matches(
        cls,
        *,
        agents: tuple[str, ...],
        matches: tuple[MatchResult, ...],
    ) -> Self:
        """Build a complete payoff matrix from arena match summaries.

        Args:
            agents: Stable matrix row/column order.
            matches: Exactly one balanced match per unordered pair.

        Returns:
            Validated complete payoff matrix.

        Raises:
            TypeError: If matches is not a tuple or contains another type.
            ValueError: If the resulting population is incomplete or duplicated.
        """

        if not isinstance(matches, tuple):
            raise TypeError("matches must be a tuple")
        pairs: list[PairResult] = []
        for match in matches:
            if not isinstance(match, MatchResult):
                raise TypeError("matches must contain MatchResult values")
            pairs.append(
                PairResult(
                    first=match.first_agent,
                    second=match.second_agent,
                    wdl=WDL(
                        wins=match.wins,
                        draws=match.draws,
                        losses=match.losses,
                    ),
                )
            )
        return cls(agents=agents, pairs=tuple(pairs))

    def _pair(self, first: str, second: str) -> PairResult:
        key = _pair_key(first, second)
        for pair in self.pairs:
            if _pair_key(pair.first, pair.second) == key:
                return pair
        raise RuntimeError("validated payoff matrix is missing a pair")

    def _checked_agent(self, agent: object) -> str:
        checked = _name(agent, "agent")
        if checked not in self.agents:
            raise ValueError(f"unknown payoff-matrix agent {checked!r}")
        return checked

    def wdl(self, row_agent: str, column_agent: str) -> WDL:
        """Return an oriented W/D/L cell.

        Args:
            row_agent: Perspective policy.
            column_agent: Opposing policy.

        Returns:
            Exact counts, with a zero diagonal.

        Raises:
            TypeError: If an agent name is not a string.
            ValueError: If an agent is absent from the matrix.
        """

        row = self._checked_agent(row_agent)
        column = self._checked_agent(column_agent)
        if row == column:
            return WDL(0, 0, 0)
        pair = self._pair(row, column)
        return pair.wdl if pair.first == row else pair.wdl.reversed()

    def score(self, row_agent: str, column_agent: str) -> float:
        """Return an oriented scalar payoff, using 0.5 on the diagonal.

        Args:
            row_agent: Perspective policy.
            column_agent: Opposing policy.

        Returns:
            Score in ``[0, 1]``.

        Raises:
            TypeError: If an agent name is not a string.
            ValueError: If an agent is absent from the matrix.
        """

        if self._checked_agent(row_agent) == self._checked_agent(column_agent):
            return 0.5
        return self.wdl(row_agent, column_agent).score

    def rows(self) -> tuple[tuple[WDL, ...], ...]:
        """Return the full square W/D/L matrix in declared agent order."""

        return tuple(tuple(self.wdl(row, column) for column in self.agents) for row in self.agents)


@dataclass(frozen=True, slots=True)
class EloRating:
    """One zero-mean league rating and approximate confidence bounds."""

    agent: str
    rating: float
    low: float
    high: float

    def __post_init__(self) -> None:
        _name(self.agent, "agent")
        values = tuple(
            _finite_number(value, field_name)
            for value, field_name in (
                (self.rating, "rating"),
                (self.low, "low"),
                (self.high, "high"),
            )
        )
        del values
        if not self.low <= self.rating <= self.high:
            raise ValueError("Elo confidence bounds must be ordered around rating")


@dataclass(frozen=True, slots=True)
class LeagueEloReport:
    """Approximate-transitivity league projection and residual diagnostics."""

    ratings: tuple[EloRating, ...]
    confidence: float
    transitivity_threshold: float
    max_abs_residual: float
    weighted_rmse: float
    approximately_transitive: bool
    ci_assumption: str = APPROXIMATE_TRANSITIVITY_ASSUMPTION

    def __post_init__(self) -> None:
        if not isinstance(self.ratings, tuple) or not all(
            isinstance(rating, EloRating) for rating in self.ratings
        ):
            raise TypeError("ratings must be a tuple of EloRating values")
        if len(self.ratings) < MIN_POPULATION_SIZE or not math.isclose(
            sum(rating.rating for rating in self.ratings),
            0.0,
            abs_tol=ZERO_SUM_TOLERANCE,
        ):
            raise ValueError("league ratings must contain at least two agents and sum to zero")
        if len({rating.agent for rating in self.ratings}) != len(self.ratings):
            raise ValueError("league rating agent names must be unique")
        _open_probability(self.confidence, "confidence")
        _positive_finite(self.transitivity_threshold, "transitivity_threshold")
        _nonnegative_finite(self.max_abs_residual, "max_abs_residual")
        _nonnegative_finite(self.weighted_rmse, "weighted_rmse")
        if not isinstance(self.approximately_transitive, bool):
            raise TypeError("approximately_transitive must be bool")
        if self.ci_assumption != APPROXIMATE_TRANSITIVITY_ASSUMPTION:
            raise ValueError("ci_assumption must retain the approximate-transitivity warning")

    def rating_for(self, agent: str) -> EloRating:
        """Return one agent's rating.

        Args:
            agent: Declared population name.

        Returns:
            Matching rating record.

        Raises:
            ValueError: If the agent is absent.
        """

        checked = _name(agent, "agent")
        for rating in self.ratings:
            if rating.agent == checked:
                return rating
        raise ValueError(f"unknown league agent {checked!r}")


def _finite_score(wdl: WDL) -> tuple[float, int]:
    score = wdl.score
    if score in {0.0, 1.0}:
        return (wdl.points + 0.5) / (wdl.games + 1), wdl.games + 1
    return score, wdl.games


def league_elo(
    matrix: PayoffMatrix,
    *,
    confidence: float = 0.95,
    transitivity_threshold: float = 100.0,
) -> LeagueEloReport:
    """Project complete pairwise scores into one constrained league-Elo vector.

    Interior pair scores use the ordinary Elo transform. A 0/1 sweep receives a half-point
    continuity correction. Pair observations are weighted by a conservative Bernoulli variance;
    the normal/delta-method confidence interval is labelled conditional on approximate
    transitivity and must be read with the returned residual diagnostics.

    Args:
        matrix: Complete W/D/L population matrix.
        confidence: Two-sided normal confidence level.
        transitivity_threshold: Maximum pairwise Elo residual accepted as approximately transitive.

    Returns:
        Zero-mean weighted league ratings, CIs, and transitivity diagnostics.

    Raises:
        TypeError: If inputs have invalid runtime types.
        ValueError: If confidence or the residual threshold is invalid.
    """

    if not isinstance(matrix, PayoffMatrix):
        raise TypeError("matrix must be a PayoffMatrix")
    checked_confidence = _open_probability(confidence, "confidence")
    checked_threshold = _positive_finite(transitivity_threshold, "transitivity_threshold")
    size = len(matrix.agents)
    index = {agent: offset for offset, agent in enumerate(matrix.agents)}
    information: FloatArray = np.zeros((size, size), dtype=np.float64)
    target: FloatArray = np.zeros(size, dtype=np.float64)
    observations: list[tuple[int, int, float, float]] = []

    for pair in matrix.pairs:
        first = index[pair.first]
        second = index[pair.second]
        score, effective_games = _finite_score(pair.wdl)
        difference = elo_difference(score)
        derivative_scale = ELO_SCALE / math.log(10.0)
        variance = derivative_scale**2 / (effective_games * score * (1.0 - score))
        weight = 1.0 / variance
        information[first, first] += weight
        information[second, second] += weight
        information[first, second] -= weight
        information[second, first] -= weight
        target[first] += weight * difference
        target[second] -= weight * difference
        observations.append((first, second, difference, weight))

    augmented: FloatArray = np.zeros((size + 1, size + 1), dtype=np.float64)
    augmented[:size, :size] = information
    augmented[:size, size] = 1.0
    augmented[size, :size] = 1.0
    augmented_target: FloatArray = np.zeros(size + 1, dtype=np.float64)
    augmented_target[:size] = target
    solution = np.linalg.solve(augmented, augmented_target)
    rating_values = solution[:size]
    covariance = np.linalg.inv(augmented)[:size, :size]
    quantile = STANDARD_NORMAL.inv_cdf((1.0 + checked_confidence) / 2.0)
    ratings = tuple(
        EloRating(
            agent=agent,
            rating=float(rating_values[offset]),
            low=float(rating_values[offset])
            - quantile * math.sqrt(max(0.0, float(covariance[offset, offset]))),
            high=float(rating_values[offset])
            + quantile * math.sqrt(max(0.0, float(covariance[offset, offset]))),
        )
        for offset, agent in enumerate(matrix.agents)
    )
    residuals = tuple(
        difference - float(rating_values[first] - rating_values[second])
        for first, second, difference, _weight in observations
    )
    weights = tuple(weight for _first, _second, _difference, weight in observations)
    max_abs_residual = max(abs(residual) for residual in residuals)
    weighted_rmse = math.sqrt(
        sum(
            weight * residual * residual
            for weight, residual in zip(weights, residuals, strict=True)
        )
        / sum(weights)
    )
    return LeagueEloReport(
        ratings=ratings,
        confidence=checked_confidence,
        transitivity_threshold=checked_threshold,
        max_abs_residual=max_abs_residual,
        weighted_rmse=weighted_rmse,
        approximately_transitive=max_abs_residual <= checked_threshold,
    )


@dataclass(frozen=True, slots=True)
class ThreeCycle:
    """Canonical directed three-cycle and its weakest-edge magnitude."""

    agents: tuple[str, str, str]
    edge_margins: tuple[float, float, float]
    magnitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.agents, tuple) or len(self.agents) != CYCLE_SIZE:
            raise TypeError("cycle agents must be a three-name tuple")
        checked_agents = tuple(_name(agent, "cycle agent") for agent in self.agents)
        if len(set(checked_agents)) != CYCLE_SIZE:
            raise ValueError("cycle agents must be distinct")
        if not isinstance(self.edge_margins, tuple) or len(self.edge_margins) != CYCLE_SIZE:
            raise TypeError("edge_margins must be a three-value tuple")
        margins = tuple(_positive_finite(margin, "edge margin") for margin in self.edge_margins)
        if any(margin > NEUTRAL_SCORE for margin in margins):
            raise ValueError("edge margin cannot exceed one half")
        checked_magnitude = _positive_finite(self.magnitude, "magnitude")
        if not math.isclose(checked_magnitude, min(margins), abs_tol=1e-12):
            raise ValueError("cycle magnitude must equal its weakest edge margin")


@dataclass(frozen=True, slots=True)
class ThreeCycleReport:
    """All strict three-cycles plus count and aggregate magnitude."""

    cycles: tuple[ThreeCycle, ...]
    count: int
    total_magnitude: float
    max_magnitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.cycles, tuple) or not all(
            isinstance(cycle, ThreeCycle) for cycle in self.cycles
        ):
            raise TypeError("cycles must be a tuple of ThreeCycle values")
        checked_count = _nonnegative_integer(self.count, "count")
        if checked_count != len(self.cycles):
            raise ValueError("cycle count must equal the number of records")
        expected_total = sum(cycle.magnitude for cycle in self.cycles)
        expected_max = max((cycle.magnitude for cycle in self.cycles), default=0.0)
        if not math.isclose(
            _nonnegative_finite(self.total_magnitude, "total_magnitude"),
            expected_total,
            abs_tol=1e-12,
        ):
            raise ValueError("total_magnitude disagrees with cycle records")
        if not math.isclose(
            _nonnegative_finite(self.max_magnitude, "max_magnitude"),
            expected_max,
            abs_tol=1e-12,
        ):
            raise ValueError("max_magnitude disagrees with cycle records")


def three_cycles(matrix: PayoffMatrix) -> ThreeCycleReport:
    """Enumerate every strict directed three-cycle in a payoff matrix.

    Args:
        matrix: Complete W/D/L population matrix.

    Returns:
        Canonically oriented cycles with weakest-edge and aggregate magnitudes.

    Raises:
        TypeError: If ``matrix`` is not a ``PayoffMatrix``.
    """

    if not isinstance(matrix, PayoffMatrix):
        raise TypeError("matrix must be a PayoffMatrix")
    cycles: list[ThreeCycle] = []
    for first, second, third in combinations(matrix.agents, 3):
        first_second = matrix.score(first, second)
        second_third = matrix.score(second, third)
        third_first = matrix.score(third, first)
        if (
            first_second > NEUTRAL_SCORE
            and second_third > NEUTRAL_SCORE
            and third_first > NEUTRAL_SCORE
        ):
            margins = (
                first_second - NEUTRAL_SCORE,
                second_third - NEUTRAL_SCORE,
                third_first - NEUTRAL_SCORE,
            )
            cycles.append(
                ThreeCycle(
                    agents=(first, second, third),
                    edge_margins=margins,
                    magnitude=min(margins),
                )
            )
        elif (
            first_second < NEUTRAL_SCORE
            and second_third < NEUTRAL_SCORE
            and third_first < NEUTRAL_SCORE
        ):
            margins = (
                NEUTRAL_SCORE - third_first,
                NEUTRAL_SCORE - second_third,
                NEUTRAL_SCORE - first_second,
            )
            cycles.append(
                ThreeCycle(
                    agents=(first, third, second),
                    edge_margins=margins,
                    magnitude=min(margins),
                )
            )
    return ThreeCycleReport(
        cycles=tuple(cycles),
        count=len(cycles),
        total_magnitude=sum(cycle.magnitude for cycle in cycles),
        max_magnitude=max((cycle.magnitude for cycle in cycles), default=0.0),
    )


@dataclass(frozen=True, slots=True)
class AnchorScore:
    """One candidate's complete W/D/L result against a stationary anchor."""

    candidate: str
    anchor: str
    wdl: WDL

    def __post_init__(self) -> None:
        candidate = _name(self.candidate, "candidate")
        anchor = _name(self.anchor, "anchor")
        if candidate == anchor:
            raise ValueError("candidate and anchor must be distinct")
        if not isinstance(self.wdl, WDL):
            raise TypeError("wdl must be a WDL")
        if self.wdl.games < 1:
            raise ValueError("anchor score must contain at least one game")

    @property
    def score(self) -> float:
        """Return the scalar score while retaining W/D/L in the record."""

        return self.wdl.score


def _name_group(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    checked = tuple(_name(value, field_name) for value in values)
    if not checked:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(checked)) != len(checked):
        raise ValueError(f"{field_name} must contain unique names")
    return checked


def fixed_anchor_scores(
    matrix: PayoffMatrix,
    *,
    candidates: tuple[str, ...],
    anchors: tuple[str, ...],
) -> tuple[AnchorScore, ...]:
    """Extract full stationary-anchor results for candidate policies.

    Args:
        matrix: Complete matrix containing candidates and anchors.
        candidates: Policies being evaluated.
        anchors: Fixed, stationary reference policies.

    Returns:
        Candidate-major Cartesian product of W/D/L anchor records.

    Raises:
        TypeError: If inputs have invalid runtime types.
        ValueError: If groups overlap, are empty/duplicated, or contain unknown agents.
    """

    if not isinstance(matrix, PayoffMatrix):
        raise TypeError("matrix must be a PayoffMatrix")
    checked_candidates = _name_group(candidates, "candidates")
    checked_anchors = _name_group(anchors, "anchors")
    if set(checked_candidates) & set(checked_anchors):
        raise ValueError("candidates and anchors must be disjoint")
    for agent in (*checked_candidates, *checked_anchors):
        if agent not in matrix.agents:
            raise ValueError(f"unknown payoff-matrix agent {agent!r}")
    return tuple(
        AnchorScore(candidate=candidate, anchor=anchor, wdl=matrix.wdl(candidate, anchor))
        for candidate in checked_candidates
        for anchor in checked_anchors
    )


@dataclass(frozen=True, slots=True)
class ExploitabilityProxy:
    """Short-budget best-response score against one frozen checkpoint."""

    checkpoint: str
    best_response: str
    training_steps: int
    score: MatchScore

    def __post_init__(self) -> None:
        checkpoint = _name(self.checkpoint, "checkpoint")
        best_response = _name(self.best_response, "best_response")
        if checkpoint == best_response:
            raise ValueError("checkpoint and best_response must be distinct")
        _positive_integer(self.training_steps, "training_steps")
        if not isinstance(self.score, MatchScore):
            raise TypeError("score must be a MatchScore")

    @property
    def proxy_score(self) -> float:
        """Return the best response's score against the frozen checkpoint."""

        return self.score.score

    @property
    def games(self) -> int:
        """Return the number of evaluation games."""

        return self.score.games


def exploitability_proxy(
    best_response_match: MatchResult,
    *,
    training_steps: int,
) -> ExploitabilityProxy:
    """Label an arena match as a short-budget exploitability proxy.

    Args:
        best_response_match: Match whose first agent is the trained response and second is frozen.
        training_steps: Explicit short optimization budget used for the response.

    Returns:
        Immutable labeled proxy record.

    Raises:
        TypeError: If the match is not an arena ``MatchResult``.
        ValueError: If the training budget is invalid.
    """

    if not isinstance(best_response_match, MatchResult):
        raise TypeError("best_response_match must be a MatchResult")
    return ExploitabilityProxy(
        checkpoint=best_response_match.second_agent,
        best_response=best_response_match.first_agent,
        training_steps=training_steps,
        score=best_response_match.score,
    )
