"""Hand-worked power, score-interval, and Elo regression examples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import inf
from typing import cast

import pytest

from checkers.eval.elo import EloEstimate, elo_difference, elo_estimate, expected_score
from checkers.eval.power import (
    MatchScore,
    PowerPlan,
    achieved_normal_power,
    plan_score_test,
    score_interval,
)

EXPECTED_BALANCED_GAMES = 784
EXPECTED_RAW_GAMES = 783
EXPECTED_ELO_DIFFERENCE = 190.84850188786498
EXPECTED_SCORE_LOW = 0.6041514536665332
EXPECTED_SCORE_HIGH = 0.7810511470506724
HAND_WORKED_GAMES = 100


def test_power_plan_for_five_point_effect_is_computed_not_guessed() -> None:
    plan = plan_score_test(
        null_score=0.50,
        alternative_score=0.55,
        alpha=0.05,
        target_power=0.80,
    )

    assert plan.raw_games == EXPECTED_RAW_GAMES
    assert plan.balanced_games == EXPECTED_BALANCED_GAMES
    assert plan.balanced_games % 2 == 0
    assert plan.achieved_power >= plan.target_power
    assert (
        achieved_normal_power(
            games=plan.raw_games - 1,
            null_score=plan.null_score,
            alternative_score=plan.alternative_score,
            alpha=plan.alpha,
        )
        < plan.target_power
    )


def test_power_plan_is_symmetric_for_equal_magnitude_effects() -> None:
    above = plan_score_test(alternative_score=0.55)
    below = plan_score_test(alternative_score=0.45)

    assert above.raw_games == below.raw_games
    assert above.balanced_games == below.balanced_games


def test_match_score_counts_draw_as_half_and_has_hand_worked_wilson_interval() -> None:
    estimate = score_interval(wins=60, draws=20, losses=20)

    assert estimate.games == HAND_WORKED_GAMES
    assert estimate.score == pytest.approx(0.70)
    assert estimate.low == pytest.approx(EXPECTED_SCORE_LOW)
    assert estimate.high == pytest.approx(EXPECTED_SCORE_HIGH)
    assert estimate.confidence == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("wins", "draws", "losses", "score"),
    [
        (1, 0, 0, 1.0),
        (0, 1, 0, 0.5),
        (0, 0, 1, 0.0),
        (2, 2, 2, 0.5),
    ],
)
def test_match_score_definition_matches_wins_plus_half_draws(
    wins: int,
    draws: int,
    losses: int,
    score: float,
) -> None:
    result = score_interval(wins=wins, draws=draws, losses=losses)
    assert result.score == score


def test_elo_difference_and_expected_score_are_exact_inverses() -> None:
    difference = elo_difference(0.75)

    assert difference == pytest.approx(EXPECTED_ELO_DIFFERENCE)
    assert expected_score(difference) == pytest.approx(0.75)
    assert elo_difference(0.5) == 0.0
    assert elo_difference(0.0) == -inf
    assert elo_difference(1.0) == inf
    assert expected_score(-inf) == 0.0
    assert expected_score(inf) == 1.0


def test_elo_interval_is_monotone_transform_of_score_interval() -> None:
    score = score_interval(wins=60, draws=20, losses=20)
    estimate = elo_estimate(score)

    assert estimate.difference == pytest.approx(elo_difference(score.score))
    assert estimate.low == pytest.approx(elo_difference(score.low))
    assert estimate.high == pytest.approx(elo_difference(score.high))
    assert estimate.confidence == score.confidence


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: plan_score_test(null_score=0.0), "null_score"),
        (lambda: plan_score_test(null_score=cast(float, "0.5")), "null_score"),
        (lambda: plan_score_test(alternative_score=1.0), "alternative_score"),
        (lambda: plan_score_test(alternative_score=0.5), "differ"),
        (lambda: plan_score_test(alpha=0.0), "alpha"),
        (lambda: plan_score_test(target_power=1.0), "target_power"),
        (
            lambda: achieved_normal_power(
                games=cast(int, True),
                null_score=0.5,
                alternative_score=0.55,
                alpha=0.05,
            ),
            "games",
        ),
        (
            lambda: achieved_normal_power(
                games=0,
                null_score=0.5,
                alternative_score=0.55,
                alpha=0.05,
            ),
            "games",
        ),
        (
            lambda: achieved_normal_power(
                games=10,
                null_score=0.5,
                alternative_score=0.5,
                alpha=0.05,
            ),
            "differ",
        ),
        (lambda: score_interval(wins=cast(int, 1.0), draws=0, losses=0), "wins"),
        (lambda: score_interval(wins=-1, draws=0, losses=1), "wins"),
        (lambda: score_interval(wins=0, draws=0, losses=0), "game"),
        (lambda: score_interval(wins=1, draws=0, losses=0, confidence=1.0), "confidence"),
        (lambda: elo_difference(-0.1), "score"),
        (lambda: expected_score(float("nan")), "rating_difference"),
        (lambda: expected_score(cast(float, "zero")), "rating_difference"),
        (lambda: elo_difference(cast(float, "half")), "score"),
        (lambda: elo_estimate(cast(MatchScore, "score")), "MatchScore"),
    ],
)
def test_statistical_functions_reject_invalid_inputs(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_public_statistical_records_validate_invariants() -> None:
    with pytest.raises(ValueError, match="balanced_games"):
        PowerPlan(
            null_score=0.5,
            alternative_score=0.55,
            alpha=0.05,
            target_power=0.8,
            raw_games=10,
            balanced_games=11,
            achieved_power=0.8,
        )
    with pytest.raises(ValueError, match="interval"):
        MatchScore(
            wins=1,
            draws=0,
            losses=0,
            score=1.0,
            low=1.0,
            high=0.9,
            confidence=0.95,
        )
    with pytest.raises(ValueError, match="ordered"):
        EloEstimate(difference=0.0, low=1.0, high=-1.0, confidence=0.95)


def _valid_match_score() -> MatchScore:
    return MatchScore(
        wins=1,
        draws=0,
        losses=1,
        score=0.5,
        low=0.1,
        high=0.9,
        confidence=0.95,
    )


def _valid_power_plan() -> PowerPlan:
    return PowerPlan(
        null_score=0.5,
        alternative_score=0.55,
        alpha=0.05,
        target_power=0.8,
        raw_games=10,
        balanced_games=10,
        achieved_power=0.8,
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(_valid_match_score(), wins=0, losses=0), "game"),
        (lambda: replace(_valid_match_score(), score=0.6), "disagrees"),
        (
            lambda: replace(_valid_match_score(), score=cast(float, "score")),
            "score",
        ),
        (lambda: replace(_valid_match_score(), low=-0.1), "interval"),
        (
            lambda: replace(
                _valid_power_plan(),
                alternative_score=0.5,
            ),
            "differ",
        ),
        (lambda: replace(_valid_power_plan(), raw_games=0), "raw_games"),
        (lambda: replace(_valid_power_plan(), achieved_power=1.1), "achieved_power"),
        (
            lambda: EloEstimate(
                difference=cast(float, "zero"),
                low=-1.0,
                high=1.0,
                confidence=0.95,
            ),
            "difference",
        ),
        (
            lambda: EloEstimate(
                difference=float("nan"),
                low=-1.0,
                high=1.0,
                confidence=0.95,
            ),
            "NaN",
        ),
        (
            lambda: EloEstimate(
                difference=0.0,
                low=-1.0,
                high=1.0,
                confidence=cast(float, "confidence"),
            ),
            "confidence",
        ),
        (
            lambda: EloEstimate(
                difference=0.0,
                low=-1.0,
                high=1.0,
                confidence=1.0,
            ),
            "confidence",
        ),
    ],
)
def test_saved_statistical_records_reject_corruption(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
