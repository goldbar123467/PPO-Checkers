"""Hand-worked payoff, league-Elo, cycle, and anchor examples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import isclose
from typing import cast

import pytest

from checkers.env.masking import legal_action_map
from checkers.eval.arena import AgentSpec, play_balanced_match
from checkers.eval.elo import elo_difference
from checkers.eval.population import (
    APPROXIMATE_TRANSITIVITY_ASSUMPTION,
    WDL,
    AnchorScore,
    EloRating,
    ExploitabilityProxy,
    LeagueEloReport,
    PairResult,
    PayoffMatrix,
    ThreeCycle,
    ThreeCycleReport,
    exploitability_proxy,
    fixed_anchor_scores,
    league_elo,
    three_cycles,
)
from checkers.eval.power import score_interval
from checkers.rules.state import PlayerId, State

EXPECTED_PAIR_ELO = 190.84850188786498
TRANSITIVITY_THRESHOLD = 100.0
HAND_GAMES = 10
HAND_POINTS = 7.0
NEUTRAL_SCORE = 0.5
PAIR_SCORE = 0.75
CYCLE_MARGIN = 0.25
STRONG_SCORE = 0.9
PROXY_SCORE = 0.7


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


class FirstLegalAgent:
    """Minimal policy used only to construct real match inputs."""

    def __init__(self, name: str) -> None:
        self.name = name

    def select_action(self, state: State) -> int:
        return next(iter(legal_action_map(state)))


def _spec(name: str) -> AgentSpec:
    return AgentSpec(name=name, factory=lambda _seed: FirstLegalAgent(name))


def _transitive_matrix() -> PayoffMatrix:
    return PayoffMatrix(
        agents=("A", "B", "C"),
        pairs=(
            PairResult("A", "B", WDL(wins=3, draws=0, losses=1)),
            PairResult("B", "C", WDL(wins=3, draws=0, losses=1)),
            PairResult("A", "C", WDL(wins=9, draws=0, losses=1)),
        ),
    )


def _cycle_matrix() -> PayoffMatrix:
    return PayoffMatrix(
        agents=("A", "B", "C"),
        pairs=(
            PairResult("A", "B", WDL(wins=3, draws=0, losses=1)),
            PairResult("B", "C", WDL(wins=3, draws=0, losses=1)),
            PairResult("C", "A", WDL(wins=3, draws=0, losses=1)),
        ),
    )


def test_wdl_uses_wins_plus_half_draws_and_reverses_exactly() -> None:
    result = WDL(wins=6, draws=2, losses=2)

    assert result.games == HAND_GAMES
    assert result.points == HAND_POINTS
    assert result.score == PROXY_SCORE
    assert result.reversed() == WDL(wins=2, draws=2, losses=6)


def test_payoff_matrix_exposes_full_oriented_wdl_rows() -> None:
    matrix = _transitive_matrix()

    assert matrix.wdl("A", "B") == WDL(3, 0, 1)
    assert matrix.wdl("B", "A") == WDL(1, 0, 3)
    assert matrix.wdl("A", "A") == WDL(0, 0, 0)
    assert matrix.score("A", "A") == NEUTRAL_SCORE
    assert matrix.score("A", "B") == PAIR_SCORE
    assert matrix.rows()[0] == (
        WDL(0, 0, 0),
        WDL(3, 0, 1),
        WDL(9, 0, 1),
    )


def test_payoff_matrix_builds_from_real_balanced_match_results() -> None:
    forced_red_win = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    match = play_balanced_match(
        first=_spec("A"),
        second=_spec("B"),
        games=4,
        seed=9,
        initial_state=forced_red_win,
    )

    matrix = PayoffMatrix.from_matches(agents=("A", "B"), matches=(match,))

    assert matrix.wdl("A", "B") == WDL(2, 0, 2)
    assert matrix.score("A", "B") == NEUTRAL_SCORE


def test_two_agent_league_elo_matches_hand_worked_pair_difference() -> None:
    matrix = PayoffMatrix(
        agents=("A", "B"),
        pairs=(PairResult("A", "B", WDL(3, 0, 1)),),
    )
    report = league_elo(matrix)
    ratings = {rating.agent: rating for rating in report.ratings}

    assert ratings["A"].rating == pytest.approx(EXPECTED_PAIR_ELO / 2.0)
    assert ratings["B"].rating == pytest.approx(-EXPECTED_PAIR_ELO / 2.0)
    assert ratings["A"].rating - ratings["B"].rating == pytest.approx(elo_difference(0.75))
    assert ratings["A"].low < ratings["A"].rating < ratings["A"].high
    assert report.approximately_transitive is True


def test_additive_hand_worked_population_has_zero_transitivity_residual() -> None:
    report = league_elo(_transitive_matrix())
    ratings = {rating.agent: rating.rating for rating in report.ratings}

    assert ratings["A"] == pytest.approx(EXPECTED_PAIR_ELO)
    assert ratings["B"] == pytest.approx(0.0, abs=1e-12)
    assert ratings["C"] == pytest.approx(-EXPECTED_PAIR_ELO)
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-12)
    assert report.max_abs_residual == pytest.approx(0.0, abs=1e-12)
    assert report.weighted_rmse == pytest.approx(0.0, abs=1e-12)
    assert report.approximately_transitive is True
    assert report.ci_assumption == APPROXIMATE_TRANSITIVITY_ASSUMPTION


def test_three_cycle_is_not_hidden_by_scalar_league_elo() -> None:
    report = league_elo(_cycle_matrix())

    assert all(rating.rating == pytest.approx(0.0, abs=1e-12) for rating in report.ratings)
    assert report.max_abs_residual == pytest.approx(EXPECTED_PAIR_ELO)
    assert report.weighted_rmse == pytest.approx(EXPECTED_PAIR_ELO)
    assert report.approximately_transitive is False
    assert (
        league_elo(
            _cycle_matrix(),
            transitivity_threshold=200.0,
        ).approximately_transitive
        is True
    )


def test_three_cycle_count_orientation_and_magnitude_are_hand_worked() -> None:
    report = three_cycles(_cycle_matrix())

    assert report.count == 1
    assert report.cycles == (
        ThreeCycle(
            agents=("A", "B", "C"),
            edge_margins=(CYCLE_MARGIN, CYCLE_MARGIN, CYCLE_MARGIN),
            magnitude=CYCLE_MARGIN,
        ),
    )
    assert report.total_magnitude == CYCLE_MARGIN
    assert report.max_magnitude == CYCLE_MARGIN
    assert three_cycles(_transitive_matrix()).count == 0


def test_reverse_three_cycle_has_canonical_winning_orientation() -> None:
    reverse = PayoffMatrix(
        agents=("A", "B", "C"),
        pairs=(
            PairResult("B", "A", WDL(3, 0, 1)),
            PairResult("C", "B", WDL(3, 0, 1)),
            PairResult("A", "C", WDL(3, 0, 1)),
        ),
    )

    assert three_cycles(reverse).cycles[0].agents == ("A", "C", "B")


def test_fixed_anchor_scores_preserve_wdl_not_only_scalar_score() -> None:
    scores = fixed_anchor_scores(
        _transitive_matrix(),
        candidates=("A", "B"),
        anchors=("C",),
    )

    assert scores == (
        AnchorScore(candidate="A", anchor="C", wdl=WDL(9, 0, 1)),
        AnchorScore(candidate="B", anchor="C", wdl=WDL(3, 0, 1)),
    )
    assert scores[0].score == STRONG_SCORE


def test_exploitability_proxy_labels_short_budget_best_response_match() -> None:
    proxy = ExploitabilityProxy(
        checkpoint="frozen-10",
        best_response="br-10",
        training_steps=500,
        score=score_interval(wins=6, draws=2, losses=2),
    )

    assert proxy.proxy_score == PROXY_SCORE
    assert proxy.games == HAND_GAMES


def test_exploitability_proxy_factory_uses_first_agent_as_best_response() -> None:
    forced_red_win = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    match = play_balanced_match(
        first=_spec("best-response"),
        second=_spec("checkpoint"),
        games=2,
        seed=1,
        initial_state=forced_red_win,
    )

    proxy = exploitability_proxy(match, training_steps=100)

    assert proxy.best_response == "best-response"
    assert proxy.checkpoint == "checkpoint"
    assert proxy.score == match.score


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WDL(wins=-1, draws=0, losses=0), "wins"),
        (lambda: WDL(wins=cast(int, True), draws=0, losses=0), "wins"),
        (lambda: WDL(0, 0, 0).score, "game"),
        (lambda: PairResult(cast(str, 1), "B", WDL(1, 0, 0)), "first"),
        (lambda: PairResult("", "B", WDL(1, 0, 0)), "first"),
        (lambda: PairResult("A", "A", WDL(1, 0, 0)), "distinct"),
        (lambda: PairResult("A", "B", cast(WDL, "wdl")), "WDL"),
        (lambda: PairResult("A", "B", WDL(0, 0, 0)), "game"),
        (lambda: PayoffMatrix(agents=cast(tuple[str, ...], ["A", "B"]), pairs=()), "agents"),
        (lambda: PayoffMatrix(agents=("A", "A"), pairs=()), "unique"),
        (lambda: PayoffMatrix(agents=("A",), pairs=()), "two"),
        (lambda: PayoffMatrix(agents=("A", "B"), pairs=()), "complete"),
        (
            lambda: PayoffMatrix(
                agents=("A", "B"),
                pairs=cast(tuple[PairResult, ...], []),
            ),
            "pairs",
        ),
        (
            lambda: PayoffMatrix(
                agents=("A", "B"),
                pairs=(cast(PairResult, "pair"),),
            ),
            "PairResult",
        ),
        (
            lambda: PayoffMatrix(
                agents=("A", "B"),
                pairs=(PairResult("A", "C", WDL(1, 0, 0)),),
            ),
            "matrix",
        ),
        (
            lambda: PayoffMatrix(
                agents=("A", "B"),
                pairs=(
                    PairResult("A", "B", WDL(1, 0, 0)),
                    PairResult("B", "A", WDL(0, 0, 1)),
                ),
            ),
            "duplicate",
        ),
        (lambda: _transitive_matrix().wdl("A", "missing"), "agent"),
        (
            lambda: PayoffMatrix.from_matches(
                agents=("A", "B"),
                matches=cast(tuple[object, ...], []),  # type: ignore[arg-type]
            ),
            "matches",
        ),
        (
            lambda: PayoffMatrix.from_matches(
                agents=("A", "B"),
                matches=(cast(object, "match"),),  # type: ignore[arg-type]
            ),
            "MatchResult",
        ),
        (lambda: league_elo(cast(PayoffMatrix, "matrix")), "PayoffMatrix"),
        (lambda: league_elo(_transitive_matrix(), confidence=cast(float, "95%")), "confidence"),
        (lambda: league_elo(_transitive_matrix(), confidence=1.0), "confidence"),
        (
            lambda: league_elo(
                _transitive_matrix(),
                transitivity_threshold=cast(float, "wide"),
            ),
            "threshold",
        ),
        (
            lambda: league_elo(_transitive_matrix(), transitivity_threshold=0.0),
            "threshold",
        ),
        (lambda: three_cycles(cast(PayoffMatrix, "matrix")), "PayoffMatrix"),
        (
            lambda: fixed_anchor_scores(
                cast(PayoffMatrix, "matrix"),
                candidates=("A",),
                anchors=("C",),
            ),
            "PayoffMatrix",
        ),
        (
            lambda: fixed_anchor_scores(
                _transitive_matrix(),
                candidates=cast(tuple[str, ...], ["A"]),
                anchors=("C",),
            ),
            "candidates",
        ),
        (
            lambda: fixed_anchor_scores(
                _transitive_matrix(),
                candidates=(),
                anchors=("C",),
            ),
            "candidates",
        ),
        (
            lambda: fixed_anchor_scores(
                _transitive_matrix(),
                candidates=("A", "A"),
                anchors=("C",),
            ),
            "unique",
        ),
        (
            lambda: fixed_anchor_scores(
                _transitive_matrix(),
                candidates=("A",),
                anchors=("A",),
            ),
            "disjoint",
        ),
        (
            lambda: fixed_anchor_scores(
                _transitive_matrix(),
                candidates=("missing",),
                anchors=("C",),
            ),
            "agent",
        ),
        (
            lambda: ExploitabilityProxy(
                checkpoint="x",
                best_response="x",
                training_steps=1,
                score=score_interval(wins=1, draws=0, losses=0),
            ),
            "distinct",
        ),
        (
            lambda: AnchorScore(
                candidate="A",
                anchor="A",
                wdl=WDL(1, 0, 0),
            ),
            "distinct",
        ),
        (
            lambda: AnchorScore(
                candidate="A",
                anchor="B",
                wdl=cast(WDL, "wdl"),
            ),
            "WDL",
        ),
        (
            lambda: AnchorScore(
                candidate="A",
                anchor="B",
                wdl=WDL(0, 0, 0),
            ),
            "game",
        ),
        (
            lambda: ExploitabilityProxy(
                checkpoint="checkpoint",
                best_response="response",
                training_steps=1,
                score=cast(object, "score"),  # type: ignore[arg-type]
            ),
            "MatchScore",
        ),
        (
            lambda: exploitability_proxy(
                cast(object, "match"),  # type: ignore[arg-type]
                training_steps=1,
            ),
            "MatchResult",
        ),
    ],
)
def test_population_functions_reject_invalid_or_incomplete_inputs(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_public_population_records_reject_corruption() -> None:
    report = league_elo(_transitive_matrix())
    cycle_report = three_cycles(_cycle_matrix())
    proxy = ExploitabilityProxy(
        checkpoint="checkpoint",
        best_response="response",
        training_steps=1,
        score=score_interval(wins=1, draws=0, losses=0),
    )

    with pytest.raises(ValueError, match="ordered"):
        replace(report.ratings[0], low=report.ratings[0].high + 1.0)
    with pytest.raises(TypeError, match="rating"):
        replace(report.ratings[0], rating=cast(float, "rating"))
    with pytest.raises(ValueError, match="finite"):
        replace(report.ratings[0], rating=float("nan"))
    with pytest.raises(ValueError, match="zero"):
        replace(report, ratings=(report.ratings[0],))
    with pytest.raises(TypeError, match="ratings"):
        replace(report, ratings=cast(tuple[EloRating, ...], list(report.ratings)))
    with pytest.raises(ValueError, match="unique"):
        replace(
            report,
            ratings=(
                report.ratings[0],
                replace(report.ratings[1], agent=report.ratings[0].agent),
                report.ratings[2],
            ),
        )
    with pytest.raises(TypeError, match="approximately_transitive"):
        replace(report, approximately_transitive=cast(bool, 1))
    with pytest.raises(ValueError, match="ci_assumption"):
        replace(report, ci_assumption="unconditional")
    with pytest.raises(TypeError, match="max_abs_residual"):
        replace(report, max_abs_residual=cast(float, "zero"))
    with pytest.raises(ValueError, match="weighted_rmse"):
        replace(report, weighted_rmse=float("nan"))
    with pytest.raises(ValueError, match="count"):
        ThreeCycleReport(
            cycles=cycle_report.cycles, count=0, total_magnitude=0.25, max_magnitude=0.25
        )
    with pytest.raises(TypeError, match="cycles"):
        replace(cycle_report, cycles=cast(tuple[ThreeCycle, ...], []))
    with pytest.raises(ValueError, match="total_magnitude"):
        replace(cycle_report, total_magnitude=0.0)
    with pytest.raises(ValueError, match="max_magnitude"):
        replace(cycle_report, max_magnitude=0.0)
    with pytest.raises(ValueError, match="magnitude"):
        replace(cycle_report.cycles[0], magnitude=0.1)
    with pytest.raises(TypeError, match="agents"):
        replace(
            cycle_report.cycles[0],
            agents=cast(tuple[str, str, str], ["A", "B", "C"]),
        )
    with pytest.raises(ValueError, match="distinct"):
        replace(cycle_report.cycles[0], agents=("A", "A", "C"))
    with pytest.raises(TypeError, match="edge_margins"):
        replace(
            cycle_report.cycles[0],
            edge_margins=cast(tuple[float, float, float], [0.25, 0.25, 0.25]),
        )
    with pytest.raises(ValueError, match="half"):
        replace(
            cycle_report.cycles[0],
            edge_margins=(0.6, 0.25, 0.25),
            magnitude=0.25,
        )
    with pytest.raises(ValueError, match="steps"):
        replace(proxy, training_steps=0)
    with pytest.raises(TypeError, match="MatchScore"):
        replace(proxy, score=cast(object, "score"))  # type: ignore[arg-type]


def test_endpoint_sweeps_use_finite_continuity_corrected_league_ratings() -> None:
    sweep = PayoffMatrix(
        agents=("A", "B"),
        pairs=(PairResult("A", "B", WDL(4, 0, 0)),),
    )

    report = league_elo(sweep)

    assert all(abs(rating.rating) < float("inf") for rating in report.ratings)
    assert report.ratings[0].rating > report.ratings[1].rating


def test_league_rating_lookup_rejects_unknown_agent() -> None:
    report = league_elo(_transitive_matrix())

    assert report.rating_for("A") == report.ratings[0]
    with pytest.raises(ValueError, match="unknown"):
        report.rating_for("missing")


def test_corrupted_matrix_missing_pair_is_detected_on_access() -> None:
    matrix = _transitive_matrix()
    object.__setattr__(matrix, "pairs", matrix.pairs[:-1])

    with pytest.raises(RuntimeError, match="missing"):
        matrix.wdl("A", "C")


def test_league_report_confidence_intervals_are_finite_and_ordered() -> None:
    report: LeagueEloReport = league_elo(_transitive_matrix())

    assert all(
        isclose(rating.rating, rating.rating) and rating.low < rating.rating < rating.high
        for rating in report.ratings
    )
