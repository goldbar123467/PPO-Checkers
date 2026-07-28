"""Hand-derived Phase 7 metric formulas, names, and training alerts."""

from __future__ import annotations

import math

import pytest
import torch

from checkers.metrics import (
    REQUIRED_METRIC_KEYS,
    GameStatistics,
    GameSummary,
    MaskStatistics,
    TrainingAlertMonitor,
    TrainingHaltError,
    calibration_mae,
    explained_variance,
    policy_health,
    value_health,
)
from checkers.rules.state import PlayerId
from checkers.rules.terminal import TerminationReason

EXPECTED_METRIC_COUNT = 55


def test_m1_policy_health_uses_mean_of_per_state_normalized_entropy() -> None:
    logits = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 0.0, 0.0],
            [7.0, 9.0, 11.0, 13.0],
        ],
        dtype=torch.float64,
    )
    masks = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
            [False, False, True, False],
        ]
    )

    metrics = policy_health(logits=logits, legal_mask=masks)

    assert metrics["policy/normalized_entropy"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["policy/max_prob_mean"] == pytest.approx((0.5 + 0.25 + 1.0) / 3.0)
    assert metrics["policy/frac_states_k_eq_1"] == pytest.approx(1.0 / 3.0)
    ratio_of_means = (math.log(2.0) + math.log(4.0)) / 2.0 / math.log(3.0)
    assert metrics["policy/normalized_entropy"] != pytest.approx(ratio_of_means)


def test_m1_value_health_matches_literal_population_variance_and_weighted_bins() -> None:
    predictions = torch.tensor([-0.8, -0.6, 0.2, 0.4], dtype=torch.float64)
    targets = torch.tensor([-1.0, -0.5, 0.0, 1.0], dtype=torch.float64)
    realized = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)

    metrics = value_health(
        values=predictions,
        targets=targets,
        calibration_predictions=predictions,
        realized_outcomes=realized,
        calibration_bins=2,
    )

    expected_variance = 1.0 - torch.var(targets - predictions, unbiased=False) / torch.var(
        targets, unbiased=False
    )
    assert metrics["value/mean"] == pytest.approx(-0.2)
    assert metrics["value/std"] == pytest.approx(float(torch.std(predictions, unbiased=False)))
    assert metrics["value/target_mean"] == pytest.approx(-0.125)
    assert metrics["value/explained_variance"] == pytest.approx(float(expected_variance))
    assert metrics["value/calibration_mae"] == pytest.approx(0.5)


def test_m1_zero_variance_and_empty_calibration_have_declared_zero_diagnostics() -> None:
    assert explained_variance(torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])) == 0.0
    assert (
        calibration_mae(
            predictions=torch.tensor([], dtype=torch.float32),
            outcomes=torch.tensor([], dtype=torch.float32),
            bins=10,
        )
        == 0.0
    )


def test_m1_mask_statistics_aggregate_literal_counts() -> None:
    stats = MaskStatistics()
    stats.record(
        legal_counts=torch.tensor([1, 3, 2]),
        continuation_states=torch.tensor([True, False, True]),
        sample_legality_violations=0,
        oracle_disagreements=0,
        empty_mask_count=0,
    )
    stats.record(
        legal_counts=torch.tensor([4]),
        continuation_states=torch.tensor([False]),
        sample_legality_violations=1,
        oracle_disagreements=2,
        empty_mask_count=3,
    )

    assert stats.metrics() == {
        "mask/sample_legality_violations": 1.0,
        "mask/oracle_disagreements": 2.0,
        "mask/empty_mask_count": 3.0,
        "mask/mean_legal_actions": 2.5,
        "mask/continuation_state_frac": 0.5,
    }


def test_m1_game_statistics_use_completed_games_and_explicit_started_roles() -> None:
    stats = GameStatistics()
    stats.start_game(current_policy_as_red=True)
    stats.start_game(current_policy_as_red=False)
    stats.record_game(
        GameSummary(
            winner=PlayerId.RED,
            reason=TerminationReason.NO_PIECES,
            steps=5,
            moves=3,
            captures=2,
            capture_sequences=(2,),
            promotions=1,
        )
    )
    stats.record_game(
        GameSummary(
            winner=None,
            reason=TerminationReason.PLY_CAP,
            steps=7,
            moves=7,
            captures=0,
            capture_sequences=(),
            promotions=0,
        )
    )

    assert stats.metrics() == {
        "env/mean_game_len_moves": 5.0,
        "env/mean_game_len_steps": 6.0,
        "env/draw_rate": 0.5,
        "env/first_player_win_rate": 0.5,
        "env/captures_per_game": 1.0,
        "env/mean_sequence_len": 2.0,
        "env/promotion_rate": 0.1,
        "env/no_progress_draws": 0.0,
        "env/ply_cap_draws": 1.0,
        "env/first_player_frac": 0.5,
    }


def test_m1_required_metric_inventory_is_exact_and_namespaced() -> None:
    assert len(REQUIRED_METRIC_KEYS) == EXPECTED_METRIC_COUNT
    assert len(set(REQUIRED_METRIC_KEYS)) == len(REQUIRED_METRIC_KEYS)
    assert all("/" in key for key in REQUIRED_METRIC_KEYS)
    assert {
        "train/policy_loss",
        "mask/sample_legality_violations",
        "policy/normalized_entropy",
        "value/calibration_mae",
        "env/draw_rate",
        "eval/payoff_matrix",
    } <= REQUIRED_METRIC_KEYS


def _healthy_metrics() -> dict[str, float]:
    return {
        "train/policy_loss": 0.1,
        "train/value_loss": 0.2,
        "train/entropy": 1.0,
        "train/approx_kl": 0.01,
        "train/clipfrac": 0.0,
        "train/explained_variance": 0.1,
        "train/grad_norm": 0.2,
        "mask/sample_legality_violations": 0.0,
        "mask/oracle_disagreements": 0.0,
        "mask/empty_mask_count": 0.0,
        "policy/normalized_entropy": 0.5,
        "policy/frac_states_k_eq_1": 0.0,
        "eval/vs_random": 0.8,
    }


@pytest.mark.parametrize(
    ("key", "value", "progress", "message"),
    [
        ("mask/sample_legality_violations", 1.0, 0.0, "sample_legality"),
        ("mask/oracle_disagreements", 1.0, 0.0, "oracle_disagreements"),
        ("mask/empty_mask_count", 1.0, 0.0, "empty_mask"),
        ("train/policy_loss", float("nan"), 0.0, "non-finite"),
        ("train/approx_kl", 0.200001, 0.0, "approx_kl"),
        ("policy/normalized_entropy", 0.019, 0.249, "normalized_entropy"),
        ("eval/vs_random", 0.599, 0.201, "vs_random"),
    ],
)
def test_m2_immediate_alerts_halt_on_exact_bad_side(
    key: str,
    value: float,
    progress: float,
    message: str,
) -> None:
    monitor = TrainingAlertMonitor(target_kl=0.02)
    metrics = _healthy_metrics()
    metrics[key] = value

    with pytest.raises(TrainingHaltError, match=message):
        monitor.check(metrics=metrics, progress=progress)


def test_m2_alert_threshold_boundaries_do_not_halt() -> None:
    monitor = TrainingAlertMonitor(target_kl=0.02)
    metrics = _healthy_metrics()
    metrics["train/approx_kl"] = 0.2
    metrics["policy/normalized_entropy"] = 0.02
    metrics["eval/vs_random"] = 0.6

    monitor.check(metrics=metrics, progress=0.25)

    assert monitor.negative_explained_variance_streak == 0


def test_m2_evaluation_alert_does_not_double_count_value_streak() -> None:
    monitor = TrainingAlertMonitor(target_kl=0.02)
    metrics = _healthy_metrics()
    metrics["train/explained_variance"] = -0.1
    monitor.check(metrics=metrics, progress=0.21)

    monitor.check_evaluation(metrics={"eval/vs_random": 0.6}, progress=0.21)

    assert monitor.negative_explained_variance_streak == 1
    with pytest.raises(TrainingHaltError, match="vs_random"):
        monitor.check_evaluation(metrics={"eval/vs_random": 0.599}, progress=0.21)


def test_m2_entropy_alert_ignores_batches_with_no_choice_states() -> None:
    monitor = TrainingAlertMonitor(target_kl=0.02)
    metrics = _healthy_metrics()
    metrics["policy/normalized_entropy"] = 0.0
    metrics["policy/frac_states_k_eq_1"] = 1.0

    monitor.check(metrics=metrics, progress=0.1)


def test_m2_negative_explained_variance_halts_on_twentieth_consecutive_update() -> None:
    monitor = TrainingAlertMonitor(target_kl=0.02)
    metrics = _healthy_metrics()
    metrics["train/explained_variance"] = -0.01
    for _ in range(19):
        monitor.check(metrics=metrics, progress=0.5)
    with pytest.raises(TrainingHaltError, match="explained_variance"):
        monitor.check(metrics=metrics, progress=0.5)

    metrics["train/explained_variance"] = 0.0
    monitor.check(metrics=metrics, progress=0.5)
    assert monitor.negative_explained_variance_streak == 0
