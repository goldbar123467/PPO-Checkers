"""Literal Phase 7 training-health metrics and fail-closed alert thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rules.state import PlayerId
from checkers.rules.terminal import DRAW_REASONS, TerminationReason

POLICY_BATCH_RANK = 2
NEGATIVE_EXPLAINED_VARIANCE_LIMIT = 20
EARLY_TRAINING_FRACTION = 0.25
ENTROPY_COLLAPSE_THRESHOLD = 0.02
RANDOM_ALERT_START_FRACTION = 0.20
RANDOM_SCORE_FLOOR = 0.6

OPTIMIZATION_METRIC_KEYS = frozenset(
    {
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/approx_kl",
        "train/clipfrac",
        "train/explained_variance",
        "train/grad_norm",
        "train/lr",
        "train/ent_coef",
        "train/kl_early_stops",
        "train/trainable_frac",
        "charts/SPS",
    }
)
MASK_METRIC_KEYS = frozenset(
    {
        "mask/sample_legality_violations",
        "mask/oracle_disagreements",
        "mask/empty_mask_count",
        "mask/mean_legal_actions",
        "mask/continuation_state_frac",
    }
)
POLICY_METRIC_KEYS = frozenset(
    {
        "policy/normalized_entropy",
        "policy/max_prob_mean",
        "policy/frac_states_k_eq_1",
    }
)
VALUE_METRIC_KEYS = frozenset(
    {
        "value/mean",
        "value/std",
        "value/target_mean",
        "value/explained_variance",
        "value/calibration_mae",
    }
)
GAME_METRIC_KEYS = frozenset(
    {
        "env/mean_game_len_moves",
        "env/mean_game_len_steps",
        "env/draw_rate",
        "env/first_player_win_rate",
        "env/captures_per_game",
        "env/mean_sequence_len",
        "env/promotion_rate",
        "env/no_progress_draws",
        "env/ply_cap_draws",
        "env/first_player_frac",
    }
)
_ANCHOR_METRIC_KEYS = frozenset(
    f"eval/vs_{anchor}{suffix}"
    for anchor in ("random", "greedy", "minimax2")
    for suffix in ("", "_ci_low", "_ci_high", "_games")
)
EVALUATION_METRIC_KEYS = _ANCHOR_METRIC_KEYS | frozenset(
    {
        "eval/league_elo",
        "eval/league_elo_ci_low",
        "eval/league_elo_ci_high",
        "eval/payoff_matrix",
        "eval/three_cycle_count",
        "eval/exploitability_proxy",
        "eval/dev_tactical_acc",
        "eval/greedy_vs_sampled_delta",
    }
)
REQUIRED_METRIC_KEYS = (
    OPTIMIZATION_METRIC_KEYS
    | MASK_METRIC_KEYS
    | POLICY_METRIC_KEYS
    | VALUE_METRIC_KEYS
    | GAME_METRIC_KEYS
    | EVALUATION_METRIC_KEYS
)


def _one_dimensional_floating(tensor: object, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return ``1 - Var(target-prediction)/Var(target)`` using population variance.

    Constant or empty targets have no explainable variance and are declared to produce zero.
    """

    checked_predictions = _one_dimensional_floating(predictions, "predictions")
    checked_targets = _one_dimensional_floating(targets, "targets")
    if checked_predictions.shape != checked_targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if checked_targets.numel() == 0:
        return 0.0
    target_variance = torch.var(checked_targets, unbiased=False)
    if float(target_variance.item()) == 0.0:
        return 0.0
    residual_variance = torch.var(checked_targets - checked_predictions, unbiased=False)
    return float((1.0 - residual_variance / target_variance).item())


def calibration_mae(
    *,
    predictions: torch.Tensor,
    outcomes: torch.Tensor,
    bins: int = 10,
) -> float:
    """Return sample-weighted binned absolute calibration error on ``[-1, 1]``."""

    checked_predictions = _one_dimensional_floating(predictions, "predictions")
    checked_outcomes = _one_dimensional_floating(outcomes, "outcomes")
    if checked_predictions.shape != checked_outcomes.shape:
        raise ValueError("predictions and outcomes must have the same shape")
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not bool(((checked_predictions >= -1.0) & (checked_predictions <= 1.0)).all().item()):
        raise ValueError("predictions must be in [-1, 1]")
    if not bool(((checked_outcomes >= -1.0) & (checked_outcomes <= 1.0)).all().item()):
        raise ValueError("outcomes must be in [-1, 1]")
    if checked_predictions.numel() == 0:
        return 0.0

    indices = torch.floor((checked_predictions + 1.0) * (bins / 2.0)).to(torch.int64)
    indices = indices.clamp(max=bins - 1)
    absolute_error = checked_predictions.new_zeros(())
    for bin_index in range(bins):
        members = indices == bin_index
        member_count = int(members.sum().item())
        if member_count:
            gap = torch.abs(checked_predictions[members].mean() - checked_outcomes[members].mean())
            absolute_error = absolute_error + gap * member_count
    return float((absolute_error / checked_predictions.numel()).item())


def policy_health(*, logits: torch.Tensor, legal_mask: torch.Tensor) -> dict[str, float]:
    """Compute policy diagnostics, averaging normalized entropy per eligible state."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a Tensor")
    if logits.ndim != POLICY_BATCH_RANK or logits.shape[0] < 1:
        raise ValueError("logits must be a non-empty two-dimensional batch")
    distribution = MaskedCategorical(logits=logits, legal_mask=legal_mask)
    legal_counts = legal_mask.sum(dim=-1)
    multiple = legal_counts > 1
    if bool(multiple.any().item()):
        normalized = distribution.entropy()[multiple] / torch.log(
            legal_counts[multiple].to(dtype=logits.dtype)
        )
        normalized_entropy = float(normalized.mean().item())
    else:
        normalized_entropy = 0.0
    return {
        "policy/normalized_entropy": normalized_entropy,
        "policy/max_prob_mean": float(distribution.probs.max(dim=-1).values.mean().item()),
        "policy/frac_states_k_eq_1": float((legal_counts == 1).to(logits.dtype).mean().item()),
    }


def value_health(
    *,
    values: torch.Tensor,
    targets: torch.Tensor,
    calibration_predictions: torch.Tensor,
    realized_outcomes: torch.Tensor,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Compute population value moments, explained variance, and calibration error."""

    checked_values = _one_dimensional_floating(values, "values")
    checked_targets = _one_dimensional_floating(targets, "targets")
    if checked_values.shape != checked_targets.shape:
        raise ValueError("values and targets must have the same shape")
    if checked_values.numel() == 0:
        raise ValueError("values and targets must not be empty")
    return {
        "value/mean": float(checked_values.mean().item()),
        "value/std": float(torch.std(checked_values, unbiased=False).item()),
        "value/target_mean": float(checked_targets.mean().item()),
        "value/explained_variance": explained_variance(checked_values, checked_targets),
        "value/calibration_mae": calibration_mae(
            predictions=calibration_predictions,
            outcomes=realized_outcomes,
            bins=calibration_bins,
        ),
    }


@dataclass(slots=True)
class MaskStatistics:
    """Cumulative masking counters and state-level denominators."""

    sample_legality_violations: int = 0
    oracle_disagreements: int = 0
    empty_mask_count: int = 0
    legal_action_total: int = 0
    continuation_state_total: int = 0
    state_total: int = 0

    def record(
        self,
        *,
        legal_counts: torch.Tensor,
        continuation_states: torch.Tensor,
        sample_legality_violations: int,
        oracle_disagreements: int,
        empty_mask_count: int,
    ) -> None:
        """Accumulate one batch of independent mask and oracle checks."""

        if not isinstance(legal_counts, torch.Tensor):
            raise TypeError("legal_counts must be a Tensor")
        if legal_counts.ndim != 1:
            raise ValueError("legal_counts must be one-dimensional")
        if legal_counts.dtype is torch.bool or legal_counts.is_floating_point():
            raise TypeError("legal_counts must have an integer dtype")
        if not isinstance(continuation_states, torch.Tensor):
            raise TypeError("continuation_states must be a Tensor")
        if continuation_states.dtype is not torch.bool:
            raise TypeError("continuation_states must have dtype bool")
        if continuation_states.shape != legal_counts.shape:
            raise ValueError("continuation_states shape must match legal_counts")
        if bool((legal_counts < 0).any().item()):
            raise ValueError("legal_counts must be non-negative")
        checked_counts: list[int] = []
        for value, name in (
            (sample_legality_violations, "sample_legality_violations"),
            (oracle_disagreements, "oracle_disagreements"),
            (empty_mask_count, "empty_mask_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            checked_counts.append(value)
        self.sample_legality_violations += checked_counts[0]
        self.oracle_disagreements += checked_counts[1]
        self.empty_mask_count += checked_counts[2]
        self.legal_action_total += int(legal_counts.sum().item())
        self.continuation_state_total += int(continuation_states.sum().item())
        self.state_total += legal_counts.numel()

    def metrics(self) -> dict[str, float]:
        """Return the five frozen masking metric names with defined zero denominators."""

        denominator = self.state_total
        return {
            "mask/sample_legality_violations": float(self.sample_legality_violations),
            "mask/oracle_disagreements": float(self.oracle_disagreements),
            "mask/empty_mask_count": float(self.empty_mask_count),
            "mask/mean_legal_actions": (
                self.legal_action_total / denominator if denominator else 0.0
            ),
            "mask/continuation_state_frac": (
                self.continuation_state_total / denominator if denominator else 0.0
            ),
        }


@dataclass(frozen=True, slots=True)
class GameSummary:
    """Sufficient statistics for exactly one completed game."""

    winner: PlayerId | None
    reason: TerminationReason
    steps: int
    moves: int
    captures: int
    capture_sequences: tuple[int, ...]
    promotions: int

    def __post_init__(self) -> None:
        if self.winner is not None and not isinstance(self.winner, PlayerId):
            raise TypeError("winner must be a PlayerId or None")
        if not isinstance(self.reason, TerminationReason):
            raise TypeError("reason must be a TerminationReason")
        if (self.winner is None) != (self.reason in DRAW_REASONS):
            raise ValueError("winner and reason disagree about draw status")
        for value, name in (
            (self.steps, "steps"),
            (self.moves, "moves"),
            (self.captures, "captures"),
            (self.promotions, "promotions"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.capture_sequences, tuple):
            raise TypeError("capture_sequences must be a tuple")
        for sequence_length in self.capture_sequences:
            if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
                raise TypeError("capture sequence lengths must be integers")
            if sequence_length < 1:
                raise ValueError("capture sequence lengths must be positive")


@dataclass(slots=True)
class GameStatistics:
    """Cumulative game, move, role, capture, promotion, and draw statistics."""

    games_started: int = 0
    current_policy_first_starts: int = 0
    games_completed: int = 0
    steps: int = 0
    moves: int = 0
    draws: int = 0
    first_player_wins: int = 0
    captures: int = 0
    capture_sequences: list[int] = field(default_factory=list)
    promotions: int = 0
    no_progress_draws: int = 0
    ply_cap_draws: int = 0

    def start_game(self, *, current_policy_as_red: bool) -> None:
        """Record one game start and the current policy's explicit colour role."""

        if not isinstance(current_policy_as_red, bool):
            raise TypeError("current_policy_as_red must be bool")
        self.games_started += 1
        self.current_policy_first_starts += int(current_policy_as_red)

    def record_game(self, summary: GameSummary) -> None:
        """Accumulate one validated terminal summary."""

        if not isinstance(summary, GameSummary):
            raise TypeError("summary must be a GameSummary")
        self.games_completed += 1
        self.steps += summary.steps
        self.moves += summary.moves
        self.draws += int(summary.winner is None)
        self.first_player_wins += int(summary.winner is PlayerId.RED)
        self.captures += summary.captures
        self.capture_sequences.extend(summary.capture_sequences)
        self.promotions += summary.promotions
        self.no_progress_draws += int(summary.reason is TerminationReason.NO_PROGRESS)
        self.ply_cap_draws += int(summary.reason is TerminationReason.PLY_CAP)

    def metrics(self) -> dict[str, float]:
        """Return the ten frozen game metric names with explicit denominators."""

        games = self.games_completed
        moves = self.moves
        return {
            "env/mean_game_len_moves": self.moves / games if games else 0.0,
            "env/mean_game_len_steps": self.steps / games if games else 0.0,
            "env/draw_rate": self.draws / games if games else 0.0,
            "env/first_player_win_rate": self.first_player_wins / games if games else 0.0,
            "env/captures_per_game": self.captures / games if games else 0.0,
            "env/mean_sequence_len": (
                sum(self.capture_sequences) / len(self.capture_sequences)
                if self.capture_sequences
                else 0.0
            ),
            "env/promotion_rate": self.promotions / moves if moves else 0.0,
            "env/no_progress_draws": float(self.no_progress_draws),
            "env/ply_cap_draws": float(self.ply_cap_draws),
            "env/first_player_frac": (
                self.current_policy_first_starts / self.games_started if self.games_started else 0.0
            ),
        }


class TrainingHaltError(RuntimeError):
    """Raised when a frozen Phase 7 training-health alert requires diagnosis."""


class TrainingAlertMonitor:
    """Stateful implementation of every hard alert in GOAL §13.3."""

    def __init__(self, *, target_kl: float) -> None:
        if isinstance(target_kl, bool) or not isinstance(target_kl, (int, float)):
            raise TypeError("target_kl must be numeric")
        self.target_kl = float(target_kl)
        if not math.isfinite(self.target_kl) or self.target_kl <= 0.0:
            raise ValueError("target_kl must be finite and positive")
        self.negative_explained_variance_streak = 0

    @staticmethod
    def _metric(metrics: Mapping[str, float], key: str) -> float:
        if key not in metrics:
            raise KeyError(f"required alert metric is missing: {key}")
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"alert metric {key} must be numeric")
        return float(value)

    def check(self, *, metrics: Mapping[str, float], progress: float) -> None:
        """Raise immediately or at the exact sustained boundary for unhealthy metrics."""

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if isinstance(progress, bool) or not isinstance(progress, (int, float)):
            raise TypeError("progress must be numeric")
        checked_progress = float(progress)
        if not math.isfinite(checked_progress) or not 0.0 <= checked_progress <= 1.0:
            raise ValueError("progress must be finite and in [0, 1]")

        for key in ("train/policy_loss", "train/value_loss"):
            if not math.isfinite(self._metric(metrics, key)):
                raise TrainingHaltError(f"non-finite loss: {key}")
        for key in (
            "mask/sample_legality_violations",
            "mask/oracle_disagreements",
            "mask/empty_mask_count",
        ):
            if self._metric(metrics, key) > 0.0:
                raise TrainingHaltError(f"mask alert: {key}")
        if self._metric(metrics, "train/approx_kl") > 10.0 * self.target_kl:
            raise TrainingHaltError("train/approx_kl exceeded 10 * target_kl")

        explained = self._metric(metrics, "train/explained_variance")
        self.negative_explained_variance_streak = (
            self.negative_explained_variance_streak + 1 if explained < 0.0 else 0
        )
        if self.negative_explained_variance_streak >= NEGATIVE_EXPLAINED_VARIANCE_LIMIT:
            raise TrainingHaltError("train/explained_variance was negative for 20 updates")
        if (
            checked_progress < EARLY_TRAINING_FRACTION
            and self._metric(metrics, "policy/frac_states_k_eq_1") < 1.0
            and self._metric(metrics, "policy/normalized_entropy") < ENTROPY_COLLAPSE_THRESHOLD
        ):
            raise TrainingHaltError("policy/normalized_entropy collapsed before 25% of training")
        if (
            checked_progress > RANDOM_ALERT_START_FRACTION
            and "eval/vs_random" in metrics
            and self._metric(metrics, "eval/vs_random") < RANDOM_SCORE_FLOOR
        ):
            raise TrainingHaltError("eval/vs_random fell below 0.6 after 20% of training")

    def check_evaluation(self, *, metrics: Mapping[str, float], progress: float) -> None:
        """Apply evaluation-only alerts without double-counting update-health streaks."""

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if isinstance(progress, bool) or not isinstance(progress, (int, float)):
            raise TypeError("progress must be numeric")
        checked_progress = float(progress)
        if not math.isfinite(checked_progress) or not 0.0 <= checked_progress <= 1.0:
            raise ValueError("progress must be finite and in [0, 1]")
        if (
            checked_progress > RANDOM_ALERT_START_FRACTION
            and self._metric(metrics, "eval/vs_random") < RANDOM_SCORE_FLOOR
        ):
            raise TrainingHaltError("eval/vs_random fell below 0.6 after 20% of training")
