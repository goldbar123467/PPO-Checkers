"""Immutable, validated configuration for Checkers PPO runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Self, cast

import yaml

UINT64_MAX = (1 << 64) - 1
PHASE = 7
STAGES = frozenset({"A", "B", "C", "practice"})
ARMS = frozenset({"A0", "A1", "A2", "A3"})
DEVICES = frozenset({"cpu", "cuda"})
AMP_DTYPES = frozenset({"float32", "bfloat16"})
WANDB_MODES = frozenset({"offline", "disabled"})
MIN_POOL_CAPACITY = 2


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    checked = value.strip()
    if not checked:
        raise ValueError(f"{name} must not be empty")
    return checked


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    checked = _integer(value, name)
    if checked < 1:
        raise ValueError(f"{name} must be positive")
    return checked


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{name} must be finite")
    return checked


def _positive(value: object, name: str) -> float:
    checked = _number(value, name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _nonnegative(value: object, name: str) -> float:
    checked = _number(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _unit_interval(value: object, name: str, *, open_zero: bool = False) -> float:
    checked = _number(value, name)
    lower_ok = checked > 0.0 if open_zero else checked >= 0.0
    if not lower_ok or checked > 1.0:
        boundary = "(0, 1]" if open_zero else "[0, 1]"
        raise ValueError(f"{name} must be in {boundary}")
    return checked


def _choice(value: object, name: str, choices: frozenset[str]) -> str:
    checked = _string(value, name)
    if checked not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {allowed}")
    return checked


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Complete immutable configuration for one self-play training run."""

    experiment_id: str = "checkers-phase7-smoke-v1"
    seed: int = 0
    phase: int = PHASE
    stage: str = "A"
    arm: str = "A0"
    device: str = "cuda"
    deterministic: bool = True
    total_updates: int = 6_144
    schedule_horizon_updates: int = 6_144
    duration_seconds: float | None = 1_800.0
    num_envs: int = 64
    num_steps: int = 128
    num_minibatches: int = 8
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.001
    ent_anneal_fraction: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    adam_eps: float = 1e-5
    max_plies: int = 512
    repetition_draws: bool = True
    snapshot_every: int = 20
    pool_capacity: int = 20
    periodic_every: int = 10
    checkpoint_every: int = 10
    eval_games: int = 364
    periodic_games: int = 2
    exploitability_train_games: int = 16
    amp_dtype: str = "float32"
    include_opponent_value_loss: bool = False
    wandb_mode: str = "offline"

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        """Return transitions collected per rollout."""

        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Return exact transitions per PPO minibatch."""

        return self.batch_size // self.num_minibatches

    @property
    def total_timesteps(self) -> int:
        """Return the exact transition budget implied by updates and rollout size."""

        return self.total_updates * self.batch_size

    @property
    def schedule_horizon_timesteps(self) -> int:
        """Return the independent LR/entropy schedule horizon in transitions."""

        return self.schedule_horizon_updates * self.batch_size

    def validate(self) -> Self:  # noqa: PLR0912, PLR0915
        """Validate every field and return this unchanged immutable object.

        Returns:
            This configuration after successful validation.

        Raises:
            TypeError: If a field has an invalid runtime type.
            ValueError: If a field or cross-field relation is outside the contract.
        """

        _string(self.experiment_id, "experiment_id")
        seed = _integer(self.seed, "seed")
        if not 0 <= seed <= UINT64_MAX:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if _integer(self.phase, "phase") != PHASE:
            raise ValueError(f"phase must be {PHASE}")
        _choice(self.stage, "stage", STAGES)
        _choice(self.arm, "arm", ARMS)
        _choice(self.device, "device", DEVICES)
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be bool")
        _positive_integer(self.total_updates, "total_updates")
        _positive_integer(self.schedule_horizon_updates, "schedule_horizon_updates")
        if self.duration_seconds is not None:
            _positive(self.duration_seconds, "duration_seconds")
        if self.stage == "practice" and self.duration_seconds is not None:
            raise ValueError("practice runs must terminate on total_updates, not duration_seconds")
        _positive_integer(self.num_envs, "num_envs")
        _positive_integer(self.num_steps, "num_steps")
        minibatches = _positive_integer(self.num_minibatches, "num_minibatches")
        _positive_integer(self.update_epochs, "update_epochs")
        if self.batch_size % minibatches:
            raise ValueError("batch_size must be divisible by num_minibatches")
        _positive(self.learning_rate, "learning_rate")
        _unit_interval(self.gamma, "gamma")
        _unit_interval(self.gae_lambda, "gae_lambda")
        _positive(self.clip_coef, "clip_coef")
        _nonnegative(self.vf_coef, "vf_coef")
        start_entropy = _nonnegative(self.ent_coef_start, "ent_coef_start")
        end_entropy = _nonnegative(self.ent_coef_end, "ent_coef_end")
        if end_entropy > start_entropy:
            raise ValueError("ent_coef_end must not exceed ent_coef_start")
        _unit_interval(self.ent_anneal_fraction, "ent_anneal_fraction", open_zero=True)
        _positive(self.max_grad_norm, "max_grad_norm")
        _positive(self.target_kl, "target_kl")
        _positive(self.adam_eps, "adam_eps")
        _positive_integer(self.max_plies, "max_plies")
        if not isinstance(self.repetition_draws, bool):
            raise TypeError("repetition_draws must be bool")
        _positive_integer(self.snapshot_every, "snapshot_every")
        if _positive_integer(self.pool_capacity, "pool_capacity") < MIN_POOL_CAPACITY:
            raise ValueError(f"pool_capacity must be at least {MIN_POOL_CAPACITY}")
        _positive_integer(self.periodic_every, "periodic_every")
        _positive_integer(self.checkpoint_every, "checkpoint_every")
        eval_games = _positive_integer(self.eval_games, "eval_games")
        if eval_games % 2:
            raise ValueError("eval_games must be even for colour balance")
        periodic_games = _positive_integer(self.periodic_games, "periodic_games")
        if periodic_games % 2:
            raise ValueError("periodic_games must be even for colour balance")
        exploitability_train_games = _positive_integer(
            self.exploitability_train_games, "exploitability_train_games"
        )
        if exploitability_train_games % 2:
            raise ValueError("exploitability_train_games must be even for colour balance")
        _choice(self.amp_dtype, "amp_dtype", AMP_DTYPES)
        if not isinstance(self.include_opponent_value_loss, bool):
            raise TypeError("include_opponent_value_loss must be bool")
        _choice(self.wandb_mode, "wandb_mode", WANDB_MODES)
        return self


def load_run_config(text: str) -> RunConfig:
    """Parse one exact YAML mapping into a validated configuration.

    Args:
        text: YAML text containing every `RunConfig` field exactly once.

    Returns:
        Validated immutable run configuration.

    Raises:
        TypeError: If text or its YAML root has an invalid type.
        ValueError: If syntax, fields, or values are invalid.
    """

    if not isinstance(text, str):
        raise TypeError("configuration text must be a string")
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError("invalid run configuration YAML") from error
    if not isinstance(loaded, dict):
        raise TypeError("run configuration root must be a mapping")
    values = cast(dict[object, object], loaded)
    expected = {field.name for field in fields(RunConfig)}
    actual = set(values)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"unknown run configuration fields: {sorted(map(str, unknown))}")
    if missing:
        raise ValueError(f"missing run configuration fields: {sorted(missing)}")
    return RunConfig(**cast(dict[str, object], values))  # type: ignore[arg-type]
