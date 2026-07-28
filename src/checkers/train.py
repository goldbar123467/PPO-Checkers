"""Exactly-once rollout updates and the Phase 7 training orchestration primitives."""

from __future__ import annotations

import math
import random
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch

from checkers.checkpoint import (
    CheckpointEvidence,
    load_checkpoint,
)
from checkers.checkpoint import (
    save_checkpoint as write_checkpoint,
)
from checkers.config import RunConfig
from checkers.metrics import TrainingAlertMonitor, explained_variance
from checkers.rl.buffer import RolloutBatch
from checkers.rl.determinism import ENV_STREAM_OFFSET, derive_stream_seed, seed_everything
from checkers.rl.league import LeaguePool
from checkers.rl.networks import CheckersNetwork
from checkers.rl.ppo import (
    PPOConfig,
    PPOMinibatch,
    PPOUpdateMetrics,
    build_ppo_minibatch,
    ppo_minibatch_update,
)
from checkers.rl.selfplay import SelfPlayCollector
from checkers.rules.state import State
from checkers.schedules import current_ent_coef, current_lr, schedule_progress
from checkers.trainer_state import TrainerState, capture_rng_states, restore_rng_states


@dataclass(frozen=True, slots=True)
class EpochPass:
    """Truthful source-index ledger for one complete or KL-truncated epoch."""

    source_indices: tuple[int, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class RolloutUpdateResult:
    """Aggregated update metrics plus exact on-policy consumption evidence."""

    metrics: dict[str, float]
    epochs: tuple[EpochPass, ...]
    transitions: int
    optimizer_steps: int
    kl_early_stopped: bool


@dataclass(frozen=True, slots=True)
class TrainingUpdate:
    """One completed collection/update boundary and its replay evidence."""

    global_step: int
    update_idx: int
    actions: tuple[int, ...]
    epochs: tuple[EpochPass, ...]
    metrics: dict[str, float]


def _entropy_coefficient(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ent_coef must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError("ent_coef must be finite and non-negative")
    return checked


def _slice_minibatch(batch: PPOMinibatch, indices: torch.Tensor) -> PPOMinibatch:
    device_indices = indices.to(device=batch.obs.device)
    return PPOMinibatch(
        obs=batch.obs[device_indices],
        legal_mask=batch.legal_mask[device_indices],
        actions=batch.actions[device_indices],
        old_logprob=batch.old_logprob[device_indices],
        old_values=batch.old_values[device_indices],
        advantages=batch.advantages[device_indices],
        returns=batch.returns[device_indices],
        source_indices=batch.source_indices[device_indices],
    )


class RolloutUpdater:
    """Apply PPO epochs once to each rollout using a dedicated permutation stream."""

    def __init__(
        self,
        *,
        config: RunConfig,
        network: CheckersNetwork,
        optimizer: torch.optim.Optimizer,
        minibatch_generator: torch.Generator,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not isinstance(network, CheckersNetwork):
            raise TypeError("network must be a CheckersNetwork")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        if not isinstance(minibatch_generator, torch.Generator):
            raise TypeError("minibatch_generator must be a torch Generator")
        self._config = config
        self._network = network
        self._optimizer = optimizer
        self._minibatch_generator = minibatch_generator
        self._consumed_rollouts: weakref.WeakValueDictionary[int, RolloutBatch] = (
            weakref.WeakValueDictionary()
        )

    def _ppo_config(self, ent_coef: float) -> PPOConfig:
        return PPOConfig(
            clip_coef=self._config.clip_coef,
            vf_coef=self._config.vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=self._config.max_grad_norm,
            normalize_advantages=True,
            advantage_eps=1e-8,
            target_kl=self._config.target_kl,
        )

    def _learning_rate(self) -> float:
        rates = {float(group["lr"]) for group in self._optimizer.param_groups}
        if len(rates) != 1:
            raise ValueError("optimizer parameter groups must share one learning rate")
        return rates.pop()

    @staticmethod
    def _aggregate(
        updates: list[PPOUpdateMetrics],
        *,
        batch: PPOMinibatch,
        rollout: RolloutBatch,
        learning_rate: float,
        ent_coef: float,
    ) -> dict[str, float]:
        total_weight = sum(update.transitions for update in updates)
        if not updates or total_weight < 1:
            raise RuntimeError("PPO update produced no optimizer steps")

        def mean(field: str) -> float:
            return (
                sum(float(getattr(update, field)) * update.transitions for update in updates)
                / total_weight
            )

        return {
            "train/policy_loss": mean("policy_loss"),
            "train/value_loss": mean("value_loss"),
            "train/entropy": mean("entropy"),
            "train/approx_kl": mean("approx_kl"),
            "train/clipfrac": mean("clipfrac"),
            "train/explained_variance": explained_variance(batch.old_values, batch.returns),
            "train/grad_norm": mean("grad_norm"),
            "train/lr": learning_rate,
            "train/ent_coef": ent_coef,
            "train/kl_early_stops": float(sum(update.kl_early_stop for update in updates)),
            "train/trainable_frac": float(batch.actions.numel() / rollout.transitions),
        }

    def update(self, *, rollout: RolloutBatch, ent_coef: float) -> RolloutUpdateResult:
        """Consume one rollout for configured shuffled epochs, stopping early on KL.

        A rollout is marked consumed immediately before the first optimizer step. If an update
        fails midway, reusing the now-partially-off-policy data is rejected.
        """

        if not isinstance(rollout, RolloutBatch):
            raise TypeError("rollout must be a RolloutBatch")
        checked_ent_coef = _entropy_coefficient(ent_coef)
        rollout_identity = id(rollout)
        if self._consumed_rollouts.get(rollout_identity) is rollout:
            raise RuntimeError("rollout was already consumed")
        batch = build_ppo_minibatch(rollout)
        trainable_count = int(batch.actions.shape[0])
        if trainable_count < self._config.num_minibatches:
            raise ValueError("trainable transitions must cover every configured minibatch")
        learning_rate = self._learning_rate()
        ppo_config = self._ppo_config(checked_ent_coef)
        self._consumed_rollouts[rollout_identity] = rollout

        updates: list[PPOUpdateMetrics] = []
        epochs: list[EpochPass] = []
        early_stopped = False
        for _ in range(self._config.update_epochs):
            permutation = torch.randperm(
                trainable_count,
                generator=self._minibatch_generator,
            )
            visited: list[int] = []
            for indices in torch.tensor_split(permutation, self._config.num_minibatches):
                minibatch = _slice_minibatch(batch, indices)
                visited.extend(int(index) for index in minibatch.source_indices.tolist())
                update = ppo_minibatch_update(
                    network=self._network,
                    optimizer=self._optimizer,
                    minibatch=minibatch,
                    config=ppo_config,
                )
                updates.append(update)
                if update.kl_early_stop:
                    early_stopped = True
                    break
            epochs.append(
                EpochPass(
                    source_indices=tuple(visited),
                    complete=len(visited) == trainable_count,
                )
            )
            if early_stopped:
                break

        metrics = self._aggregate(
            updates,
            batch=batch,
            rollout=rollout,
            learning_rate=learning_rate,
            ent_coef=checked_ent_coef,
        )
        return RolloutUpdateResult(
            metrics=metrics,
            epochs=tuple(epochs),
            transitions=trainable_count,
            optimizer_steps=len(updates),
            kl_early_stopped=early_stopped,
        )


class TrainingSession:
    """Own all mutable state required for deterministic Phase 7 A0 training and resume."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        config: RunConfig,
        network: CheckersNetwork,
        optimizer: torch.optim.Optimizer,
        state: TrainerState,
        league: LeaguePool,
        collector: SelfPlayCollector,
        opponent_rng: random.Random,
        minibatch_generator: torch.Generator,
        env_rngs: tuple[random.Random, ...],
        learning_rate: float,
        entropy_coefficient: float,
    ) -> None:
        self.config = config
        self.network = network
        self.optimizer = optimizer
        self.state = state
        self.league = league
        self.collector = collector
        self._opponent_rng = opponent_rng
        self._minibatch_generator = minibatch_generator
        self._env_rngs = env_rngs
        self._last_learning_rate = learning_rate
        self._last_entropy_coefficient = entropy_coefficient
        self._updater = RolloutUpdater(
            config=config,
            network=network,
            optimizer=optimizer,
            minibatch_generator=minibatch_generator,
        )
        self._alerts = TrainingAlertMonitor(target_kl=config.target_kl)
        self._alerts.negative_explained_variance_streak = state.negative_explained_variance_streak

    @staticmethod
    def _owned_rngs(
        config: RunConfig,
    ) -> tuple[random.Random, torch.Generator, tuple[random.Random, ...]]:
        opponent_seed = derive_stream_seed(config.seed, ENV_STREAM_OFFSET + config.num_envs)
        minibatch_seed = derive_stream_seed(config.seed, ENV_STREAM_OFFSET + config.num_envs + 1)
        env_rngs = tuple(
            random.Random(derive_stream_seed(config.seed, ENV_STREAM_OFFSET + lane))
            for lane in range(config.num_envs)
        )
        return (
            random.Random(opponent_seed),
            torch.Generator().manual_seed(minibatch_seed),
            env_rngs,
        )

    @classmethod
    def create(
        cls,
        *,
        config: RunConfig,
        initial_states: tuple[State, ...] | None = None,
    ) -> TrainingSession:
        """Seed and construct a fresh network, optimizer, pool, collector, and trainer state."""

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        seed_everything(
            config.seed,
            num_envs=config.num_envs,
            deterministic=config.deterministic,
        )
        network = CheckersNetwork().to(torch.device(config.device))
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
        )
        collector = SelfPlayCollector(
            config=config,
            network=network,
            initial_states=initial_states,
        )
        league = LeaguePool(capacity=config.pool_capacity)
        league.pin_initial(network.state_dict())
        opponent_rng, minibatch_generator, env_rngs = cls._owned_rngs(config)
        state = TrainerState(
            env_episode_indices=collector.episode_indices,
            league_snapshot_ids=league.snapshot_ids,
        )
        state.rng_states = capture_rng_states(
            opponent_rng=opponent_rng,
            minibatch_generator=minibatch_generator,
            env_rngs=env_rngs,
        )
        return cls(
            config=config,
            network=network,
            optimizer=optimizer,
            state=state,
            league=league,
            collector=collector,
            opponent_rng=opponent_rng,
            minibatch_generator=minibatch_generator,
            env_rngs=env_rngs,
            learning_rate=config.learning_rate,
            entropy_coefficient=config.ent_coef_start,
        )

    @classmethod
    def resume(
        cls,
        *,
        config: RunConfig,
        checkpoint_path: Path,
    ) -> TrainingSession:
        """Load a checkpoint and restore every stochastic stream before the next action."""

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        seed_everything(
            config.seed,
            num_envs=config.num_envs,
            deterministic=config.deterministic,
        )
        network = CheckersNetwork().to(torch.device(config.device))
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
        )
        loaded = load_checkpoint(
            path=checkpoint_path,
            expected_config=config,
            network=network,
            optimizer=optimizer,
        )
        opponent_rng, minibatch_generator, env_rngs = cls._owned_rngs(config)
        if loaded.state.rng_states is None:
            raise RuntimeError("loaded checkpoint omitted RNG states")
        restore_rng_states(
            loaded.state.rng_states,
            opponent_rng=opponent_rng,
            minibatch_generator=minibatch_generator,
            env_rngs=env_rngs,
        )
        return cls(
            config=config,
            network=network,
            optimizer=optimizer,
            state=loaded.state,
            league=loaded.league,
            collector=loaded.collector,
            opponent_rng=opponent_rng,
            minibatch_generator=minibatch_generator,
            env_rngs=env_rngs,
            learning_rate=loaded.learning_rate,
            entropy_coefficient=loaded.entropy_coefficient,
        )

    def _capture_rngs(self) -> None:
        self.state.rng_states = capture_rng_states(
            opponent_rng=self._opponent_rng,
            minibatch_generator=self._minibatch_generator,
            env_rngs=self._env_rngs,
        )

    def run_update(self, *, elapsed_seconds: float | None = None) -> TrainingUpdate:
        """Collect one rollout, apply PPO once, advance state, alert, and snapshot RNGs."""

        if self.state.update_idx >= self.config.total_updates:
            raise OverflowError("configured update budget is exhausted")
        if (
            self.config.duration_seconds is not None
            and self.state.elapsed_training_seconds >= self.config.duration_seconds
        ):
            raise OverflowError("configured duration budget is exhausted")
        if elapsed_seconds is not None and (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0.0
        ):
            raise ValueError("elapsed_seconds must be finite, non-negative, or None")

        self._last_learning_rate = current_lr(self.config, self.state)
        self._last_entropy_coefficient = current_ent_coef(self.config, self.state)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self._last_learning_rate
        started = time.perf_counter()
        collection = self.collector.collect()
        update = self._updater.update(
            rollout=collection.rollout,
            ent_coef=self._last_entropy_coefficient,
        )
        measured_elapsed = time.perf_counter() - started
        boundary_elapsed = measured_elapsed if elapsed_seconds is None else float(elapsed_seconds)
        self.state.advance_update(self.config, elapsed_seconds=boundary_elapsed)
        if self.state.update_idx % self.config.snapshot_every == 0:
            self.league.add_snapshot(
                update_idx=self.state.update_idx,
                model_state=self.network.state_dict(),
            )
        self.state.env_episode_indices = self.collector.episode_indices
        self.state.league_snapshot_ids = self.league.snapshot_ids
        metrics = dict(collection.metrics)
        metrics.update(update.metrics)
        metrics["charts/SPS"] = (
            self.config.batch_size / boundary_elapsed if boundary_elapsed > 0.0 else 0.0
        )
        try:
            self._alerts.check(
                metrics=metrics,
                progress=schedule_progress(self.config, self.state),
            )
        finally:
            self.state.negative_explained_variance_streak = (
                self._alerts.negative_explained_variance_streak
            )
            self._capture_rngs()
        return TrainingUpdate(
            global_step=self.state.global_step,
            update_idx=self.state.update_idx,
            actions=tuple(int(action) for action in collection.rollout.action.tolist()),
            epochs=update.epochs,
            metrics=metrics,
        )

    def save_checkpoint(
        self,
        path: Path,
        *,
        git_sha: str,
        git_dirty: bool,
    ) -> CheckpointEvidence:
        """Capture current RNGs and atomically persist this completed update boundary."""

        self._capture_rngs()
        return write_checkpoint(
            path=path,
            config=self.config,
            state=self.state,
            network=self.network,
            optimizer=self.optimizer,
            league=self.league,
            collector=self.collector,
            git_sha=git_sha,
            git_dirty=git_dirty,
            learning_rate=self._last_learning_rate,
            entropy_coefficient=self._last_entropy_coefficient,
        )

    def check_evaluation_alerts(self, metrics: dict[str, float]) -> None:
        """Apply evaluation-only hard alerts at the current schedule boundary."""

        self._alerts.check_evaluation(
            metrics=metrics,
            progress=schedule_progress(self.config, self.state),
        )
