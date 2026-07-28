"""Phase 7 rollout-consumption, epoch-ledger, and directional update tests."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from checkers.config import RunConfig
from checkers.rl.buffer import RolloutBatch
from checkers.rl.determinism import seed_everything
from checkers.rl.networks import CheckersNetwork
from checkers.rl.ppo import PPOUpdateMetrics
from checkers.rl.selfplay import SelfPlayCollector
from checkers.train import RolloutUpdater

EXPECTED_TRANSITIONS = 4
EXPECTED_OPTIMIZER_STEPS = 6
EARLY_STOP_OPTIMIZER_STEPS = 3


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="updater-unit",
        seed=19,
        device="cpu",
        total_timesteps=4,
        duration_seconds=None,
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        update_epochs=3,
        learning_rate=1e-3,
        target_kl=100.0,
        eval_games=2,
    )


def _setup() -> tuple[
    RunConfig,
    CheckersNetwork,
    torch.optim.Optimizer,
    torch.Generator,
    RolloutBatch,
]:
    config = _config()
    seed_everything(config.seed, num_envs=config.num_envs, deterministic=True)
    network = CheckersNetwork()
    rollout = SelfPlayCollector(config=config, network=network).collect().rollout
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    generator = torch.Generator().manual_seed(123)
    return config, network, optimizer, generator, rollout


def test_u1_each_row_is_used_once_per_epoch_then_rollout_is_discarded() -> None:
    config, network, optimizer, generator, rollout = _setup()
    original_config = asdict(config)
    expected_generator = torch.Generator().manual_seed(123)
    expected = tuple(
        tuple(torch.randperm(EXPECTED_TRANSITIONS, generator=expected_generator).tolist())
        for _ in range(config.update_epochs)
    )
    updater = RolloutUpdater(
        config=config,
        network=network,
        optimizer=optimizer,
        minibatch_generator=generator,
    )

    result = updater.update(rollout=rollout, ent_coef=config.ent_coef_start)

    assert tuple(epoch.source_indices for epoch in result.epochs) == expected
    assert all(epoch.complete for epoch in result.epochs)
    assert result.optimizer_steps == EXPECTED_OPTIMIZER_STEPS
    assert not result.kl_early_stopped
    assert result.metrics["train/trainable_frac"] == 1.0
    assert result.metrics["train/ent_coef"] == config.ent_coef_start
    assert result.metrics["train/lr"] == config.learning_rate
    assert asdict(config) == original_config
    with pytest.raises(RuntimeError, match="already consumed"):
        updater.update(rollout=rollout, ent_coef=config.ent_coef_start)


def test_u2_complete_update_changes_parameters_and_reports_finite_metrics() -> None:
    config, network, optimizer, generator, rollout = _setup()
    before = tuple(parameter.detach().clone() for parameter in network.parameters())
    updater = RolloutUpdater(
        config=config,
        network=network,
        optimizer=optimizer,
        minibatch_generator=generator,
    )

    result = updater.update(rollout=rollout, ent_coef=config.ent_coef_start)

    assert any(
        not torch.equal(parameter.detach(), original)
        for parameter, original in zip(network.parameters(), before, strict=True)
    )
    assert result.transitions == EXPECTED_TRANSITIONS
    assert set(result.metrics) == {
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
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())


def test_u1_kl_early_stop_retains_a_truthful_partial_epoch_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, network, optimizer, generator, rollout = _setup()
    calls = 0

    def fake_update(**_: object) -> PPOUpdateMetrics:
        nonlocal calls
        calls += 1
        return PPOUpdateMetrics(
            policy_loss=float(calls),
            value_loss=2.0,
            entropy=3.0,
            total_loss=4.0,
            approx_kl=5.0,
            clipfrac=0.5,
            grad_norm=0.25,
            kl_early_stop=calls == EARLY_STOP_OPTIMIZER_STEPS,
            transitions=2,
        )

    monkeypatch.setattr("checkers.train.ppo_minibatch_update", fake_update)
    expected_generator = torch.Generator().manual_seed(123)
    first_epoch = tuple(torch.randperm(4, generator=expected_generator).tolist())
    second_permutation = tuple(torch.randperm(4, generator=expected_generator).tolist())
    updater = RolloutUpdater(
        config=config,
        network=network,
        optimizer=optimizer,
        minibatch_generator=generator,
    )

    result = updater.update(rollout=rollout, ent_coef=0.0)

    assert calls == EARLY_STOP_OPTIMIZER_STEPS
    assert result.optimizer_steps == EARLY_STOP_OPTIMIZER_STEPS
    assert result.kl_early_stopped
    assert result.metrics["train/kl_early_stops"] == 1.0
    assert result.epochs[0].source_indices == first_epoch
    assert result.epochs[0].complete
    assert result.epochs[1].source_indices == second_permutation[:2]
    assert not result.epochs[1].complete
