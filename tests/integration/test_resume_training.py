"""Forked end-to-end CPU training resume, including a mid-capture checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from checkers.config import RunConfig
from checkers.rules.state import PlayerId, State
from checkers.train import TrainingSession

TOTAL_UPDATES = 11
POST_CHECKPOINT_UPDATES = 10


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _multijump_state() -> State:
    return State(
        men=(_mask(9), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="resume-cpu-bitwise",
        seed=59,
        device="cpu",
        total_timesteps=TOTAL_UPDATES,
        duration_seconds=None,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        snapshot_every=20,
        eval_games=2,
    )


def test_r2_ten_updates_after_midsequence_resume_are_cpu_bitwise_equal(
    tmp_path: Path,
) -> None:
    config = _config()
    uninterrupted = TrainingSession.create(
        config=config,
        initial_states=(_multijump_state(),),
    )
    first = uninterrupted.run_update(elapsed_seconds=0.0)
    assert first.update_idx == 1
    assert uninterrupted.collector.vector_env.envs[0].state.capture_in_progress
    checkpoint = tmp_path / "update-000001.pt"
    uninterrupted.save_checkpoint(
        checkpoint,
        git_sha="0123456789abcdef",
        git_dirty=True,
    )

    expected = tuple(
        uninterrupted.run_update(elapsed_seconds=0.0) for _ in range(POST_CHECKPOINT_UPDATES)
    )
    resumed = TrainingSession.resume(config=config, checkpoint_path=checkpoint)
    actual = tuple(resumed.run_update(elapsed_seconds=0.0) for _ in range(POST_CHECKPOINT_UPDATES))

    assert [update.update_idx for update in expected] == list(range(2, TOTAL_UPDATES + 1))
    assert [update.update_idx for update in actual] == list(range(2, TOTAL_UPDATES + 1))
    for expected_update, actual_update in zip(expected, actual, strict=True):
        assert actual_update.actions == expected_update.actions
        assert actual_update.epochs == expected_update.epochs
        assert actual_update.metrics == expected_update.metrics
    for field in (
        "global_step",
        "update_idx",
        "schedule_phase",
        "elapsed_training_seconds",
        "wandb_run_id",
        "logging_step",
        "env_episode_indices",
        "league_snapshot_ids",
        "negative_explained_variance_streak",
        "amp_scaler_state",
    ):
        assert getattr(resumed.state, field) == getattr(uninterrupted.state, field)
    resumed_rng = resumed.state.rng_states
    expected_rng = uninterrupted.state.rng_states
    assert resumed_rng is not None and expected_rng is not None
    assert resumed_rng.python == expected_rng.python
    assert resumed_rng.opponent == expected_rng.opponent
    assert resumed_rng.environments == expected_rng.environments
    assert resumed_rng.numpy[0] == expected_rng.numpy[0]
    assert np.array_equal(resumed_rng.numpy[1], expected_rng.numpy[1])
    assert resumed_rng.numpy[2:] == expected_rng.numpy[2:]
    assert torch.equal(resumed_rng.torch_cpu, expected_rng.torch_cpu)
    assert torch.equal(resumed_rng.minibatch, expected_rng.minibatch)
    assert all(
        torch.equal(actual_state, expected_state)
        for actual_state, expected_state in zip(
            resumed_rng.torch_cuda, expected_rng.torch_cuda, strict=True
        )
    )
    assert resumed.collector.to_record() == uninterrupted.collector.to_record()
    assert resumed.league.snapshot_ids == uninterrupted.league.snapshot_ids
    for name, expected_tensor in uninterrupted.network.state_dict().items():
        assert torch.equal(resumed.network.state_dict()[name], expected_tensor)
