"""Atomic, weights-only, update-boundary checkpoint and resume tests."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from checkers.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from checkers.config import RunConfig
from checkers.rl.determinism import seed_everything
from checkers.rl.league import LeaguePool
from checkers.rl.networks import CheckersNetwork
from checkers.rl.selfplay import SelfPlayCollector
from checkers.rules.state import PlayerId, State
from checkers.train import RolloutUpdater
from checkers.trainer_state import TrainerState, capture_rng_states, restore_rng_states

RESUMED_LOGGING_STEP = 5


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="checkpoint-unit",
        seed=31,
        device="cpu",
        total_updates=1,
        schedule_horizon_updates=1,
        duration_seconds=None,
        num_envs=2,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        eval_games=2,
    )


def _multijump_state() -> State:
    return State(
        men=(_mask(9), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _live_state() -> tuple[
    RunConfig,
    CheckersNetwork,
    torch.optim.Optimizer,
    TrainerState,
    LeaguePool,
    SelfPlayCollector,
    random.Random,
    torch.Generator,
    tuple[random.Random, ...],
]:
    config = _config()
    seed_everything(config.seed, num_envs=config.num_envs, deterministic=True)
    network = CheckersNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    minibatch_generator = torch.Generator().manual_seed(41)
    collector = SelfPlayCollector(
        config=config,
        network=network,
        initial_states=(_multijump_state(), State.initial()),
    )
    league = LeaguePool(capacity=config.pool_capacity)
    league.pin_initial(network.state_dict())
    rollout = collector.collect().rollout
    RolloutUpdater(
        config=config,
        network=network,
        optimizer=optimizer,
        minibatch_generator=minibatch_generator,
    ).update(rollout=rollout, ent_coef=config.ent_coef_start)
    state = TrainerState(
        wandb_run_id="resume-id",
        logging_step=RESUMED_LOGGING_STEP,
        env_episode_indices=collector.episode_indices,
        league_snapshot_ids=league.snapshot_ids,
    )
    state.advance_update(config, elapsed_seconds=1.25)
    opponent_rng = random.Random(42)
    env_rngs = (random.Random(43), random.Random(44))
    state.rng_states = capture_rng_states(
        opponent_rng=opponent_rng,
        minibatch_generator=minibatch_generator,
        env_rngs=env_rngs,
    )
    return (
        config,
        network,
        optimizer,
        state,
        league,
        collector,
        opponent_rng,
        minibatch_generator,
        env_rngs,
    )


def test_r1_atomic_checkpoint_round_trip_restores_every_required_state(tmp_path: Path) -> None:
    (
        config,
        network,
        optimizer,
        state,
        league,
        collector,
        opponent_rng,
        minibatch_generator,
        env_rngs,
    ) = _live_state()
    checkpoint_path = tmp_path / "update-000001.pt"
    model_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
    vector_snapshot = collector.vector_env.serialize()
    rng_snapshot = state.rng_states

    evidence = save_checkpoint(
        path=checkpoint_path,
        config=config,
        state=state,
        network=network,
        optimizer=optimizer,
        league=league,
        collector=collector,
        git_sha="0123456789abcdef",
        git_dirty=True,
        learning_rate=config.learning_rate,
        entropy_coefficient=config.ent_coef_start,
    )

    assert evidence.path == checkpoint_path
    assert evidence.size_bytes == checkpoint_path.stat().st_size
    assert evidence.sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert checkpoint_path.with_suffix(".pt.sha256").read_text(encoding="ascii").strip() == (
        evidence.sha256
    )
    assert not tuple(tmp_path.glob("*.tmp"))

    restored_network = CheckersNetwork()
    restored_optimizer = torch.optim.Adam(
        restored_network.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )
    loaded = load_checkpoint(
        path=checkpoint_path,
        expected_config=config,
        network=restored_network,
        optimizer=restored_optimizer,
    )

    assert loaded.config == config
    assert loaded.state.global_step == config.batch_size
    assert loaded.state.update_idx == 1
    assert loaded.state.logging_step == RESUMED_LOGGING_STEP
    assert loaded.state.wandb_run_id == "resume-id"
    assert loaded.state.env_episode_indices == collector.episode_indices
    assert loaded.league.snapshot_ids == league.snapshot_ids
    assert loaded.collector.vector_env.serialize() == vector_snapshot
    assert loaded.collector.vector_env.envs[0].state.capture_in_progress
    assert loaded.git_sha == "0123456789abcdef"
    assert loaded.git_dirty
    assert loaded.learning_rate == config.learning_rate
    assert loaded.entropy_coefficient == config.ent_coef_start
    for name, tensor in restored_network.state_dict().items():
        assert torch.equal(tensor, model_state[name])

    assert rng_snapshot is not None
    expected_rng_values = (
        opponent_rng.random(),
        float(torch.rand((), generator=minibatch_generator)),
        tuple(stream.random() for stream in env_rngs),
    )
    restored_opponent = random.Random()
    restored_minibatch = torch.Generator()
    restored_envs = (random.Random(), random.Random())
    assert loaded.state.rng_states is not None
    restore_rng_states(
        loaded.state.rng_states,
        opponent_rng=restored_opponent,
        minibatch_generator=restored_minibatch,
        env_rngs=restored_envs,
    )
    actual_rng_values = (
        restored_opponent.random(),
        float(torch.rand((), generator=restored_minibatch)),
        tuple(stream.random() for stream in restored_envs),
    )
    assert actual_rng_values == expected_rng_values


def test_r1_save_rejects_non_boundary_or_incomplete_state(tmp_path: Path) -> None:
    config, network, optimizer, state, league, collector, *_ = _live_state()
    state.global_step -= 1
    with pytest.raises(ValueError, match="update boundary"):
        save_checkpoint(
            path=tmp_path / "bad.pt",
            config=config,
            state=state,
            network=network,
            optimizer=optimizer,
            league=league,
            collector=collector,
            git_sha="abc",
            git_dirty=False,
            learning_rate=config.learning_rate,
            entropy_coefficient=config.ent_coef_start,
        )


def test_r1_load_rejects_config_mismatch_and_digest_corruption(tmp_path: Path) -> None:
    config, network, optimizer, state, league, collector, *_ = _live_state()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path=path,
        config=config,
        state=state,
        network=network,
        optimizer=optimizer,
        league=league,
        collector=collector,
        git_sha="abc",
        git_dirty=False,
        learning_rate=config.learning_rate,
        entropy_coefficient=config.ent_coef_start,
    )
    target_network = CheckersNetwork()
    target_optimizer = torch.optim.Adam(target_network.parameters())
    with pytest.raises(CheckpointError, match="config"):
        load_checkpoint(
            path=path,
            expected_config=replace(config, experiment_id="different"),
            network=target_network,
            optimizer=target_optimizer,
        )

    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CheckpointError, match="digest"):
        load_checkpoint(
            path=path,
            expected_config=config,
            network=target_network,
            optimizer=target_optimizer,
        )


class _UntrustedPayload:
    pass


def test_r1_weights_only_loader_rejects_untrusted_pickle_global(tmp_path: Path) -> None:
    path = tmp_path / "untrusted.pt"
    torch.save({"payload": _UntrustedPayload()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".pt.sha256").write_text(f"{digest}\n", encoding="ascii")
    config = _config()
    network = CheckersNetwork()
    optimizer = torch.optim.Adam(network.parameters())

    with pytest.raises(CheckpointError, match="weights-only"):
        load_checkpoint(
            path=path,
            expected_config=config,
            network=network,
            optimizer=optimizer,
        )
