"""Phase 7 configuration, schedule, and mutable-state contracts."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest
import torch
import yaml

from checkers.config import RunConfig, load_run_config
from checkers.schedules import current_ent_coef, current_lr, schedule_progress
from checkers.trainer_state import (
    TrainerState,
    capture_rng_states,
    restore_rng_states,
)

TOTAL_TIMESTEPS = 320
BATCH_SIZE = 32
MINIBATCH_SIZE = 8
TOTAL_UPDATES = 10
QUARTER_PROGRESS = 0.25
UPDATE_SECONDS = 1.25


def _config(**overrides: object) -> RunConfig:
    values: dict[str, object] = {
        "experiment_id": "phase7-unit",
        "seed": 7,
        "device": "cpu",
        "total_timesteps": TOTAL_TIMESTEPS,
        "duration_seconds": None,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 4,
        "update_epochs": 2,
        "eval_games": 364,
        "periodic_eval_games": 2,
        "exploitability_train_games": 16,
    }
    values.update(overrides)
    return RunConfig(**values)  # type: ignore[arg-type]


def test_c1_run_config_is_frozen_typed_and_derives_exact_batch_sizes() -> None:
    config = _config()

    assert config.batch_size == BATCH_SIZE
    assert config.minibatch_size == MINIBATCH_SIZE
    assert config.total_updates == TOTAL_UPDATES
    assert config.validate() is config
    with pytest.raises(FrozenInstanceError):
        config.seed = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"experiment_id": " "}, ValueError, "experiment_id"),
        ({"experiment_id": 1}, TypeError, "experiment_id"),
        ({"seed": True}, TypeError, "seed"),
        ({"seed": -1}, ValueError, "unsigned 64-bit"),
        ({"phase": 6}, ValueError, "phase must be 7"),
        ({"stage": "smoke"}, ValueError, "stage"),
        ({"arm": "A4"}, ValueError, "arm"),
        ({"device": "tpu"}, ValueError, "device"),
        ({"deterministic": 1}, TypeError, "deterministic"),
        ({"total_timesteps": 0}, ValueError, "total_timesteps"),
        ({"total_timesteps": 321}, ValueError, "whole rollout"),
        ({"duration_seconds": 0.0}, ValueError, "duration_seconds"),
        ({"num_envs": 0}, ValueError, "num_envs"),
        ({"num_steps": 0}, ValueError, "num_steps"),
        ({"num_minibatches": 3}, ValueError, "num_minibatches"),
        ({"update_epochs": 0}, ValueError, "update_epochs"),
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"learning_rate": "fast"}, TypeError, "learning_rate"),
        ({"learning_rate": float("inf")}, ValueError, "finite"),
        ({"gamma": 1.01}, ValueError, "gamma"),
        ({"gae_lambda": -0.1}, ValueError, "gae_lambda"),
        ({"clip_coef": 0.0}, ValueError, "clip_coef"),
        ({"vf_coef": -0.1}, ValueError, "vf_coef"),
        ({"ent_coef_start": -0.1}, ValueError, "ent_coef_start"),
        ({"ent_coef_end": 0.02}, ValueError, "ent_coef_end"),
        ({"ent_anneal_fraction": 0.0}, ValueError, "ent_anneal_fraction"),
        ({"max_grad_norm": 0.0}, ValueError, "max_grad_norm"),
        ({"target_kl": 0.0}, ValueError, "target_kl"),
        ({"adam_eps": 0.0}, ValueError, "adam_eps"),
        ({"max_plies": 0}, ValueError, "max_plies"),
        ({"repetition_draws": 1}, TypeError, "repetition_draws"),
        ({"snapshot_every": 0}, ValueError, "snapshot_every"),
        ({"pool_capacity": 1}, ValueError, "pool_capacity"),
        ({"eval_every": 0}, ValueError, "eval_every"),
        ({"checkpoint_every": 0}, ValueError, "checkpoint_every"),
        ({"eval_games": 363}, ValueError, "eval_games"),
        ({"periodic_eval_games": 3}, ValueError, "periodic_eval_games"),
        ({"exploitability_train_games": 3}, ValueError, "exploitability_train_games"),
        ({"amp_dtype": "float16"}, ValueError, "amp_dtype"),
        ({"include_opponent_value_loss": 1}, TypeError, "include_opponent_value_loss"),
        ({"wandb_mode": "online"}, ValueError, "wandb_mode"),
    ],
)
def test_c1_invalid_run_configuration_raises(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _config(**overrides)


def test_c1_load_run_config_rejects_unknown_or_missing_yaml_fields() -> None:
    config = _config()
    text = yaml.safe_dump(asdict(config), sort_keys=False)

    loaded = load_run_config(text)

    assert loaded == config
    with pytest.raises(ValueError, match="unknown"):
        load_run_config(f"{text}\nunknown_field: 1\n")
    with pytest.raises(ValueError, match="missing"):
        load_run_config("experiment_id: incomplete")
    with pytest.raises(TypeError, match="mapping"):
        load_run_config("- not\n- a\n- mapping\n")
    with pytest.raises(TypeError, match="text"):
        load_run_config(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid"):
        load_run_config("root: [unterminated")


def test_c2_schedules_hit_hand_computed_endpoints_without_mutating_config() -> None:
    config = _config()
    original = asdict(config)
    state = TrainerState()

    assert schedule_progress(config, state) == 0.0
    assert current_lr(config, state) == pytest.approx(3e-4, abs=0.0)
    assert current_ent_coef(config, state) == pytest.approx(0.01, abs=0.0)

    state.global_step = TOTAL_TIMESTEPS // 4
    assert schedule_progress(config, state) == QUARTER_PROGRESS
    assert current_lr(config, state) == pytest.approx(2.25e-4)
    assert current_ent_coef(config, state) == pytest.approx(0.0055)

    state.global_step = TOTAL_TIMESTEPS // 2
    assert current_lr(config, state) == pytest.approx(1.5e-4)
    assert current_ent_coef(config, state) == pytest.approx(0.001)

    state.global_step = TOTAL_TIMESTEPS
    assert schedule_progress(config, state) == 1.0
    assert current_lr(config, state) == 0.0
    assert current_ent_coef(config, state) == pytest.approx(0.001, abs=0.0)
    assert asdict(config) == original


def test_c2_schedule_helpers_reject_wrong_runtime_objects() -> None:
    config = _config()
    state = TrainerState()
    with pytest.raises(TypeError, match="config"):
        schedule_progress(object(), state)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state"):
        current_lr(config, object())  # type: ignore[arg-type]


def test_trainer_state_advances_only_by_complete_rollouts() -> None:
    config = _config()
    state = TrainerState(env_episode_indices=(0,) * config.num_envs)

    state.advance_update(config, elapsed_seconds=UPDATE_SECONDS)

    assert state.global_step == config.batch_size
    assert state.update_idx == 1
    assert state.elapsed_training_seconds == UPDATE_SECONDS
    assert state.schedule_phase == schedule_progress(config, state)
    state.advance_logging_step()
    assert state.logging_step == 1
    with pytest.raises(ValueError, match="elapsed_seconds"):
        state.advance_update(config, elapsed_seconds=-1.0)
    with pytest.raises(TypeError, match="config"):
        state.advance_update(object(), elapsed_seconds=0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="elapsed_seconds"):
        state.advance_update(config, elapsed_seconds="slow")  # type: ignore[arg-type]


def test_trainer_state_rejects_update_after_budget_exhaustion() -> None:
    config = _config(total_timesteps=BATCH_SIZE)
    state = TrainerState()
    state.advance_update(config, elapsed_seconds=0.0)
    with pytest.raises(OverflowError, match="exhausted"):
        state.advance_update(config, elapsed_seconds=0.0)


def test_rng_snapshot_restores_python_numpy_torch_opponent_minibatch_and_env_streams() -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    opponent = random.Random(104)
    minibatch = torch.Generator().manual_seed(105)
    env_streams = tuple(random.Random(seed) for seed in (106, 107))
    snapshot = capture_rng_states(
        opponent_rng=opponent,
        minibatch_generator=minibatch,
        env_rngs=env_streams,
    )
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        opponent.random(),
        float(torch.rand((), generator=minibatch)),
        tuple(stream.random() for stream in env_streams),
    )
    for _ in range(3):
        random.random()
        np.random.random()
        torch.rand(())
        opponent.random()
        torch.rand((), generator=minibatch)
        for stream in env_streams:
            stream.random()

    restore_rng_states(
        snapshot,
        opponent_rng=opponent,
        minibatch_generator=minibatch,
        env_rngs=env_streams,
    )
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        opponent.random(),
        float(torch.rand((), generator=minibatch)),
        tuple(stream.random() for stream in env_streams),
    )

    assert actual == expected


def test_rng_restore_rejects_wrong_environment_lane_count() -> None:
    snapshot = capture_rng_states(
        opponent_rng=random.Random(1),
        minibatch_generator=torch.Generator().manual_seed(2),
        env_rngs=(random.Random(3),),
    )
    with pytest.raises(ValueError, match="environment RNG"):
        restore_rng_states(
            snapshot,
            opponent_rng=random.Random(1),
            minibatch_generator=torch.Generator().manual_seed(2),
            env_rngs=(random.Random(3), random.Random(4)),
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"opponent_rng": object()}, "opponent_rng"),
        ({"minibatch_generator": object()}, "minibatch_generator"),
        ({"env_rngs": [random.Random(3)]}, "env_rngs"),
    ],
)
def test_capture_rng_states_rejects_invalid_stream_objects(
    arguments: dict[str, object],
    message: str,
) -> None:
    complete: dict[str, object] = {
        "opponent_rng": random.Random(1),
        "minibatch_generator": torch.Generator().manual_seed(2),
        "env_rngs": (random.Random(3),),
    }
    complete.update(arguments)
    with pytest.raises(TypeError, match=message):
        capture_rng_states(**complete)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("snapshot", object(), "snapshot"),
        ("opponent_rng", object(), "opponent_rng"),
        ("minibatch_generator", object(), "minibatch_generator"),
        ("env_rngs", [random.Random(3)], "env_rngs"),
    ],
)
def test_restore_rng_states_rejects_invalid_objects(
    field: str,
    value: object,
    message: str,
) -> None:
    snapshot = capture_rng_states(
        opponent_rng=random.Random(1),
        minibatch_generator=torch.Generator().manual_seed(2),
        env_rngs=(random.Random(3),),
    )
    complete: dict[str, object] = {
        "snapshot": snapshot,
        "opponent_rng": random.Random(1),
        "minibatch_generator": torch.Generator().manual_seed(2),
        "env_rngs": (random.Random(3),),
    }
    complete[field] = value
    with pytest.raises(TypeError, match=message):
        restore_rng_states(**complete)  # type: ignore[arg-type]


def test_restore_rng_states_rejects_cuda_device_count_mismatch() -> None:
    snapshot = capture_rng_states(
        opponent_rng=random.Random(1),
        minibatch_generator=torch.Generator().manual_seed(2),
        env_rngs=(random.Random(3),),
    )
    bad = replace(snapshot, torch_cuda=snapshot.torch_cuda + (torch.zeros(1, dtype=torch.uint8),))
    with pytest.raises(ValueError, match="CUDA RNG"):
        restore_rng_states(
            bad,
            opponent_rng=random.Random(1),
            minibatch_generator=torch.Generator().manual_seed(2),
            env_rngs=(random.Random(3),),
        )


def test_rng_snapshot_supports_cpu_only_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    snapshot = capture_rng_states(
        opponent_rng=random.Random(1),
        minibatch_generator=torch.Generator().manual_seed(2),
        env_rngs=(random.Random(3),),
    )

    restore_rng_states(
        snapshot,
        opponent_rng=random.Random(1),
        minibatch_generator=torch.Generator().manual_seed(2),
        env_rngs=(random.Random(3),),
    )

    assert snapshot.torch_cuda == ()
