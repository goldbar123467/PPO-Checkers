"""Atomic, portable, weights-only Phase 7 update-boundary checkpoints."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast

import numpy as np
import torch

from checkers.config import RunConfig
from checkers.rl.league import LeaguePool
from checkers.rl.networks import CheckersNetwork
from checkers.rl.selfplay import SelfPlayCollector
from checkers.trainer_state import RNGStates, TrainerState

CHECKPOINT_SCHEMA = "CHECKERS_PPO_CHECKPOINT_1"
CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "config",
        "trainer_state",
        "model_state",
        "optimizer_state",
        "league",
        "collector",
        "provenance",
        "schedule",
    }
)
PROVENANCE_FIELDS = frozenset({"git_sha", "git_dirty"})
SCHEDULE_FIELDS = frozenset({"learning_rate", "entropy_coefficient", "phase"})
TRAINER_STATE_FIELDS = frozenset(
    {
        "global_step",
        "update_idx",
        "schedule_phase",
        "elapsed_training_seconds",
        "wandb_run_id",
        "logging_step",
        "env_episode_indices",
        "league_snapshot_ids",
        "negative_explained_variance_streak",
        "rng_states",
        "amp_scaler_state",
    }
)
RNG_FIELDS = frozenset(
    {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "opponent",
        "minibatch",
        "environments",
    }
)
NUMPY_RNG_FIELDS = frozenset({"algorithm", "keys", "position", "has_gauss", "cached_gaussian"})
SHA256_LENGTH = 64
UINT32_MAX = (1 << 32) - 1


class CheckpointError(RuntimeError):
    """Raised when checkpoint integrity, schema, trust, or compatibility checks fail."""


@dataclass(frozen=True, slots=True)
class CheckpointEvidence:
    """Immutable local artifact path, content digest, and size."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated runtime state reconstructed from one trusted local checkpoint."""

    config: RunConfig
    state: TrainerState
    league: LeaguePool
    collector: SelfPlayCollector
    git_sha: str
    git_dirty: bool
    learning_rate: float
    entropy_coefficient: float
    evidence: CheckpointEvidence


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return checked


def _cpu_copy(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _rng_record(snapshot: RNGStates) -> dict[str, object]:
    numpy_state = snapshot.numpy
    return {
        "python": copy.deepcopy(snapshot.python),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": [int(key) for key in numpy_state[1]],
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": snapshot.torch_cpu.detach().cpu().clone(),
        "torch_cuda": [state.detach().cpu().clone() for state in snapshot.torch_cuda],
        "opponent": copy.deepcopy(snapshot.opponent),
        "minibatch": snapshot.minibatch.detach().cpu().clone(),
        "environments": list(copy.deepcopy(snapshot.environments)),
    }


def _trainer_state_record(state: TrainerState) -> dict[str, object]:
    if state.rng_states is None:
        raise ValueError("trainer state must contain complete RNG states")
    return {
        "global_step": state.global_step,
        "update_idx": state.update_idx,
        "schedule_phase": state.schedule_phase,
        "elapsed_training_seconds": state.elapsed_training_seconds,
        "wandb_run_id": state.wandb_run_id,
        "logging_step": state.logging_step,
        "env_episode_indices": list(state.env_episode_indices),
        "league_snapshot_ids": list(state.league_snapshot_ids),
        "negative_explained_variance_streak": state.negative_explained_variance_streak,
        "rng_states": _rng_record(state.rng_states),
        "amp_scaler_state": _cpu_copy(state.amp_scaler_state),
    }


def _validate_boundary(
    *,
    config: RunConfig,
    state: TrainerState,
    league: LeaguePool,
    collector: SelfPlayCollector,
) -> None:
    if state.global_step != state.update_idx * config.batch_size:
        raise ValueError("checkpoint must be written at a complete update boundary")
    if state.update_idx > config.total_updates:
        raise ValueError("trainer update index exceeds configured budget")
    expected_phase = min(1.0, state.global_step / config.total_timesteps)
    if state.schedule_phase != expected_phase:
        raise ValueError("trainer schedule phase disagrees with update boundary")
    if state.env_episode_indices != collector.episode_indices:
        raise ValueError("trainer and collector episode indices disagree")
    if state.league_snapshot_ids != league.snapshot_ids:
        raise ValueError("trainer and league snapshot IDs disagree")


def _model_state(network: CheckersNetwork) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}


def _checkpoint_record(  # noqa: PLR0913
    *,
    config: RunConfig,
    state: TrainerState,
    network: CheckersNetwork,
    optimizer: torch.optim.Optimizer,
    league: LeaguePool,
    collector: SelfPlayCollector,
    git_sha: str,
    git_dirty: bool,
    learning_rate: float,
    entropy_coefficient: float,
) -> dict[str, object]:
    _validate_boundary(config=config, state=state, league=league, collector=collector)
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError("git_sha must be non-empty text")
    if not isinstance(git_dirty, bool):
        raise TypeError("git_dirty must be bool")
    checked_lr = _finite_nonnegative(learning_rate, "learning_rate")
    checked_entropy = _finite_nonnegative(entropy_coefficient, "entropy_coefficient")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config": asdict(config),
        "trainer_state": _trainer_state_record(state),
        "model_state": _model_state(network),
        "optimizer_state": _cpu_copy(optimizer.state_dict()),
        "league": league.to_record(),
        "collector": collector.to_record(),
        "provenance": {"git_sha": git_sha, "git_dirty": git_dirty},
        "schedule": {
            "learning_rate": checked_lr,
            "entropy_coefficient": checked_entropy,
            "phase": state.schedule_phase,
        },
    }


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_save(record: dict[str, object], path: Path) -> CheckpointEvidence:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    digest_path = path.with_suffix(f"{path.suffix}.sha256")
    digest_temporary = temporary.with_name(f"{temporary.name}.sha256.tmp")
    try:
        torch.save(record, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        sha256 = _digest(temporary)
        digest_temporary.write_text(f"{sha256}\n", encoding="ascii")
        with digest_temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        digest_temporary.replace(digest_path)
    finally:
        temporary.unlink(missing_ok=True)
        digest_temporary.unlink(missing_ok=True)
    return CheckpointEvidence(path=path, sha256=sha256, size_bytes=path.stat().st_size)


def save_checkpoint(  # noqa: PLR0913
    *,
    path: Path,
    config: RunConfig,
    state: TrainerState,
    network: CheckersNetwork,
    optimizer: torch.optim.Optimizer,
    league: LeaguePool,
    collector: SelfPlayCollector,
    git_sha: str,
    git_dirty: bool,
    learning_rate: float,
    entropy_coefficient: float,
) -> CheckpointEvidence:
    """Atomically save every exact-resume field at a completed update boundary."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if not isinstance(state, TrainerState):
        raise TypeError("state must be a TrainerState")
    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not isinstance(league, LeaguePool):
        raise TypeError("league must be a LeaguePool")
    if not isinstance(collector, SelfPlayCollector):
        raise TypeError("collector must be a SelfPlayCollector")
    record = _checkpoint_record(
        config=config,
        state=state,
        network=network,
        optimizer=optimizer,
        league=league,
        collector=collector,
        git_sha=git_sha,
        git_dirty=git_dirty,
        learning_rate=learning_rate,
        entropy_coefficient=entropy_coefficient,
    )
    return _atomic_save(record, path)


def _checked_mapping(value: object, fields_: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    record = cast(dict[str, object], value)
    if set(record) != fields_:
        raise ValueError(f"{name} fields are invalid")
    return record


def _rng_from_record(value: object) -> RNGStates:  # noqa: PLR0915
    record = _checked_mapping(value, RNG_FIELDS, "RNG state")
    numpy_record = _checked_mapping(record["numpy"], NUMPY_RNG_FIELDS, "NumPy RNG state")
    algorithm = numpy_record["algorithm"]
    if not isinstance(algorithm, str):
        raise TypeError("NumPy RNG algorithm must be text")
    raw_keys = numpy_record["keys"]
    if not isinstance(raw_keys, list):
        raise TypeError("NumPy RNG keys must be a list")
    keys: list[int] = []
    for key in raw_keys:
        checked = _nonnegative_integer(key, "NumPy RNG key")
        if checked > UINT32_MAX:
            raise ValueError("NumPy RNG key must fit uint32")
        keys.append(checked)
    position = _nonnegative_integer(numpy_record["position"], "NumPy RNG position")
    has_gauss = _nonnegative_integer(numpy_record["has_gauss"], "NumPy RNG has_gauss")
    cached = numpy_record["cached_gaussian"]
    if isinstance(cached, bool) or not isinstance(cached, (int, float)):
        raise TypeError("NumPy cached Gaussian must be numeric")
    torch_cpu = record["torch_cpu"]
    minibatch = record["minibatch"]
    raw_cuda = record["torch_cuda"]
    if not isinstance(torch_cpu, torch.Tensor) or not isinstance(minibatch, torch.Tensor):
        raise TypeError("Torch RNG states must be Tensors")
    if not isinstance(raw_cuda, list) or not all(
        isinstance(item, torch.Tensor) for item in raw_cuda
    ):
        raise TypeError("CUDA RNG states must be a list of Tensors")
    raw_environments = record["environments"]
    if not isinstance(raw_environments, list):
        raise TypeError("environment RNG states must be a list")
    return RNGStates(
        python=copy.deepcopy(record["python"]),
        numpy=(
            algorithm,
            np.asarray(keys, dtype=np.uint32),
            position,
            has_gauss,
            float(cached),
        ),
        torch_cpu=torch_cpu.detach().cpu().clone(),
        torch_cuda=tuple(item.detach().cpu().clone() for item in raw_cuda),
        opponent=copy.deepcopy(record["opponent"]),
        minibatch=minibatch.detach().cpu().clone(),
        environments=tuple(copy.deepcopy(raw_environments)),
    )


def _trainer_state_from_record(value: object) -> TrainerState:
    record = _checked_mapping(value, TRAINER_STATE_FIELDS, "trainer state")
    env_indices_raw = record["env_episode_indices"]
    league_ids_raw = record["league_snapshot_ids"]
    scaler_state = record["amp_scaler_state"]
    if not isinstance(env_indices_raw, list):
        raise TypeError("env_episode_indices must be a list")
    if not isinstance(league_ids_raw, list) or not all(
        isinstance(item, str) and item for item in league_ids_raw
    ):
        raise TypeError("league_snapshot_ids must be non-empty strings")
    if not isinstance(scaler_state, dict):
        raise TypeError("amp_scaler_state must be a mapping")
    run_id = record["wandb_run_id"]
    if not isinstance(run_id, str):
        raise TypeError("wandb_run_id must be text")
    schedule_phase = _finite_nonnegative(record["schedule_phase"], "schedule_phase")
    if schedule_phase > 1.0:
        raise ValueError("schedule_phase must be in [0, 1]")
    return TrainerState(
        global_step=_nonnegative_integer(record["global_step"], "global_step"),
        update_idx=_nonnegative_integer(record["update_idx"], "update_idx"),
        schedule_phase=schedule_phase,
        elapsed_training_seconds=_finite_nonnegative(
            record["elapsed_training_seconds"], "elapsed_training_seconds"
        ),
        wandb_run_id=run_id,
        logging_step=_nonnegative_integer(record["logging_step"], "logging_step"),
        env_episode_indices=tuple(
            _nonnegative_integer(item, "env_episode_indices") for item in env_indices_raw
        ),
        league_snapshot_ids=tuple(cast(list[str], league_ids_raw)),
        negative_explained_variance_streak=_nonnegative_integer(
            record["negative_explained_variance_streak"],
            "negative_explained_variance_streak",
        ),
        rng_states=_rng_from_record(record["rng_states"]),
        amp_scaler_state=cast(dict[str, object], _cpu_copy(scaler_state)),
    )


def _config_from_record(value: object) -> RunConfig:
    if not isinstance(value, dict):
        raise TypeError("checkpoint config must be a mapping")
    record = cast(dict[object, object], value)
    expected = {field.name for field in fields(RunConfig)}
    if set(record) != expected or not all(isinstance(key, str) for key in record):
        raise ValueError("checkpoint config fields are invalid")
    return RunConfig(**cast(dict[str, object], record))  # type: ignore[arg-type]


def _validate_model_state(value: object, network: CheckersNetwork) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("model_state must be a mapping")
    loaded = cast(Mapping[object, object], value)
    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in loaded.items()
    ):
        raise TypeError("model_state must map strings to Tensors")
    state = cast(dict[str, torch.Tensor], dict(loaded))
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("model_state keys do not match network")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"model_state tensor metadata mismatch for {name}")
    return state


def _load_record(path: Path) -> tuple[dict[str, object], CheckpointEvidence]:
    if not path.is_file():
        raise CheckpointError("checkpoint file does not exist")
    digest_path = path.with_suffix(f"{path.suffix}.sha256")
    try:
        expected_digest = digest_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise CheckpointError("checkpoint digest file is missing") from error
    if len(expected_digest) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise CheckpointError("checkpoint digest record is invalid")
    actual_digest = _digest(path)
    if actual_digest != expected_digest:
        raise CheckpointError("checkpoint digest mismatch")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise CheckpointError("weights-only checkpoint load failed") from error
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint root must be a mapping")
    return cast(dict[str, object], raw), CheckpointEvidence(
        path=path,
        sha256=actual_digest,
        size_bytes=path.stat().st_size,
    )


def load_checkpoint(
    *,
    path: Path,
    expected_config: RunConfig,
    network: CheckersNetwork,
    optimizer: torch.optim.Optimizer,
) -> LoadedCheckpoint:
    """Verify and load a compatible trusted-local checkpoint without unsafe pickle globals."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(expected_config, RunConfig):
        raise TypeError("expected_config must be a RunConfig")
    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    record, evidence = _load_record(path)
    try:
        if set(record) != CHECKPOINT_FIELDS:
            raise ValueError("checkpoint fields are invalid")
        if record["schema"] != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema is unsupported")
        config = _config_from_record(record["config"])
        if config != expected_config:
            raise ValueError("checkpoint config does not match expected config")
        state = _trainer_state_from_record(record["trainer_state"])
        league = LeaguePool.from_record(record["league"])
        model_state = _validate_model_state(record["model_state"], network)
        optimizer_state = record["optimizer_state"]
        if not isinstance(optimizer_state, dict):
            raise TypeError("optimizer_state must be a mapping")
        provenance = _checked_mapping(record["provenance"], PROVENANCE_FIELDS, "provenance")
        git_sha = provenance["git_sha"]
        git_dirty = provenance["git_dirty"]
        if not isinstance(git_sha, str) or not git_sha:
            raise ValueError("checkpoint git_sha must be non-empty text")
        if not isinstance(git_dirty, bool):
            raise TypeError("checkpoint git_dirty must be bool")
        schedule = _checked_mapping(record["schedule"], SCHEDULE_FIELDS, "schedule")
        learning_rate = _finite_nonnegative(schedule["learning_rate"], "learning_rate")
        entropy_coefficient = _finite_nonnegative(
            schedule["entropy_coefficient"], "entropy_coefficient"
        )
        phase = _finite_nonnegative(schedule["phase"], "schedule phase")
        if phase != state.schedule_phase:
            raise ValueError("checkpoint schedule phases disagree")
        network.to(torch.device(config.device))
        collector = SelfPlayCollector.from_record(
            config=config,
            network=network,
            record=record["collector"],
        )
        _validate_boundary(config=config, state=state, league=league, collector=collector)
        network.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise CheckpointError(f"invalid checkpoint: {error}") from error
    return LoadedCheckpoint(
        config=config,
        state=state,
        league=league,
        collector=collector,
        git_sha=git_sha,
        git_dirty=git_dirty,
        learning_rate=learning_rate,
        entropy_coefficient=entropy_coefficient,
        evidence=evidence,
    )
