"""Small, checksummed, weights-only inference bundles for the checkers web harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from checkers.config import RunConfig
from checkers.rl.networks import ACTION_COUNT, OBSERVATION_SHAPE, CheckersNetwork

POLICY_BUNDLE_SCHEMA = "CHECKERS_POLICY_BUNDLE_1"
POLICY_BUNDLE_FIELDS = frozenset({"schema", "metadata", "model_state"})
METADATA_FIELDS = frozenset(
    {
        "bundle_id",
        "experiment_id",
        "update_idx",
        "global_step",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_checkpoint_size_bytes",
        "source_git_sha",
        "source_git_dirty",
        "config_sha256",
        "max_plies",
        "repetition_draws",
        "network_class",
        "observation_shape",
        "action_count",
    }
)
SHA256_LENGTH = 64


class PolicyBundleError(RuntimeError):
    """Raised when an inference bundle fails integrity or schema validation."""


@dataclass(frozen=True, slots=True)
class PolicyBundleMetadata:
    """Immutable provenance and runtime contract for one exported policy."""

    bundle_id: str
    experiment_id: str
    update_idx: int
    global_step: int
    source_checkpoint: str
    source_checkpoint_sha256: str
    source_checkpoint_size_bytes: int
    source_git_sha: str
    source_git_dirty: bool
    config_sha256: str
    max_plies: int
    repetition_draws: bool
    network_class: str = "checkers.rl.networks.CheckersNetwork"
    observation_shape: tuple[int, int, int] = OBSERVATION_SHAPE
    action_count: int = ACTION_COUNT

    def __post_init__(self) -> None:
        text_fields = {
            "bundle_id": self.bundle_id,
            "experiment_id": self.experiment_id,
            "source_checkpoint": self.source_checkpoint,
            "source_git_sha": self.source_git_sha,
            "network_class": self.network_class,
        }
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValueError("policy metadata text fields must be non-empty")
        _nonnegative_integer(self.update_idx, "update_idx")
        _nonnegative_integer(self.global_step, "global_step")
        _positive_integer(self.source_checkpoint_size_bytes, "source_checkpoint_size_bytes")
        _positive_integer(self.max_plies, "max_plies")
        if not isinstance(self.source_git_dirty, bool):
            raise TypeError("source_git_dirty must be bool")
        if not isinstance(self.repetition_draws, bool):
            raise TypeError("repetition_draws must be bool")
        for name, digest in (
            ("source_checkpoint_sha256", self.source_checkpoint_sha256),
            ("config_sha256", self.config_sha256),
        ):
            _validate_digest(digest, name)
        if self.network_class != "checkers.rl.networks.CheckersNetwork":
            raise ValueError("unsupported policy network class")
        if self.observation_shape != OBSERVATION_SHAPE:
            raise ValueError("unsupported observation shape")
        if self.action_count != ACTION_COUNT:
            raise ValueError("unsupported action count")


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    """A strictly loaded CPU policy and its validated provenance."""

    network: CheckersNetwork
    metadata: PolicyBundleMetadata
    path: Path
    sha256: str
    size_bytes: int


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_integer(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked < 1:
        raise ValueError(f"{name} must be positive")
    return checked


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local file."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def config_sha256(config: RunConfig) -> str:
    """Hash the exact validated run configuration in canonical JSON form."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _model_state(network: CheckersNetwork) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}


def _atomic_save(record: dict[str, object], path: Path) -> tuple[str, int]:
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
        digest = sha256_file(temporary)
        digest_temporary.write_text(f"{digest}\n", encoding="ascii")
        with digest_temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        digest_temporary.replace(digest_path)
    finally:
        temporary.unlink(missing_ok=True)
        digest_temporary.unlink(missing_ok=True)
    return digest, path.stat().st_size


def save_policy_bundle(
    *, path: Path, network: CheckersNetwork, metadata: PolicyBundleMetadata
) -> tuple[str, int]:
    """Atomically save a model-only, weights-only-loadable policy bundle."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    if not isinstance(metadata, PolicyBundleMetadata):
        raise TypeError("metadata must be PolicyBundleMetadata")
    record: dict[str, object] = {
        "schema": POLICY_BUNDLE_SCHEMA,
        "metadata": {**asdict(metadata), "observation_shape": list(metadata.observation_shape)},
        "model_state": _model_state(network),
    }
    return _atomic_save(record, path)


def _read_expected_digest(path: Path) -> str:
    digest_path = path.with_suffix(f"{path.suffix}.sha256")
    try:
        digest = digest_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise PolicyBundleError("policy bundle digest file is missing") from error
    try:
        return _validate_digest(digest, "policy bundle digest")
    except ValueError as error:
        raise PolicyBundleError(str(error)) from error


def _metadata_from_record(value: object) -> PolicyBundleMetadata:
    if not isinstance(value, dict) or set(value) != METADATA_FIELDS:
        raise ValueError("policy metadata fields are invalid")
    record = cast(dict[str, object], value)
    shape = record["observation_shape"]
    if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
        raise TypeError("observation_shape must be a list of integers")
    return PolicyBundleMetadata(
        bundle_id=cast(str, record["bundle_id"]),
        experiment_id=cast(str, record["experiment_id"]),
        update_idx=cast(int, record["update_idx"]),
        global_step=cast(int, record["global_step"]),
        source_checkpoint=cast(str, record["source_checkpoint"]),
        source_checkpoint_sha256=cast(str, record["source_checkpoint_sha256"]),
        source_checkpoint_size_bytes=cast(int, record["source_checkpoint_size_bytes"]),
        source_git_sha=cast(str, record["source_git_sha"]),
        source_git_dirty=cast(bool, record["source_git_dirty"]),
        config_sha256=cast(str, record["config_sha256"]),
        max_plies=cast(int, record["max_plies"]),
        repetition_draws=cast(bool, record["repetition_draws"]),
        network_class=cast(str, record["network_class"]),
        observation_shape=cast(tuple[int, int, int], tuple(shape)),
        action_count=cast(int, record["action_count"]),
    )


def _validated_model_state(value: object, network: CheckersNetwork) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("model_state must be a mapping")
    loaded = cast(Mapping[object, object], value)
    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in loaded.items()
    ):
        raise TypeError("model_state must map strings to tensors")
    state = cast(dict[str, torch.Tensor], dict(loaded))
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("model_state keys do not match CheckersNetwork")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"model_state tensor metadata mismatch for {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"model_state tensor contains non-finite values for {name}")
    return state


def load_policy_bundle(path: Path) -> LoadedPolicy:
    """Verify and strictly load a local policy bundle onto CPU."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.is_file():
        raise PolicyBundleError("policy bundle does not exist")
    expected_digest = _read_expected_digest(path)
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise PolicyBundleError("policy bundle digest mismatch")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise PolicyBundleError("weights-only policy bundle load failed") from error
    try:
        if not isinstance(raw, dict) or set(raw) != POLICY_BUNDLE_FIELDS:
            raise ValueError("policy bundle fields are invalid")
        if raw["schema"] != POLICY_BUNDLE_SCHEMA:
            raise ValueError("policy bundle schema is unsupported")
        metadata = _metadata_from_record(raw["metadata"])
        network = CheckersNetwork().cpu()
        model_state = _validated_model_state(raw["model_state"], network)
        network.load_state_dict(model_state, strict=True)
        network.eval()
        parameter_count = sum(parameter.numel() for parameter in network.parameters())
        if parameter_count < 1 or not math.isfinite(float(parameter_count)):
            raise ValueError("loaded policy has no parameters")
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise PolicyBundleError(f"invalid policy bundle: {error}") from error
    return LoadedPolicy(
        network=network,
        metadata=metadata,
        path=path,
        sha256=actual_digest,
        size_bytes=path.stat().st_size,
    )
