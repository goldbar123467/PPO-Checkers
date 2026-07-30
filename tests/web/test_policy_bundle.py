"""Inference bundle integrity, schema, and reload tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from checkers.config import RunConfig
from checkers.rl.networks import CheckersNetwork
from checkers.web.policy_bundle import (
    PolicyBundleError,
    PolicyBundleMetadata,
    config_sha256,
    load_policy_bundle,
    save_policy_bundle,
    sha256_file,
)


def _write_raw_bundle(path: Path, value: object) -> None:
    torch.save(value, path)
    path.with_suffix(".pt.sha256").write_text(f"{sha256_file(path)}\n", encoding="ascii")


def _valid_raw_bundle(policy_metadata: PolicyBundleMetadata) -> dict[str, object]:
    return {
        "schema": "CHECKERS_POLICY_BUNDLE_1",
        "metadata": {
            **dataclasses.asdict(policy_metadata),
            "observation_shape": list(policy_metadata.observation_shape),
        },
        "model_state": {
            name: tensor.detach().clone() for name, tensor in CheckersNetwork().state_dict().items()
        },
    }


def test_round_trip_preserves_exact_weights_and_metadata(
    tmp_path: Path, policy_metadata: PolicyBundleMetadata
) -> None:
    """A saved policy must strictly reload with equal tensors and a matching sidecar."""

    network = CheckersNetwork()
    path = tmp_path / "bundle.pt"
    digest, size_bytes = save_policy_bundle(
        path=path,
        network=network,
        metadata=policy_metadata,
    )
    loaded = load_policy_bundle(path)

    assert loaded.metadata == policy_metadata
    assert loaded.sha256 == digest == sha256_file(path)
    assert loaded.size_bytes == size_bytes == path.stat().st_size
    assert path.with_suffix(".pt.sha256").read_text(encoding="ascii").strip() == digest
    assert not loaded.network.training
    for name, tensor in network.state_dict().items():
        assert torch.equal(loaded.network.state_dict()[name], tensor)


def test_tampering_and_invalid_metadata_are_rejected(
    tmp_path: Path, policy_metadata: PolicyBundleMetadata
) -> None:
    """Both content tampering and malformed provenance must fail closed."""

    path = tmp_path / "bundle.pt"
    save_policy_bundle(path=path, network=CheckersNetwork(), metadata=policy_metadata)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(PolicyBundleError, match="digest mismatch"):
        load_policy_bundle(path)

    with pytest.raises(ValueError, match="SHA-256"):
        dataclasses.replace(policy_metadata, config_sha256="invalid")


def test_missing_bundle_and_digest_fail_closed(tmp_path: Path) -> None:
    """Runtime startup must not continue without both bundle and checksum."""

    with pytest.raises(PolicyBundleError, match="does not exist"):
        load_policy_bundle(tmp_path / "missing.pt")

    path = tmp_path / "bundle.pt"
    torch.save({}, path)
    with pytest.raises(PolicyBundleError, match="digest file is missing"):
        load_policy_bundle(path)


def test_run_config_hash_is_stable_and_sensitive() -> None:
    """Canonical config hashing must be deterministic and change with run variables."""

    config = RunConfig(device="cpu", periodic_games=2, eval_games=2)
    assert config_sha256(config) == config_sha256(config)
    changed = dataclasses.replace(config, max_plies=config.max_plies + 1)
    assert config_sha256(changed) != config_sha256(config)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("bundle_id", "", "text fields"),
        ("update_idx", True, "integer"),
        ("global_step", -1, "non-negative"),
        ("source_checkpoint_size_bytes", 0, "positive"),
        ("max_plies", 0, "positive"),
        ("source_git_dirty", "no", "must be bool"),
        ("repetition_draws", "yes", "must be bool"),
        ("network_class", "other.Network", "network class"),
        ("observation_shape", (1, 2, 3), "observation shape"),
        ("action_count", 64, "action count"),
    ],
)
def test_metadata_rejects_invalid_runtime_contracts(
    policy_metadata: PolicyBundleMetadata, field: str, value: object, match: str
) -> None:
    """Every metadata boundary must fail before an invalid network is served."""

    fields = dataclasses.asdict(policy_metadata)
    fields[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        PolicyBundleMetadata(**cast(Any, fields))


def test_public_helpers_reject_wrong_object_types(
    tmp_path: Path, policy_metadata: PolicyBundleMetadata
) -> None:
    """Public artifact helpers must reject duck-typed paths, configs, models, and metadata."""

    with pytest.raises(TypeError, match="path must be"):
        sha256_file(cast(Any, "bundle.pt"))
    with pytest.raises(TypeError, match="config must be"):
        config_sha256(cast(Any, {}))
    with pytest.raises(TypeError, match="path must be"):
        save_policy_bundle(
            path=cast(Any, "bundle.pt"),
            network=CheckersNetwork(),
            metadata=policy_metadata,
        )
    with pytest.raises(TypeError, match="network must be"):
        save_policy_bundle(
            path=tmp_path / "bundle.pt",
            network=cast(Any, object()),
            metadata=policy_metadata,
        )
    with pytest.raises(TypeError, match="metadata must be"):
        save_policy_bundle(
            path=tmp_path / "bundle.pt",
            network=CheckersNetwork(),
            metadata=cast(Any, object()),
        )
    with pytest.raises(TypeError, match="path must be"):
        load_policy_bundle(cast(Any, "bundle.pt"))


def test_malformed_digest_and_unloadable_payload_fail_closed(tmp_path: Path) -> None:
    """A syntactically bad sidecar and a checksummed non-Torch payload must both stop startup."""

    path = tmp_path / "bundle.pt"
    path.write_bytes(b"not a torch archive")
    path.with_suffix(".pt.sha256").write_text("BAD\n", encoding="ascii")
    with pytest.raises(PolicyBundleError, match="lowercase SHA-256"):
        load_policy_bundle(path)

    path.with_suffix(".pt.sha256").write_text(f"{sha256_file(path)}\n", encoding="ascii")
    with pytest.raises(PolicyBundleError, match="weights-only"):
        load_policy_bundle(path)


def test_bundle_schema_and_tensor_corruption_fail_closed(
    tmp_path: Path, policy_metadata: PolicyBundleMetadata
) -> None:
    """Closed-schema and tensor checks must reject representative corruptions."""

    cases: list[tuple[str, object, str]] = []
    cases.append(("not-object", [], "fields are invalid"))

    bad_fields = _valid_raw_bundle(policy_metadata)
    bad_fields["extra"] = True
    cases.append(("fields", bad_fields, "fields are invalid"))

    bad_schema = _valid_raw_bundle(policy_metadata)
    bad_schema["schema"] = "OTHER"
    cases.append(("schema", bad_schema, "schema is unsupported"))

    bad_metadata = _valid_raw_bundle(policy_metadata)
    bad_metadata["metadata"] = {}
    cases.append(("metadata", bad_metadata, "metadata fields are invalid"))

    bad_shape = _valid_raw_bundle(policy_metadata)
    shape_metadata = cast(dict[str, object], bad_shape["metadata"])
    shape_metadata["observation_shape"] = [8, "8", 8]
    cases.append(("shape", bad_shape, "list of integers"))

    bad_mapping = _valid_raw_bundle(policy_metadata)
    bad_mapping["model_state"] = []
    cases.append(("mapping", bad_mapping, "must be a mapping"))

    bad_value = _valid_raw_bundle(policy_metadata)
    bad_value["model_state"] = {"stem.0.weight": "not a tensor"}
    cases.append(("value", bad_value, "map strings to tensors"))

    bad_keys = _valid_raw_bundle(policy_metadata)
    key_state = cast(dict[str, torch.Tensor], bad_keys["model_state"])
    key_state.pop(next(iter(key_state)))
    cases.append(("keys", bad_keys, "keys do not match"))

    bad_tensor_shape = _valid_raw_bundle(policy_metadata)
    shape_state = cast(dict[str, torch.Tensor], bad_tensor_shape["model_state"])
    first_name = next(iter(shape_state))
    shape_state[first_name] = torch.zeros(1)
    cases.append(("tensor-shape", bad_tensor_shape, "tensor metadata mismatch"))

    bad_finite = _valid_raw_bundle(policy_metadata)
    finite_state = cast(dict[str, torch.Tensor], bad_finite["model_state"])
    finite_name = next(name for name, tensor in finite_state.items() if tensor.is_floating_point())
    finite_state[finite_name].view(-1)[0] = torch.nan
    cases.append(("finite", bad_finite, "non-finite"))

    for name, value, match in cases:
        path = tmp_path / f"{name}.pt"
        _write_raw_bundle(path, value)
        with pytest.raises(PolicyBundleError, match=match):
            load_policy_bundle(path)
