"""Shared local policy bundle fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkers.rl.networks import CheckersNetwork
from checkers.web.policy_bundle import (
    LoadedPolicy,
    PolicyBundleMetadata,
    load_policy_bundle,
    save_policy_bundle,
)


@pytest.fixture
def policy_metadata() -> PolicyBundleMetadata:
    """Return valid deterministic test provenance."""

    return PolicyBundleMetadata(
        bundle_id="web-test-update-000001",
        experiment_id="web-test",
        update_idx=1,
        global_step=8192,
        source_checkpoint="runs/test/checkpoint.pt",
        source_checkpoint_sha256="a" * 64,
        source_checkpoint_size_bytes=1024,
        source_git_sha="0123456789abcdef",
        source_git_dirty=False,
        config_sha256="b" * 64,
        max_plies=512,
        repetition_draws=True,
    )


@pytest.fixture
def loaded_policy(tmp_path: Path, policy_metadata: PolicyBundleMetadata) -> LoadedPolicy:
    """Write and strictly reload a tiny inference bundle."""

    path = tmp_path / "policy.pt"
    save_policy_bundle(
        path=path,
        network=CheckersNetwork(),
        metadata=policy_metadata,
    )
    return load_policy_bundle(path)
