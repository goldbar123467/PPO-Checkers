"""Deterministic smoke tests for the Phase 4 environment fuzz gate."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from scripts.fuzz_environment import FuzzConfig, FuzzProgress, run_environment_fuzz

SMOKE_STEPS = 5_000


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FuzzConfig(steps=0), "steps"),
        (lambda: FuzzConfig(steps=True), "steps"),
        (lambda: FuzzConfig(seed=True), "seed"),
        (lambda: FuzzConfig(snapshot_interval=0), "snapshot_interval"),
        (lambda: FuzzConfig(progress_interval=-1), "progress_interval"),
    ],
)
def test_fuzz_config_rejects_invalid_budgets(
    factory: Callable[[], FuzzConfig],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_environment_fuzz_smoke_has_zero_disagreements_and_invariant_failures() -> None:
    result = run_environment_fuzz(
        FuzzConfig(
            steps=SMOKE_STEPS,
            seed=20260728,
            snapshot_interval=97,
            progress_interval=SMOKE_STEPS,
        )
    )

    assert result.steps_completed == SMOKE_STEPS
    assert result.invariant_violations == 0
    assert result.mask_disagreements == 0
    assert result.empty_nonterminal_masks == 0
    assert result.snapshot_roundtrips == SMOKE_STEPS // 97
    assert result.games_started > 1
    assert result.games_terminated > 0
    assert result.capture_steps > 0
    assert result.continuation_steps > 0
    assert result.fixture_starts["initial"] > 0
    assert sum(result.termination_reasons.values()) == result.games_terminated
    json.dumps(result.as_dict(), sort_keys=True)


def test_seeded_fuzz_counts_are_reproducible() -> None:
    config = FuzzConfig(
        steps=400,
        seed=17,
        snapshot_interval=101,
        progress_interval=400,
    )

    first = run_environment_fuzz(config)
    second = run_environment_fuzz(config)

    assert first.as_dict() == second.as_dict()


def test_progress_callback_reports_exact_intervals_and_final_step() -> None:
    progress: list[FuzzProgress] = []
    config = FuzzConfig(
        steps=23,
        seed=3,
        snapshot_interval=7,
        progress_interval=10,
    )

    run_environment_fuzz(config, progress.append)

    assert [item.steps_completed for item in progress] == [10, 20, 23]
    assert all(item.steps_completed <= config.steps for item in progress)
    assert all(item.games_started >= 1 for item in progress)
