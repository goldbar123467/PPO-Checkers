"""Deterministic differential-runner tests for the Phase 2 large gate."""

from __future__ import annotations

import pytest

from checkers.rules.differential import (
    DifferentialConfig,
    DifferentialMismatchError,
    run_differential,
)
from checkers.rules.moves import Step
from checkers.rules.state import State

SMALL_RUN_POSITIONS = 1_000
SHA256_HEX_LENGTH = 64


def test_differential_runner_is_deterministic_and_nonvacuous() -> None:
    config = DifferentialConfig(
        positions=SMALL_RUN_POSITIONS,
        seed=20260727,
        max_plies=128,
        bfs_depth=3,
        digest_interval=17,
    )

    first = run_differential(config)
    second = run_differential(config)

    assert first == second
    assert first.playout_positions == SMALL_RUN_POSITIONS
    assert first.bfs_positions > 1
    assert first.unique_bfs_states >= first.bfs_positions
    assert first.steps_applied > 0
    assert first.games_started > 1
    assert first.digest_samples > 0
    assert len(first.state_digest_sha256) == SHA256_HEX_LENGTH
    assert first.disagreements == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"positions": 0}, "positions"),
        ({"seed": -1}, "seed"),
        ({"max_plies": 0}, "max_plies"),
        ({"bfs_depth": -1}, "bfs_depth"),
        ({"digest_interval": 0}, "digest_interval"),
    ],
)
def test_differential_config_rejects_invalid_budgets(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DifferentialConfig(**kwargs)


def test_differential_runner_halts_on_first_disagreement() -> None:
    def empty_oracle(_state: State) -> tuple[Step, ...]:
        return ()

    with pytest.raises(DifferentialMismatchError, match="bfs") as captured:
        run_differential(
            DifferentialConfig(positions=1, bfs_depth=0),
            oracle_generator=empty_oracle,
        )

    assert captured.value.stage == "bfs"
    assert captured.value.index == 0
    assert captured.value.fast
    assert not captured.value.oracle
