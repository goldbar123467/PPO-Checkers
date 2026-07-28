"""Deterministic differential-runner tests for the Phase 2 large gate."""

from __future__ import annotations

import hashlib

import pytest

from checkers.rules.differential import (
    DifferentialConfig,
    DifferentialMismatchError,
    DifferentialResult,
    _digest_state,
    run_differential,
)
from checkers.rules.moves import Step
from checkers.rules.state import PlayerId, State

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


def test_differential_runner_has_a_pinned_small_regression_result() -> None:
    result = run_differential(
        DifferentialConfig(
            positions=20,
            seed=7,
            max_plies=4,
            bfs_depth=0,
            digest_interval=3,
        )
    )

    assert result == DifferentialResult(
        playout_positions=20,
        bfs_positions=1,
        unique_bfs_states=8,
        steps_applied=16,
        games_started=5,
        capture_steps=1,
        continuation_steps=0,
        max_pending_captures=0,
        digest_samples=7,
        state_digest_sha256="ed84ebb9658dc3821766cde52f74a0b38284fb8d98670184055d9a32c5ca1d0f",
    )


def test_state_digest_separates_each_asymmetric_bitboard_and_counter() -> None:
    state = State(
        men=(1 << 0, 1 << 1),
        kings=(1 << 2, 1 << 3),
        side_to_move=PlayerId.RED,
        capture_in_progress=True,
        moving_square=0,
        sequence_origin=4,
        captured_pending=1 << 1,
        no_progress=(7, 9),
        ply=11,
    )
    digest = hashlib.sha256()

    _digest_state(digest, state)

    assert digest.hexdigest() == "dad5c16873a3dcc0f62610614d0b5de13de3a0577a2af7cf90bdc9395b5e35d7"


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
    assert captured.value.state == State.initial()
    assert captured.value.fast
    assert captured.value.oracle == ()
