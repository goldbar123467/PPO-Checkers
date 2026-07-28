"""Phase 7 historical snapshot-pool and opponent-selection tests."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import cast

import pytest
import torch

from checkers.rl.league import LeaguePool, LeagueSnapshot, OpponentSelection

FULL_POOL_SIZE = 4
UPDATE_FOUR_WEIGHT = 4.0


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([value]), "counter": torch.tensor(7)}


def test_l1_initial_snapshot_is_pinned_and_fifo_evicts_only_unpinned_history() -> None:
    pool = LeaguePool(capacity=4)
    initial_source = _state(0.0)
    pool.pin_initial(initial_source)
    initial_source["weight"].fill_(99.0)
    for update in range(1, 6):
        pool.add_snapshot(update_idx=update, model_state=_state(float(update)))

    assert pool.snapshot_ids == ("initial", "update-3", "update-4", "update-5")
    assert pool.initial.model_state["weight"].item() == 0.0
    assert pool.initial.pinned
    assert len(pool) == FULL_POOL_SIZE


def test_l1_snapshot_state_is_cloned_on_input_and_output() -> None:
    pool = LeaguePool(capacity=2)
    pool.pin_initial(_state(0.0))
    source = _state(1.0)
    snapshot = pool.add_snapshot(update_idx=1, model_state=source)
    source["weight"].fill_(8.0)
    exported = snapshot.clone_model_state()
    exported["weight"].fill_(9.0)

    assert snapshot.model_state["weight"].item() == 1.0


@pytest.mark.parametrize(
    ("arm", "draw", "expected_current"),
    [
        ("A0", 0.999, True),
        ("A1", 0.799, True),
        ("A1", 0.800, False),
        ("A2", 0.599, True),
        ("A2", 0.600, False),
    ],
)
def test_l1_current_historical_thresholds_are_literal(
    arm: str,
    draw: float,
    expected_current: bool,
) -> None:
    pool = LeaguePool(capacity=3)
    pool.pin_initial(_state(0.0))

    selection = pool.select(arm=arm, mixture_draw=draw, historical_draw=0.0)

    assert isinstance(selection, OpponentSelection)
    assert selection.current is expected_current
    assert (selection.snapshot_id is None) is expected_current


def test_l1_uniform_historical_selection_uses_exact_draw_bucket() -> None:
    pool = LeaguePool(capacity=4)
    pool.pin_initial(_state(0.0))
    pool.add_snapshot(update_idx=1, model_state=_state(1.0))
    pool.add_snapshot(update_idx=2, model_state=_state(2.0))

    selections = tuple(
        pool.select(arm="A2", mixture_draw=0.9, historical_draw=draw).snapshot_id
        for draw in (0.0, 0.34, 0.67, 0.999)
    )

    assert selections == ("initial", "update-1", "update-2", "update-2")


def test_l1_payoff_weighted_selection_prefers_snapshots_that_beat_current() -> None:
    pool = LeaguePool(capacity=4)
    pool.pin_initial(_state(0.0))
    pool.add_snapshot(update_idx=1, model_state=_state(1.0))
    pool.add_snapshot(update_idx=2, model_state=_state(2.0))
    payoffs = {"initial": 0.1, "update-1": 0.8, "update-2": 0.1}

    middle = pool.select(
        arm="A3",
        mixture_draw=0.0,
        historical_draw=0.5,
        historical_scores=payoffs,
    )

    assert middle.snapshot_id == "update-1"
    with pytest.raises(ValueError, match="historical_scores"):
        pool.select(arm="A3", mixture_draw=0.0, historical_draw=0.5)


def test_l1_pool_record_round_trip_is_exact_and_independent() -> None:
    pool = LeaguePool(capacity=3)
    pool.pin_initial(_state(0.0))
    pool.add_snapshot(update_idx=4, model_state=_state(4.0))

    restored = LeaguePool.from_record(pool.to_record())
    restored.add_snapshot(update_idx=5, model_state=_state(5.0))

    assert restored.snapshot_ids == ("initial", "update-4", "update-5")
    assert pool.snapshot_ids == ("initial", "update-4")
    assert restored.clone_snapshot("update-4").model_state["weight"].item() == UPDATE_FOUR_WEIGHT


@pytest.mark.parametrize(
    ("operation", "error", "message"),
    [
        (lambda: LeaguePool(capacity=1), ValueError, "capacity"),
        (lambda: LeaguePool(capacity=True), TypeError, "capacity"),
        (
            lambda: LeaguePool(capacity=2).add_snapshot(update_idx=1, model_state=_state(1.0)),
            RuntimeError,
            "initial",
        ),
        (
            lambda: LeaguePool(capacity=2).select(arm="A4", mixture_draw=0.0, historical_draw=0.0),
            ValueError,
            "arm",
        ),
    ],
)
def test_l1_invalid_pool_operations_raise(
    operation: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        operation()


def test_l1_seeded_selection_replays_after_python_rng_restore() -> None:
    pool = LeaguePool(capacity=3)
    pool.pin_initial(_state(0.0))
    pool.add_snapshot(update_idx=1, model_state=_state(1.0))
    rng = random.Random(2026)
    state = rng.getstate()
    first = pool.select(arm="A2", mixture_draw=rng.random(), historical_draw=rng.random())
    rng.setstate(state)
    second = pool.select(arm="A2", mixture_draw=rng.random(), historical_draw=rng.random())

    assert first == second


def test_l1_public_validators_reject_malformed_snapshots_and_model_states() -> None:
    with pytest.raises(TypeError, match="mapping"):
        LeaguePool(capacity=2).pin_initial(cast(dict[str, torch.Tensor], object()))
    with pytest.raises(ValueError, match="keys"):
        LeaguePool(capacity=2).pin_initial(cast(dict[str, torch.Tensor], {1: torch.ones(1)}))
    with pytest.raises(TypeError, match="values"):
        LeaguePool(capacity=2).pin_initial(cast(dict[str, torch.Tensor], {"w": 1}))
    with pytest.raises(ValueError, match="empty"):
        LeaguePool(capacity=2).pin_initial({})

    base = {"snapshot_id": "x", "update_idx": 0, "model_state": _state(0.0), "pinned": False}
    for field, value, error, message in (
        ("snapshot_id", 1, TypeError, "snapshot_id"),
        ("snapshot_id", "", ValueError, "snapshot_id"),
        ("update_idx", True, TypeError, "update_idx"),
        ("update_idx", -1, ValueError, "update_idx"),
        ("pinned", 1, TypeError, "pinned"),
    ):
        arguments = dict(base)
        arguments[field] = value
        with pytest.raises(error, match=message):
            LeagueSnapshot(**arguments)  # type: ignore[arg-type]


def test_l1_selection_and_pool_state_invalid_boundaries_raise() -> None:
    pool = LeaguePool(capacity=2)
    with pytest.raises(RuntimeError, match="initial"):
        _ = pool.initial
    with pytest.raises(RuntimeError, match="initial"):
        pool.select(arm="A0", mixture_draw=0.0, historical_draw=0.0)
    pool.pin_initial(_state(0.0))
    with pytest.raises(RuntimeError, match="already"):
        pool.pin_initial(_state(0.0))
    pool.add_snapshot(update_idx=1, model_state=_state(1.0))
    with pytest.raises(ValueError, match="update_idx"):
        pool.add_snapshot(update_idx=0, model_state=_state(1.0))
    with pytest.raises(RuntimeError, match="already"):
        pool.add_snapshot(update_idx=1, model_state=_state(1.0))
    with pytest.raises(ValueError, match="unknown"):
        pool.clone_snapshot("absent")
    with pytest.raises(TypeError, match="arm"):
        pool.select(arm=1, mixture_draw=0.0, historical_draw=0.0)  # type: ignore[arg-type]
    for field, value, error in (
        ("mixture_draw", "x", TypeError),
        ("mixture_draw", float("nan"), ValueError),
        ("historical_draw", 1.0, ValueError),
    ):
        arguments: dict[str, object] = {
            "arm": "A2",
            "mixture_draw": 0.0,
            "historical_draw": 0.0,
        }
        arguments[field] = value
        with pytest.raises(error, match=field):
            pool.select(**arguments)  # type: ignore[arg-type]


def test_l1_payoff_weights_validate_values_and_zero_total_falls_back_to_uniform() -> None:
    pool = LeaguePool(capacity=3)
    pool.pin_initial(_state(0.0))
    pool.add_snapshot(update_idx=1, model_state=_state(1.0))
    zero = pool.select(
        arm="A3",
        mixture_draw=0.0,
        historical_draw=0.75,
        historical_scores={"initial": 0.0, "update-1": 0.0},
    )
    assert zero.snapshot_id == "update-1"
    last = pool.select(
        arm="A3",
        mixture_draw=0.0,
        historical_draw=0.99,
        historical_scores={"initial": 0.1, "update-1": 0.9},
    )
    assert last.snapshot_id == "update-1"
    with pytest.raises(TypeError, match="numeric"):
        pool.select(
            arm="A3",
            mixture_draw=0.0,
            historical_draw=0.0,
            historical_scores={"initial": 0.0, "update-1": True},
        )
    with pytest.raises(ValueError, match="non-negative"):
        pool.select(
            arm="A3",
            mixture_draw=0.0,
            historical_draw=0.0,
            historical_scores={"initial": 0.0, "update-1": -1.0},
        )


def test_l1_opponent_selection_validates_discriminated_union() -> None:
    with pytest.raises(TypeError, match="current"):
        OpponentSelection(current=1, snapshot_id=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="disagree"):
        OpponentSelection(current=True, snapshot_id="initial")


@pytest.mark.parametrize(
    ("record", "error", "message"),
    [
        ([], TypeError, "mapping"),
        ({"capacity": 2}, ValueError, "fields"),
        ({"capacity": 2, "snapshots": {}}, TypeError, "list"),
        ({"capacity": 2, "snapshots": [1]}, TypeError, "mapping"),
        (
            {"capacity": 2, "snapshots": [{"snapshot_id": "initial"}]},
            ValueError,
            "fields",
        ),
        (
            {
                "capacity": 2,
                "snapshots": [
                    {
                        "snapshot_id": "wrong",
                        "update_idx": 0,
                        "model_state": _state(0.0),
                        "pinned": False,
                    }
                ],
            },
            ValueError,
            "pinned initial",
        ),
        (
            {
                "capacity": 2,
                "snapshots": [
                    {
                        "snapshot_id": "initial",
                        "update_idx": 0,
                        "model_state": _state(0.0),
                        "pinned": True,
                    },
                    {
                        "snapshot_id": "update-1",
                        "update_idx": 1,
                        "model_state": _state(1.0),
                        "pinned": True,
                    },
                ],
            },
            ValueError,
            "only initial",
        ),
        (
            {
                "capacity": 2,
                "snapshots": [
                    {
                        "snapshot_id": "initial",
                        "update_idx": 0,
                        "model_state": _state(0.0),
                        "pinned": True,
                    },
                    {
                        "snapshot_id": "update-1",
                        "update_idx": 1,
                        "model_state": _state(1.0),
                        "pinned": False,
                    },
                    {
                        "snapshot_id": "update-2",
                        "update_idx": 2,
                        "model_state": _state(2.0),
                        "pinned": False,
                    },
                ],
            },
            ValueError,
            "exceeds",
        ),
    ],
)
def test_l1_malformed_pool_records_raise(
    record: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        LeaguePool.from_record(record)
