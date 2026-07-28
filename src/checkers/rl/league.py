"""Bounded historical-policy pool and explicit Phase 8 opponent arms."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch

ARMS = frozenset({"A0", "A1", "A2", "A3"})
CURRENT_PROBABILITY = {"A0": 1.0, "A1": 0.8, "A2": 0.6, "A3": 0.0}
MIN_CAPACITY = 2


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _draw(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return checked


def _clone_state(model_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(model_state, Mapping):
        raise TypeError("model_state must be a mapping")
    cloned: dict[str, torch.Tensor] = {}
    for name, tensor in model_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model_state keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("model_state values must be Tensors")
        cloned[name] = tensor.detach().cpu().clone()
    if not cloned:
        raise ValueError("model_state must not be empty")
    return cloned


@dataclass(frozen=True, slots=True, init=False)
class LeagueSnapshot:
    """One immutable, CPU-cloned historical policy snapshot."""

    snapshot_id: str
    update_idx: int
    model_state: dict[str, torch.Tensor]
    pinned: bool

    def __init__(
        self,
        *,
        snapshot_id: str,
        update_idx: int,
        model_state: Mapping[str, torch.Tensor],
        pinned: bool,
    ) -> None:
        if not isinstance(snapshot_id, str):
            raise TypeError("snapshot_id must be a string")
        if not snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        if isinstance(update_idx, bool) or not isinstance(update_idx, int):
            raise TypeError("update_idx must be an integer")
        if update_idx < 0:
            raise ValueError("update_idx must be non-negative")
        if not isinstance(pinned, bool):
            raise TypeError("pinned must be bool")
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "update_idx", update_idx)
        object.__setattr__(self, "model_state", _clone_state(model_state))
        object.__setattr__(self, "pinned", pinned)

    def clone_model_state(self) -> dict[str, torch.Tensor]:
        """Return an independent CPU clone of every model tensor."""

        return _clone_state(self.model_state)


@dataclass(frozen=True, slots=True)
class OpponentSelection:
    """Current-policy or exact historical-snapshot opponent choice."""

    current: bool
    snapshot_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.current, bool):
            raise TypeError("current must be bool")
        if self.current != (self.snapshot_id is None):
            raise ValueError("current selection and snapshot_id disagree")


class LeaguePool:
    """Pinned-initial FIFO pool with explicit experimental selection arms."""

    def __init__(self, *, capacity: int) -> None:
        checked_capacity = _positive_integer(capacity, "capacity")
        if checked_capacity < MIN_CAPACITY:
            raise ValueError(f"capacity must be at least {MIN_CAPACITY}")
        self.capacity = checked_capacity
        self._snapshots: list[LeagueSnapshot] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def snapshot_ids(self) -> tuple[str, ...]:
        """Return stable pool order, oldest first and pinned initial first."""

        return tuple(snapshot.snapshot_id for snapshot in self._snapshots)

    @property
    def initial(self) -> LeagueSnapshot:
        """Return the required pinned random initialization.

        Raises:
            RuntimeError: If the initial snapshot has not been pinned.
        """

        if not self._snapshots:
            raise RuntimeError("initial snapshot has not been pinned")
        return self._snapshots[0]

    def pin_initial(self, model_state: Mapping[str, torch.Tensor]) -> LeagueSnapshot:
        """Clone and pin the initialization exactly once.

        Args:
            model_state: Network state at update zero.

        Returns:
            The newly pinned snapshot.

        Raises:
            RuntimeError: If an initial snapshot already exists.
        """

        if self._snapshots:
            raise RuntimeError("initial snapshot is already pinned")
        snapshot = LeagueSnapshot(
            snapshot_id="initial",
            update_idx=0,
            model_state=model_state,
            pinned=True,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def add_snapshot(
        self,
        *,
        update_idx: int,
        model_state: Mapping[str, torch.Tensor],
    ) -> LeagueSnapshot:
        """Add a cloned historical state and evict the oldest unpinned entry.

        Args:
            update_idx: Positive completed update index.
            model_state: Network tensors to clone onto CPU.

        Returns:
            Newly added snapshot.

        Raises:
            RuntimeError: If initialization is missing or ID would duplicate.
            TypeError: If inputs have invalid runtime types.
            ValueError: If the update index is not positive.
        """

        if not self._snapshots:
            raise RuntimeError("initial snapshot must be pinned before history")
        checked_update = _positive_integer(update_idx, "update_idx")
        snapshot_id = f"update-{checked_update}"
        if snapshot_id in self.snapshot_ids:
            raise RuntimeError("snapshot ID already exists")
        snapshot = LeagueSnapshot(
            snapshot_id=snapshot_id,
            update_idx=checked_update,
            model_state=model_state,
            pinned=False,
        )
        self._snapshots.append(snapshot)
        while len(self._snapshots) > self.capacity:
            del self._snapshots[1]
        return snapshot

    def clone_snapshot(self, snapshot_id: str) -> LeagueSnapshot:
        """Return an independent snapshot clone by ID.

        Args:
            snapshot_id: Stable pool identifier.

        Returns:
            Independent snapshot and tensor storage.

        Raises:
            ValueError: If no matching snapshot exists.
        """

        for snapshot in self._snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return LeagueSnapshot(
                    snapshot_id=snapshot.snapshot_id,
                    update_idx=snapshot.update_idx,
                    model_state=snapshot.model_state,
                    pinned=snapshot.pinned,
                )
        raise ValueError(f"unknown snapshot {snapshot_id!r}")

    def select(  # noqa: PLR0913
        self,
        *,
        arm: str,
        mixture_draw: float,
        historical_draw: float,
        historical_scores: Mapping[str, float] | None = None,
    ) -> OpponentSelection:
        """Select current or historical opponent from literal arm probabilities.

        Args:
            arm: One of A0 current-only, A1 80/20, A2 60/40, or A3 payoff-weighted history.
            mixture_draw: Uniform `[0,1)` draw for current-vs-history selection.
            historical_draw: Uniform `[0,1)` draw within history.
            historical_scores: A3 non-negative weights keyed by every snapshot ID.

        Returns:
            Explicit current-policy or snapshot selection.

        Raises:
            TypeError: If arguments have invalid runtime types.
            ValueError: If arm/draw/score records are invalid.
            RuntimeError: If no initial snapshot exists.
        """

        if not isinstance(arm, str):
            raise TypeError("arm must be a string")
        if arm not in ARMS:
            raise ValueError("arm must be one of A0, A1, A2, A3")
        checked_mixture = _draw(mixture_draw, "mixture_draw")
        checked_historical = _draw(historical_draw, "historical_draw")
        if not self._snapshots:
            raise RuntimeError("initial snapshot must be pinned before selection")
        if checked_mixture < CURRENT_PROBABILITY[arm]:
            return OpponentSelection(current=True, snapshot_id=None)
        if arm == "A3":
            return OpponentSelection(
                current=False,
                snapshot_id=self._weighted_snapshot(checked_historical, historical_scores),
            )
        index = min(int(checked_historical * len(self._snapshots)), len(self._snapshots) - 1)
        return OpponentSelection(current=False, snapshot_id=self._snapshots[index].snapshot_id)

    def _weighted_snapshot(
        self,
        draw: float,
        scores: Mapping[str, float] | None,
    ) -> str:
        if scores is None or set(scores) != set(self.snapshot_ids):
            raise ValueError("historical_scores must cover every snapshot ID")
        weights: list[float] = []
        for snapshot in self._snapshots:
            raw = scores[snapshot.snapshot_id]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("historical_scores values must be numeric")
            weight = float(raw)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("historical_scores values must be finite and non-negative")
            weights.append(weight)
        total = sum(weights)
        if total == 0.0:
            index = min(int(draw * len(self._snapshots)), len(self._snapshots) - 1)
            return self._snapshots[index].snapshot_id
        threshold = draw * total
        cumulative = 0.0
        for snapshot, weight in zip(self._snapshots[:-1], weights[:-1], strict=True):
            cumulative += weight
            if threshold < cumulative:
                return snapshot.snapshot_id
        return self._snapshots[-1].snapshot_id

    def to_record(self) -> dict[str, object]:
        """Return a trusted-checkpoint record with cloned CPU tensors."""

        return {
            "capacity": self.capacity,
            "snapshots": [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "update_idx": snapshot.update_idx,
                    "model_state": snapshot.clone_model_state(),
                    "pinned": snapshot.pinned,
                }
                for snapshot in self._snapshots
            ],
        }

    @classmethod
    def from_record(cls, value: object) -> LeaguePool:
        """Reconstruct and validate a trusted local checkpoint record.

        Args:
            value: Record returned by `to_record` after trusted deserialization.

        Returns:
            Independent validated pool.

        Raises:
            TypeError: If record fields have invalid runtime types.
            ValueError: If ordering, capacity, or pinning invariants fail.
        """

        if not isinstance(value, dict):
            raise TypeError("league record must be a mapping")
        record = cast(dict[str, object], value)
        if set(record) != {"capacity", "snapshots"}:
            raise ValueError("league record fields are invalid")
        pool = cls(capacity=cast(int, record["capacity"]))
        raw_snapshots = record["snapshots"]
        if not isinstance(raw_snapshots, list):
            raise TypeError("league snapshots must be a list")
        for index, raw in enumerate(raw_snapshots):
            if not isinstance(raw, dict):
                raise TypeError("league snapshot record must be a mapping")
            item = cast(dict[str, object], raw)
            if set(item) != {"snapshot_id", "update_idx", "model_state", "pinned"}:
                raise ValueError("league snapshot fields are invalid")
            snapshot = LeagueSnapshot(
                snapshot_id=cast(str, item["snapshot_id"]),
                update_idx=cast(int, item["update_idx"]),
                model_state=cast(Mapping[str, torch.Tensor], item["model_state"]),
                pinned=cast(bool, item["pinned"]),
            )
            if index == 0 and (snapshot.snapshot_id != "initial" or not snapshot.pinned):
                raise ValueError("first league snapshot must be pinned initial")
            if index > 0 and snapshot.pinned:
                raise ValueError("only initial league snapshot may be pinned")
            pool._snapshots.append(snapshot)
        if len(pool._snapshots) > pool.capacity:
            raise ValueError("league record exceeds capacity")
        return pool
