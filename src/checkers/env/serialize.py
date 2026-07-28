"""Canonical, validated snapshots for resumable checkers environments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from checkers.rules.board import PLAYABLE_SQUARES
from checkers.rules.notation import parse_state, serialize_state
from checkers.rules.state import State
from checkers.rules.zobrist import UINT64_MASK, position_key

ENVIRONMENT_SCHEMA = "CHECKERS_ENV_1"
SNAPSHOT_FIELDS = frozenset(
    {
        "schema",
        "state",
        "initial_state",
        "max_plies",
        "repetition_draws",
        "position_counts",
        "active_move_squares",
    }
)
POSITION_COUNT_FIELDS = frozenset({"key", "count"})
POSITION_KEY_PATTERN = re.compile(r"[0-9a-f]{16}")
MINIMUM_ACTIVE_PATH = 2
POSITION_COUNT_PAIR_SIZE = 2


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_square(square: object) -> int:
    if isinstance(square, bool) or not isinstance(square, int):
        raise TypeError("active move squares must be integers")
    if not 0 <= square < PLAYABLE_SQUARES:
        raise ValueError("active move square must be in [0, 31]")
    return square


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """All dynamic state needed to resume an environment exactly.

    Args:
        state: Current complete rules state.
        initial_state: Boundary state restored by a future environment reset.
        max_plies: Immutable R6.5 rule configuration.
        repetition_draws: Whether the optional R6.4 arena rule is enabled.
        position_counts: Sorted official-position visit counts.
        active_move_squares: Partial ACF capture path, empty at move boundaries.

    Raises:
        TypeError: If a field has the wrong runtime type.
        ValueError: If fields are noncanonical or inconsistent with the state.
    """

    state: State
    initial_state: State
    max_plies: int
    repetition_draws: bool
    position_counts: tuple[tuple[int, int], ...]
    active_move_squares: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise TypeError("snapshot state must be a State")
        if not isinstance(self.initial_state, State):
            raise TypeError("snapshot initial_state must be a State")
        if self.initial_state.capture_in_progress:
            raise ValueError("snapshot initial_state must be a move boundary")
        _require_positive_integer(self.max_plies, "max_plies")
        if not isinstance(self.repetition_draws, bool):
            raise TypeError("repetition_draws must be bool")
        self._validate_position_counts()
        self._validate_active_move()

    def _validate_position_counts(self) -> None:
        if not isinstance(self.position_counts, tuple):
            raise TypeError("position_counts must be a tuple")
        if not self.position_counts:
            raise ValueError("position_counts must contain at least one position")
        prior_key = -1
        for entry in self.position_counts:
            if not isinstance(entry, tuple) or len(entry) != POSITION_COUNT_PAIR_SIZE:
                raise TypeError("each position count must be a key/count pair")
            key, count = entry
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError("position key must be an integer")
            if not 0 <= key <= UINT64_MASK:
                raise ValueError("position key must fit uint64")
            _require_positive_integer(count, "position count")
            if key <= prior_key:
                raise ValueError("position_counts must have unique keys in sorted order")
            prior_key = key
        if not self.state.capture_in_progress:
            current_key = position_key(self.state)
            if current_key not in dict(self.position_counts):
                raise ValueError("position_counts must contain the current position")

    def _validate_active_move(self) -> None:
        if not isinstance(self.active_move_squares, tuple):
            raise TypeError("active move squares must be a tuple")
        for square in self.active_move_squares:
            _validate_square(square)
        if not self.state.capture_in_progress:
            if self.active_move_squares:
                raise ValueError("active move must be empty at a move boundary")
            return
        if len(self.active_move_squares) < MINIMUM_ACTIVE_PATH:
            raise ValueError("active move must include an origin and landing")
        if self.active_move_squares[0] != self.state.sequence_origin:
            raise ValueError("active move origin must match sequence_origin")
        if self.active_move_squares[-1] != self.state.moving_square:
            raise ValueError("active move landing must match moving_square")


def serialize_environment_snapshot(snapshot: EnvironmentSnapshot) -> str:
    """Serialize a validated snapshot as deterministic compact JSON.

    Args:
        snapshot: Complete validated environment snapshot.

    Returns:
        Canonical JSON using a versioned schema and fixed-width hexadecimal keys.

    Raises:
        TypeError: If ``snapshot`` is not an ``EnvironmentSnapshot``.
    """

    if not isinstance(snapshot, EnvironmentSnapshot):
        raise TypeError("snapshot must be an EnvironmentSnapshot")
    payload = {
        "active_move_squares": list(snapshot.active_move_squares),
        "initial_state": serialize_state(snapshot.initial_state),
        "max_plies": snapshot.max_plies,
        "position_counts": [
            {"count": count, "key": f"{key:016x}"} for key, count in snapshot.position_counts
        ],
        "repetition_draws": snapshot.repetition_draws,
        "schema": ENVIRONMENT_SCHEMA,
        "state": serialize_state(snapshot.state),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_position_counts(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise TypeError("position_counts must be a list")
    entries: list[tuple[int, int]] = []
    for raw_entry in value:
        if not isinstance(raw_entry, dict) or set(raw_entry) != POSITION_COUNT_FIELDS:
            raise ValueError("position count fields are invalid")
        key_text = raw_entry["key"]
        if not isinstance(key_text, str) or POSITION_KEY_PATTERN.fullmatch(key_text) is None:
            raise ValueError("position key must be 16 lowercase hexadecimal digits")
        count = _require_positive_integer(raw_entry["count"], "position count")
        entries.append((int(key_text, 16), count))
    return tuple(entries)


def _parse_active_move(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("active_move_squares must be a list")
    return tuple(_validate_square(square) for square in value)


def parse_environment_snapshot(text: str) -> EnvironmentSnapshot:
    """Parse and validate a canonical environment snapshot.

    Args:
        text: JSON produced by ``serialize_environment_snapshot``.

    Returns:
        A fully validated immutable snapshot.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If JSON syntax, schema, fields, or cross-field invariants are invalid.
    """

    if not isinstance(text, str):
        raise TypeError("environment snapshot must be text")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid environment snapshot JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("environment snapshot must be a JSON object")
    payload = cast(dict[str, Any], raw)
    if set(payload) != SNAPSHOT_FIELDS:
        raise ValueError("environment snapshot fields are invalid")
    if payload["schema"] != ENVIRONMENT_SCHEMA:
        raise ValueError("environment snapshot schema is unsupported")
    try:
        return EnvironmentSnapshot(
            state=parse_state(payload["state"]),
            initial_state=parse_state(payload["initial_state"]),
            max_plies=_require_positive_integer(payload["max_plies"], "max_plies"),
            repetition_draws=payload["repetition_draws"],
            position_counts=_parse_position_counts(payload["position_counts"]),
            active_move_squares=_parse_active_move(payload["active_move_squares"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid environment snapshot: {error}") from error
