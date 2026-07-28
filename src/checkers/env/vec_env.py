"""Synchronous, transactional vectorization for checkers environments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from checkers.env.checkers_env import CheckersEnv, IllegalActionError
from checkers.env.encoding import DEFAULT_MAX_PLIES, Float32Array
from checkers.env.masking import action_to_step
from checkers.env.serialize import (
    parse_environment_snapshot,
    serialize_environment_snapshot,
)
from checkers.rules.state import State

VECTOR_ENVIRONMENT_SCHEMA = "CHECKERS_VECTOR_ENV_1"
VECTOR_SNAPSHOT_FIELDS = frozenset({"schema", "environments"})

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]
UInt64Array = NDArray[np.uint64]
IntegerArray = NDArray[np.integer[Any]]
VectorInfos = tuple[dict[str, Any], ...]
VectorStepResult = tuple[Float32Array, FloatArray, BoolArray, BoolArray, VectorInfos]


def _validate_num_envs(num_envs: int) -> int:
    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError("num_envs must be an integer")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return num_envs


def _build_environments(
    num_envs: int,
    *,
    max_plies: int,
    repetition_draws: bool,
    initial_states: tuple[State, ...] | None,
) -> tuple[CheckersEnv, ...]:
    if initial_states is not None:
        if not isinstance(initial_states, tuple):
            raise TypeError("initial_states must be a tuple or None")
        if len(initial_states) != num_envs:
            raise ValueError("initial_states must contain exactly num_envs states")
        if not all(isinstance(state, State) for state in initial_states):
            raise TypeError("initial_states must contain only State values")
    states: tuple[State | None, ...] = (
        (None,) * num_envs if initial_states is None else initial_states
    )
    return tuple(
        CheckersEnv(
            max_plies=max_plies,
            repetition_draws=repetition_draws,
            initial_state=state,
        )
        for state in states
    )


def _canonical_environment_records(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("vector environment snapshot must be text")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid vector environment snapshot JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("vector environment snapshot must be a JSON object")
    payload = cast(dict[str, Any], raw)
    if set(payload) != VECTOR_SNAPSHOT_FIELDS:
        raise ValueError("vector environment snapshot fields are invalid")
    if payload["schema"] != VECTOR_ENVIRONMENT_SCHEMA:
        raise ValueError("vector environment snapshot schema is unsupported")
    records = payload["environments"]
    if not isinstance(records, list):
        raise ValueError("vector snapshot environments must be a list")
    if not records:
        raise ValueError("vector snapshot must contain at least one environment")

    canonical: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"invalid vector snapshot lane {index}: expected an object")
        nested_text = json.dumps(record, sort_keys=True, separators=(",", ":"))
        try:
            snapshot = parse_environment_snapshot(nested_text)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid vector snapshot lane {index}: {error}") from error
        canonical.append(serialize_environment_snapshot(snapshot))
    return tuple(canonical)


class CheckersVectorEnv:
    """Advance independent checkers games in synchronous environment-step lockstep."""

    def __init__(
        self,
        num_envs: int,
        *,
        max_plies: int = DEFAULT_MAX_PLIES,
        repetition_draws: bool = False,
        initial_states: tuple[State, ...] | None = None,
    ) -> None:
        """Construct a fixed number of independently resettable environments.

        Args:
            num_envs: Positive number of synchronous lanes.
            max_plies: Shared R6.5 limit for newly constructed lanes.
            repetition_draws: Shared optional R6.4 setting for newly constructed lanes.
            initial_states: Optional exact reset state for each lane.

        Raises:
            TypeError: If counts, booleans, or state values have invalid runtime types.
            ValueError: If counts or the initial-state collection are invalid.
        """

        checked_num_envs = _validate_num_envs(num_envs)
        self._envs = _build_environments(
            checked_num_envs,
            max_plies=max_plies,
            repetition_draws=repetition_draws,
            initial_states=initial_states,
        )

    @property
    def num_envs(self) -> int:
        """Return the immutable number of vector lanes."""

        return len(self._envs)

    @property
    def envs(self) -> tuple[CheckersEnv, ...]:
        """Return the lane environments as an immutable tuple."""

        return self._envs

    def observations(self) -> Float32Array:
        """Return freshly encoded observations for every lane.

        Returns:
            Float32 array shaped ``(num_envs, 8, 8, 8)``.
        """

        return np.stack([environment.observe() for environment in self._envs])

    def legal_masks(self) -> BoolArray:
        """Return current fixed-width legal masks for every lane.

        Returns:
            Boolean array shaped ``(num_envs, 128)``.
        """

        return np.stack([environment.legal_mask() for environment in self._envs])

    def state_keys(self) -> UInt64Array:
        """Return complete-state keys for every lane.

        Returns:
            Unsigned 64-bit array shaped ``(num_envs,)``.
        """

        return np.asarray(
            [environment.state_key() for environment in self._envs],
            dtype=np.uint64,
        )

    def reset(self, *, seed: int | None = None) -> tuple[Float32Array, VectorInfos]:
        """Reset every lane, deriving stable per-lane seeds from one seed.

        Args:
            seed: Optional base RNG seed; lane ``i`` receives ``seed + i``.

        Returns:
            Batched observations and one info mapping per lane.

        Raises:
            TypeError: If ``seed`` is not an integer or ``None``.
        """

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        results = tuple(
            environment.reset(seed=None if seed is None else seed + index)
            for index, environment in enumerate(self._envs)
        )
        observations, infos = zip(*results, strict=True)
        return np.stack(observations), infos

    def _normalize_actions(
        self,
        actions: Sequence[int] | IntegerArray,
    ) -> tuple[object, ...]:
        if isinstance(actions, np.ndarray):
            if actions.ndim != 1:
                raise ValueError("vector actions must be one-dimensional")
            normalized = tuple(actions)
        elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
            normalized = tuple(actions)
        else:
            raise TypeError("vector actions must be a sequence")
        if len(normalized) != self.num_envs:
            raise ValueError(f"vector step requires exactly {self.num_envs} actions")
        return normalized

    def _prevalidate_actions(self, actions: tuple[object, ...]) -> None:
        for index, (environment, action) in enumerate(zip(self._envs, actions, strict=True)):
            if environment.terminated:
                raise IllegalActionError(f"illegal action in terminated vector lane {index}")
            try:
                action_to_step(environment.state, cast(int, action))
            except (TypeError, ValueError) as error:
                raise IllegalActionError(f"illegal action in vector lane {index}") from error

    def step(self, actions: Sequence[int] | IntegerArray) -> VectorStepResult:
        """Advance every lane by exactly one environment step.

        The whole batch is prevalidated before any lane mutates, so an invalid action leaves every
        lane untouched.

        Args:
            actions: One canonical action ID per lane.

        Returns:
            Batched Gymnasium observations, rewards, terminal flags, truncation flags, and infos.

        Raises:
            TypeError: If the action batch is not a supported one-dimensional sequence.
            ValueError: If the batch has the wrong shape or length.
            IllegalActionError: If any lane is terminal or any action is illegal.
        """

        normalized = self._normalize_actions(actions)
        self._prevalidate_actions(normalized)
        results = tuple(
            environment.step(cast(int, action))
            for environment, action in zip(self._envs, normalized, strict=True)
        )
        observations, rewards, terminated, truncated, infos = zip(*results, strict=True)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            infos,
        )

    def serialize(self) -> str:
        """Serialize all lanes as canonical, versioned vector JSON.

        Returns:
            Deterministic vector snapshot containing each full environment snapshot.
        """

        payload = {
            "environments": [json.loads(environment.serialize()) for environment in self._envs],
            "schema": VECTOR_ENVIRONMENT_SCHEMA,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def restore(self, text: str) -> None:
        """Atomically restore every lane from a compatible vector snapshot.

        Args:
            text: Canonical vector snapshot.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If syntax, lane count, lane content, or rule configuration is invalid.
        """

        records = _canonical_environment_records(text)
        if len(records) != self.num_envs:
            raise ValueError(f"vector snapshot must contain exactly {self.num_envs} lanes")

        candidates: list[CheckersEnv] = []
        for index, (current, record) in enumerate(zip(self._envs, records, strict=True)):
            try:
                candidate = CheckersEnv.from_serialized(
                    current.serialize(),
                    render_mode=current.render_mode,
                )
                candidate.restore(record)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid vector snapshot lane {index}: {error}") from error
            candidates.append(candidate)
        self._envs = tuple(candidates)

    @classmethod
    def from_serialized(cls, text: str) -> CheckersVectorEnv:
        """Construct an exactly resumed vector environment from a snapshot.

        Args:
            text: Canonical vector snapshot.

        Returns:
            New vector environment with all lane state restored.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If snapshot syntax or any lane is invalid.
        """

        records = _canonical_environment_records(text)
        environments = tuple(CheckersEnv.from_serialized(record) for record in records)
        instance = cls.__new__(cls)
        instance._envs = environments
        return instance
