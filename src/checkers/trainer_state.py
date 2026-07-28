"""Mutable trainer counters and complete random-number-generator snapshots."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from checkers.config import RunConfig

NumpyRNGState = tuple[str, NDArray[np.uint32], int, int, float]


@dataclass(frozen=True, slots=True)
class RNGStates:
    """Captured states for every random stream owned by one trainer."""

    python: object
    numpy: NumpyRNGState
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    opponent: object
    minibatch: torch.Tensor
    environments: tuple[object, ...]


@dataclass(slots=True)
class TrainerState:
    """Mutable counters/state kept separate from immutable `RunConfig`."""

    global_step: int = 0
    update_idx: int = 0
    schedule_phase: float = 0.0
    elapsed_training_seconds: float = 0.0
    wandb_run_id: str = ""
    logging_step: int = 0
    env_episode_indices: tuple[int, ...] = ()
    league_snapshot_ids: tuple[str, ...] = ()
    negative_explained_variance_streak: int = 0
    rng_states: RNGStates | None = None
    amp_scaler_state: dict[str, Any] = field(default_factory=dict)

    def advance_update(self, config: RunConfig, *, elapsed_seconds: float) -> None:
        """Advance exactly one completed rollout/update boundary.

        Args:
            config: Immutable run configuration defining rollout size.
            elapsed_seconds: Non-negative measured duration of this update.

        Raises:
            TypeError: If config or elapsed time has an invalid type.
            ValueError: If elapsed time is negative/non-finite.
            OverflowError: If the configured update budget is already exhausted.
        """

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)):
            raise TypeError("elapsed_seconds must be numeric")
        checked_elapsed = float(elapsed_seconds)
        if not math.isfinite(checked_elapsed) or checked_elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.update_idx >= config.total_updates:
            raise OverflowError("configured update budget is exhausted")
        self.global_step += config.batch_size
        self.update_idx += 1
        self.elapsed_training_seconds += checked_elapsed
        self.schedule_phase = min(1.0, self.global_step / config.total_timesteps)

    def advance_logging_step(self) -> None:
        """Increment the monotonic W&B/history logging step by one."""

        self.logging_step += 1


def capture_rng_states(
    *,
    opponent_rng: random.Random,
    minibatch_generator: torch.Generator,
    env_rngs: tuple[random.Random, ...],
) -> RNGStates:
    """Capture every global and trainer-owned RNG stream.

    Args:
        opponent_rng: League/opponent-selection PRNG.
        minibatch_generator: Torch generator used only for update permutations.
        env_rngs: Stable per-environment PRNGs.

    Returns:
        Deep, immutable snapshot suitable for a trusted local checkpoint.

    Raises:
        TypeError: If supplied stream objects have invalid runtime types.
    """

    if not isinstance(opponent_rng, random.Random):
        raise TypeError("opponent_rng must be random.Random")
    if not isinstance(minibatch_generator, torch.Generator):
        raise TypeError("minibatch_generator must be a torch Generator")
    if not isinstance(env_rngs, tuple) or not all(
        isinstance(stream, random.Random) for stream in env_rngs
    ):
        raise TypeError("env_rngs must be a tuple of random.Random streams")
    numpy_state = cast(NumpyRNGState, np.random.get_state())
    checked_numpy = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return RNGStates(
        python=copy.deepcopy(random.getstate()),
        numpy=checked_numpy,
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_states,
        opponent=copy.deepcopy(opponent_rng.getstate()),
        minibatch=minibatch_generator.get_state().clone(),
        environments=tuple(copy.deepcopy(stream.getstate()) for stream in env_rngs),
    )


def restore_rng_states(
    snapshot: RNGStates,
    *,
    opponent_rng: random.Random,
    minibatch_generator: torch.Generator,
    env_rngs: tuple[random.Random, ...],
) -> None:
    """Restore every global and trainer-owned RNG stream exactly.

    Args:
        snapshot: Previously captured RNG state.
        opponent_rng: Live league/opponent-selection PRNG.
        minibatch_generator: Live update-permutation generator.
        env_rngs: Live stable per-environment PRNGs.

    Raises:
        TypeError: If objects have invalid runtime types.
        ValueError: If environment lanes or CUDA-device counts differ.
    """

    if not isinstance(snapshot, RNGStates):
        raise TypeError("snapshot must be RNGStates")
    if not isinstance(opponent_rng, random.Random):
        raise TypeError("opponent_rng must be random.Random")
    if not isinstance(minibatch_generator, torch.Generator):
        raise TypeError("minibatch_generator must be a torch Generator")
    if not isinstance(env_rngs, tuple) or not all(
        isinstance(stream, random.Random) for stream in env_rngs
    ):
        raise TypeError("env_rngs must be a tuple of random.Random streams")
    if len(snapshot.environments) != len(env_rngs):
        raise ValueError("environment RNG lane count does not match checkpoint")
    if snapshot.torch_cuda and len(snapshot.torch_cuda) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device count does not match checkpoint")
    random.setstate(snapshot.python)  # type: ignore[arg-type]
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda:
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))
    opponent_rng.setstate(snapshot.opponent)  # type: ignore[arg-type]
    minibatch_generator.set_state(snapshot.minibatch)
    for stream, state in zip(env_rngs, snapshot.environments, strict=True):
        stream.setstate(state)  # type: ignore[arg-type]
