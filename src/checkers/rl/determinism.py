"""Single-entry deterministic seeding for Python, NumPy, Torch, CUDA, and env lanes."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

UINT64_MASK = (1 << 64) - 1
UINT32_MASK = (1 << 32) - 1
SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
ENV_STREAM_OFFSET = 4


def _uint64(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= UINT64_MASK:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_integer(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked < 1:
        raise ValueError(f"{name} must be positive")
    return checked


def _splitmix64(value: int) -> int:
    mixed = (value + SPLITMIX_INCREMENT) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * SPLITMIX_MULTIPLIER_1) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * SPLITMIX_MULTIPLIER_2) & UINT64_MASK
    return mixed ^ (mixed >> 31)


def derive_stream_seed(root_seed: int, stream_index: int) -> int:
    """Derive one deterministic unsigned 64-bit sub-seed.

    Args:
        root_seed: Unsigned 64-bit experiment seed.
        stream_index: Non-negative deterministic stream ordinal.

    Returns:
        SplitMix64-derived unsigned 64-bit seed.

    Raises:
        TypeError: If an input is not an integer or is boolean.
        ValueError: If the root is outside uint64 or the index is negative.
    """

    root = _uint64(root_seed, "root_seed")
    index = _nonnegative_integer(stream_index, "stream_index")
    return _splitmix64((root + index) & UINT64_MASK)


@dataclass(frozen=True, slots=True)
class SeedStreams:
    """Exact global and per-environment sub-seeds applied by one call."""

    root_seed: int
    python_seed: int
    numpy_seed: int
    torch_seed: int
    cuda_seed: int
    env_seeds: tuple[int, ...]
    deterministic: bool


def seed_everything(
    seed: int,
    *,
    num_envs: int,
    deterministic: bool,
) -> SeedStreams:
    """Seed all local RNG families and configure deterministic Torch execution.

    Args:
        seed: Unsigned 64-bit root seed.
        num_envs: Number of distinct per-environment sub-seeds to derive.
        deterministic: Enable or disable PyTorch deterministic algorithms.

    Returns:
        Immutable record of every applied and derived seed.

    Raises:
        TypeError: If arguments have invalid runtime types.
        ValueError: If seed/dimensions are outside their allowed range.
    """

    root = _uint64(seed, "seed")
    env_count = _positive_integer(num_envs, "num_envs")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be bool")
    python_seed = derive_stream_seed(root, 0)
    numpy_seed = derive_stream_seed(root, 1) & UINT32_MASK
    torch_seed = derive_stream_seed(root, 2)
    cuda_seed = derive_stream_seed(root, 3)
    env_seeds = tuple(
        derive_stream_seed(root, ENV_STREAM_OFFSET + index) for index in range(env_count)
    )

    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cuda_seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return SeedStreams(
        root_seed=root,
        python_seed=python_seed,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed,
        cuda_seed=cuda_seed,
        env_seeds=env_seeds,
        deterministic=deterministic,
    )
