"""Deterministic large-scale differential runner for the two move generators."""

from __future__ import annotations

import hashlib
import random
import struct
from collections.abc import Callable
from dataclasses import dataclass

from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.oracle import oracle_legal_steps
from checkers.rules.state import State

DEFAULT_POSITIONS = 5_000_000
DEFAULT_SEED = 20_260_727
DEFAULT_MAX_PLIES = 512
DEFAULT_BFS_DEPTH = 7
DEFAULT_DIGEST_INTERVAL = 1_024
OPTIONAL_SQUARE_SENTINEL = 32

StepGenerator = Callable[[State], tuple[Step, ...]]


@dataclass(frozen=True, slots=True)
class DifferentialConfig:
    """Fixed budget and seed for one reproducible differential run.

    Args:
        positions: Number of reachable playout positions to compare.
        seed: Non-negative Python PRNG seed.
        max_plies: Per-game reset boundary used only by this finite runner.
        bfs_depth: Inclusive breadth-first comparison depth from the initial state.
        digest_interval: Position interval for deterministic state-digest sampling.

    Raises:
        ValueError: If any budget is outside its valid non-negative range.
    """

    positions: int = DEFAULT_POSITIONS
    seed: int = DEFAULT_SEED
    max_plies: int = DEFAULT_MAX_PLIES
    bfs_depth: int = DEFAULT_BFS_DEPTH
    digest_interval: int = DEFAULT_DIGEST_INTERVAL

    def __post_init__(self) -> None:
        if self.positions <= 0:
            raise ValueError("positions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_plies <= 0:
            raise ValueError("max_plies must be positive")
        if self.bfs_depth < 0:
            raise ValueError("bfs_depth must be non-negative")
        if self.digest_interval <= 0:
            raise ValueError("digest_interval must be positive")


@dataclass(frozen=True, slots=True)
class DifferentialResult:
    """Deterministic counters and digest emitted by a successful run."""

    playout_positions: int
    bfs_positions: int
    unique_bfs_states: int
    steps_applied: int
    games_started: int
    capture_steps: int
    continuation_steps: int
    max_pending_captures: int
    digest_samples: int
    state_digest_sha256: str
    disagreements: int = 0


class DifferentialMismatchError(AssertionError):
    """A fast/oracle disagreement with its exact reproducible state."""

    def __init__(
        self,
        *,
        state: State,
        fast: tuple[Step, ...],
        oracle: tuple[Step, ...],
        stage: str,
        index: int,
    ) -> None:
        super().__init__(f"generator disagreement during {stage} at index {index}")
        self.state = state
        self.fast = fast
        self.oracle = oracle
        self.stage = stage
        self.index = index


def _compare(
    state: State,
    oracle_generator: StepGenerator,
    *,
    stage: str,
    index: int,
) -> tuple[Step, ...]:
    fast = legal_steps(state)
    oracle = oracle_generator(state)
    if fast != oracle:
        raise DifferentialMismatchError(
            state=state,
            fast=fast,
            oracle=oracle,
            stage=stage,
            index=index,
        )
    return fast


def _run_bfs(depth: int, oracle_generator: StepGenerator) -> tuple[int, int]:
    frontier = {State.initial()}
    seen = set(frontier)
    comparisons = 0
    for _level in range(depth + 1):
        next_frontier: set[State] = set()
        for state in frontier:
            fast = _compare(state, oracle_generator, stage="bfs", index=comparisons)
            comparisons += 1
            for step in fast:
                child = apply_step(state, step).after
                if child not in seen:
                    seen.add(child)
                    next_frontier.add(child)
        frontier = next_frontier
    return comparisons, len(seen)


def _digest_state(digest: hashlib._Hash, state: State) -> None:
    moving = OPTIONAL_SQUARE_SENTINEL if state.moving_square is None else state.moving_square
    origin = OPTIONAL_SQUARE_SENTINEL if state.sequence_origin is None else state.sequence_origin
    digest.update(
        struct.pack(
            "<IIIIBBBBIQQQ",
            state.men[0],
            state.men[1],
            state.kings[0],
            state.kings[1],
            int(state.side_to_move),
            int(state.capture_in_progress),
            moving,
            origin,
            state.captured_pending,
            state.no_progress[0],
            state.no_progress[1],
            state.ply,
        )
    )


def run_differential(
    config: DifferentialConfig,
    *,
    oracle_generator: StepGenerator = oracle_legal_steps,
) -> DifferentialResult:
    """Compare fast and oracle generators over BFS and seeded reachable playouts.

    Args:
        config: Immutable position, seed, BFS, cap, and digest configuration.
        oracle_generator: Injectable independent generator; defaults to the object-grid oracle.

    Returns:
        Deterministic successful-run counters and sampled state digest.

    Raises:
        DifferentialMismatchError: On the first generator disagreement.
    """

    bfs_positions, unique_bfs_states = _run_bfs(config.bfs_depth, oracle_generator)
    rng = random.Random(config.seed)
    digest = hashlib.sha256()
    state = State.initial()
    games_started = 1
    steps_applied = 0
    capture_steps = 0
    continuation_steps = 0
    max_pending_captures = 0
    digest_samples = 0

    for index in range(config.positions):
        fast = _compare(state, oracle_generator, stage="playout", index=index)
        if index % config.digest_interval == 0:
            _digest_state(digest, state)
            digest_samples += 1
        if state.ply >= config.max_plies or not fast:
            state = State.initial()
            games_started += 1
            continue

        step = fast[rng.randrange(len(fast))]
        transition = apply_step(state, step)
        state = transition.after
        steps_applied += 1
        capture_steps += int(step.is_capture)
        continuation_steps += int(not transition.move_completed)
        max_pending_captures = max(max_pending_captures, state.captured_pending.bit_count())

    return DifferentialResult(
        playout_positions=config.positions,
        bfs_positions=bfs_positions,
        unique_bfs_states=unique_bfs_states,
        steps_applied=steps_applied,
        games_started=games_started,
        capture_steps=capture_steps,
        continuation_steps=continuation_steps,
        max_pending_captures=max_pending_captures,
        digest_samples=digest_samples,
        state_digest_sha256=digest.hexdigest(),
    )
