#!/usr/bin/env python3
"""Run and persist the deterministic Phase 4 environment fuzz gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Never, cast

import numpy as np

from checkers.env.checkers_env import CheckersEnv, StepResult
from checkers.env.encoding import encode_observation
from checkers.env.masking import ACTION_COUNT, legal_action_map
from checkers.rules.board import bit
from checkers.rules.moves import Step, legal_steps
from checkers.rules.notation import format_move, parse_move
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import TerminationReason, terminal_outcome
from checkers.rules.zobrist import state_key

DEFAULT_STEPS = 5_000_000
DEFAULT_SEED = 20260728
DEFAULT_SNAPSHOT_INTERVAL = 10_000
DEFAULT_PROGRESS_INTERVAL = 100_000
REPORT_SCHEMA_VERSION = 1
INFO_KEYS = frozenset(
    {
        "legal_mask",
        "actor",
        "move_completed",
        "checkers_move_san",
        "outcome",
    }
)
FIXTURE_SCHEDULE = (
    "initial",
    "initial",
    "initial",
    "initial",
    "two_jump",
    "no_progress_39",
    "ply_511",
    "promotion_capture",
    "last_piece_capture",
    "white_to_move",
)
SOURCE_PATHS = (
    Path("src/checkers/env/checkers_env.py"),
    Path("src/checkers/env/encoding.py"),
    Path("src/checkers/env/masking.py"),
    Path("src/checkers/env/serialize.py"),
    Path("src/checkers/env/vec_env.py"),
    Path("src/checkers/rules/moves.py"),
    Path("src/checkers/rules/state.py"),
    Path("src/checkers/rules/terminal.py"),
    Path("src/checkers/rules/zobrist.py"),
    Path("scripts/fuzz_environment.py"),
)


class FuzzInvariantError(AssertionError):
    """Raised immediately on the first fuzz invariant violation."""


@dataclass(frozen=True, slots=True)
class FuzzConfig:
    """Immutable deterministic environment-fuzz configuration."""

    steps: int = DEFAULT_STEPS
    seed: int = DEFAULT_SEED
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL

    def __post_init__(self) -> None:
        _positive_integer(self.steps, "steps")
        _integer(self.seed, "seed")
        _positive_integer(self.snapshot_interval, "snapshot_interval")
        _positive_integer(self.progress_interval, "progress_interval")


@dataclass(frozen=True, slots=True)
class FuzzProgress:
    """Small deterministic progress event emitted during a long gate."""

    steps_completed: int
    games_started: int
    games_terminated: int


@dataclass(frozen=True, slots=True)
class FuzzResult:
    """Deterministic counts from a successfully completed fuzz run."""

    steps_requested: int
    steps_completed: int
    seed: int
    games_started: int
    games_terminated: int
    fixture_starts: dict[str, int]
    termination_reasons: dict[str, int]
    simple_steps: int
    capture_steps: int
    continuation_steps: int
    completed_moves: int
    promotions: int
    snapshot_roundtrips: int
    midsequence_snapshot_roundtrips: int
    invariant_violations: int
    mask_disagreements: int
    empty_nonterminal_masks: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible deterministic result mapping.

        Returns:
            All result fields with fixture and terminal counts sorted by key.
        """

        payload = asdict(self)
        payload["fixture_starts"] = dict(sorted(self.fixture_starts.items()))
        payload["termination_reasons"] = dict(sorted(self.termination_reasons.items()))
        return cast(dict[str, object], payload)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    checked = _integer(value, name)
    if checked < 1:
        raise ValueError(f"{name} must be positive")
    return checked


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _fixture_state(name: str) -> State:
    if name == "initial":
        state = State.initial()
    elif name == "two_jump":
        state = State(
            men=(_mask(9, 11), _mask(14, 15, 22)),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        )
    elif name == "no_progress_39":
        state = State(
            men=(0, 0),
            kings=(_mask(14), _mask(24)),
            side_to_move=PlayerId.RED,
            no_progress=(39, 39),
        )
    elif name == "ply_511":
        state = State(
            men=(0, 0),
            kings=(_mask(14), _mask(24)),
            side_to_move=PlayerId.RED,
            ply=511,
        )
    elif name == "promotion_capture":
        state = State(
            men=(_mask(21), _mask(25, 26)),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        )
    elif name == "last_piece_capture":
        state = State(
            men=(_mask(9), _mask(14)),
            kings=(0, 0),
            side_to_move=PlayerId.RED,
        )
    elif name == "white_to_move":
        state = State(
            men=(_mask(9), _mask(24)),
            kings=(0, 0),
            side_to_move=PlayerId.WHITE,
        )
    else:
        raise ValueError(f"unknown fuzz fixture: {name}")
    return state


def _new_environment(
    game_index: int,
    fixture_starts: dict[str, int],
) -> CheckersEnv:
    fixture = FIXTURE_SCHEDULE[game_index % len(FIXTURE_SCHEDULE)]
    fixture_starts[fixture] += 1
    environment = CheckersEnv(initial_state=_fixture_state(fixture))
    if environment.terminated:
        raise FuzzInvariantError(f"fixture {fixture} unexpectedly starts terminal")
    return environment


def _fail(step_index: int, message: str) -> Never:
    raise FuzzInvariantError(f"fuzz step {step_index}: {message}")


def _validated_actions(environment: CheckersEnv, step_index: int) -> tuple[int, ...]:
    if environment.terminated:
        _fail(step_index, "current environment is already terminal")
    steps = legal_steps(environment.state)
    action_map = legal_action_map(environment.state)
    mask = environment.legal_mask()
    if mask.dtype != np.bool_ or mask.shape != (ACTION_COUNT,):
        _fail(step_index, "legal mask dtype or shape disagrees with contract")
    if not mask.any():
        _fail(step_index, "nonterminal state has an empty legal mask")
    encoded_actions = tuple(action_map)
    if tuple(action_map.values()) != steps:
        _fail(step_index, "action map does not preserve generated legal steps")
    if set(np.flatnonzero(mask).tolist()) != set(encoded_actions):
        _fail(step_index, "legal mask disagrees with action map")
    return encoded_actions


def _validate_pre_observation(environment: CheckersEnv, step_index: int) -> None:
    observation = environment.observe()
    expected = encode_observation(environment.state, max_plies=environment.max_plies)
    if not np.array_equal(observation, expected):
        _fail(step_index, "observation disagrees with canonical encoder")
    if not environment.observation_space.contains(observation):
        _fail(step_index, "observation lies outside declared Gymnasium space")
    if environment.state_key() != state_key(environment.state):
        _fail(step_index, "incremental state key disagrees with recomputation")


def _validate_piece_and_counter_transition(
    before: State,
    after: State,
    step: Step,
    move_completed: bool,
    step_index: int,
) -> bool:
    actor = int(before.side_to_move)
    opponent = int(before.side_to_move.opponent)
    was_man = bool(before.men[actor] & bit(step.origin))
    pending_removal = before.captured_pending
    if step.captured is not None:
        pending_removal |= bit(step.captured)
    expected_removed = pending_removal.bit_count() if move_completed and step.is_capture else 0
    actual_removed = before.occupied.bit_count() - after.occupied.bit_count()
    if actual_removed != expected_removed:
        _fail(step_index, "piece-count delta disagrees with delayed capture removal")

    if not move_completed:
        if after.no_progress != before.no_progress:
            _fail(step_index, "no-progress counter changed during a capture continuation")
    else:
        if after.no_progress[opponent] != before.no_progress[opponent]:
            _fail(step_index, "non-actor no-progress counter changed")
        expected_actor_counter = 0 if step.is_capture or was_man else before.no_progress[actor] + 1
        if after.no_progress[actor] != expected_actor_counter:
            _fail(step_index, "actor no-progress counter changed in the wrong unit")
    return was_man


def _validate_notation(
    before: State,
    step: Step,
    move_completed: bool,
    notation_value: object,
    step_index: int,
) -> None:
    notation = cast(str | None, notation_value)
    if move_completed:
        if notation is None:
            _fail(step_index, "completed move omitted notation")
        parsed = parse_move(notation)
        expected_origin = before.sequence_origin if before.capture_in_progress else step.origin
        if (
            parsed.squares[0] != expected_origin
            or parsed.squares[-1] != step.destination
            or parsed.is_capture != step.is_capture
            or format_move(parsed) != notation
        ):
            _fail(step_index, "completed notation disagrees with applied move")
    elif notation is not None:
        _fail(step_index, "capture continuation emitted completed-move notation")


def _validate_outcome(
    environment: CheckersEnv,
    actor: PlayerId,
    result_fields: tuple[float, bool, object],
    step_index: int,
) -> None:
    reward, terminated, reported_outcome = result_fields
    after = environment.state
    expected_outcome = terminal_outcome(after, max_plies=environment.max_plies)
    if terminated != (expected_outcome is not None):
        _fail(step_index, "terminal flag disagrees with rules evaluation")
    if reported_outcome != expected_outcome or environment.outcome != expected_outcome:
        _fail(step_index, "reported outcome disagrees with rules evaluation")
    expected_reward = 0.0 if expected_outcome is None else float(expected_outcome.score_for(actor))
    if reward != expected_reward:
        _fail(step_index, "reward disagrees with terminal actor perspective")


def _validate_returned_state(
    environment: CheckersEnv,
    observation: np.ndarray[Any, Any],
    returned_mask_value: object,
    terminated: bool,
    step_index: int,
) -> None:
    after = environment.state
    expected_observation = encode_observation(after, max_plies=environment.max_plies)
    if not np.array_equal(observation, expected_observation):
        _fail(step_index, "returned observation disagrees with resulting state")
    if environment.state_key() != state_key(after):
        _fail(step_index, "post-step state key disagrees with recomputation")
    returned_mask = cast(np.ndarray[Any, Any], returned_mask_value)
    if not np.array_equal(returned_mask, environment.legal_mask()):
        _fail(step_index, "returned legal mask disagrees with environment")
    if terminated:
        if returned_mask.any():
            _fail(step_index, "terminal state exposed an actionable mask")
    elif not returned_mask.any():
        _fail(step_index, "nonterminal result exposed an empty legal mask")


def _validate_step_result(
    environment: CheckersEnv,
    before: State,
    step: Step,
    result: StepResult,
    step_index: int,
) -> tuple[bool, bool]:
    observation, reward, terminated, truncated, info = result
    after = environment.state
    if set(info) != INFO_KEYS:
        _fail(step_index, "info fields disagree with the public contract")
    if info["actor"] is not before.side_to_move:
        _fail(step_index, "info actor is not the actor of the transition")
    if truncated:
        _fail(step_index, "a game rule set truncated instead of terminated")
    if after.ply != before.ply + 1:
        _fail(step_index, "ply did not increment by one environment step")

    move_completed = cast(bool, info["move_completed"])
    expected_side = before.side_to_move.opponent if move_completed else before.side_to_move
    if after.side_to_move is not expected_side:
        _fail(step_index, "side-to-move changed at the wrong boundary")
    was_man = _validate_piece_and_counter_transition(
        before,
        after,
        step,
        move_completed,
        step_index,
    )
    _validate_notation(
        before,
        step,
        move_completed,
        info["checkers_move_san"],
        step_index,
    )
    _validate_outcome(
        environment,
        before.side_to_move,
        (reward, terminated, info["outcome"]),
        step_index,
    )
    _validate_returned_state(
        environment,
        observation,
        info["legal_mask"],
        terminated,
        step_index,
    )

    promoted = bool(
        was_man and move_completed and after.kings[int(before.side_to_move)] & bit(step.destination)
    )
    return move_completed, promoted


def _validate_snapshot(environment: CheckersEnv, step_index: int) -> bool:
    serialized = environment.serialize()
    restored = CheckersEnv.from_serialized(serialized)
    if restored.serialize() != serialized:
        _fail(step_index, "environment snapshot is not a canonical round trip")
    if restored.state != environment.state or restored.state_key() != environment.state_key():
        _fail(step_index, "restored snapshot changed state or hash")
    if not np.array_equal(restored.observe(), environment.observe()):
        _fail(step_index, "restored snapshot changed observation")
    if not np.array_equal(restored.legal_mask(), environment.legal_mask()):
        _fail(step_index, "restored snapshot changed legal mask")
    return bool(environment.state.capture_in_progress)


def run_environment_fuzz(
    config: FuzzConfig,
    progress_callback: Callable[[FuzzProgress], None] | None = None,
) -> FuzzResult:
    """Run deterministic randomized environment transitions and halt on first failure.

    Args:
        config: Validated immutable run budget and seed.
        progress_callback: Optional callback at configured intervals and the final step.

    Returns:
        Deterministic counts with all three failure counters equal to zero.

    Raises:
        TypeError: If ``config`` is not a ``FuzzConfig``.
        FuzzInvariantError: On the first mask, transition, observation, hash, or snapshot failure.
    """

    if not isinstance(config, FuzzConfig):
        raise TypeError("config must be a FuzzConfig")
    rng = random.Random(config.seed)
    fixture_starts = {name: 0 for name in FIXTURE_SCHEDULE}
    termination_reasons = {reason.value: 0 for reason in TerminationReason}
    games_started = 1
    games_terminated = 0
    simple_steps = 0
    capture_steps = 0
    continuation_steps = 0
    completed_moves = 0
    promotions = 0
    snapshot_roundtrips = 0
    midsequence_snapshot_roundtrips = 0
    environment = _new_environment(0, fixture_starts)

    for index in range(config.steps):
        step_number = index + 1
        _validate_pre_observation(environment, index)
        action_ids = _validated_actions(environment, index)
        action = action_ids[rng.randrange(len(action_ids))]
        step = legal_action_map(environment.state)[action]
        before = environment.state
        result = environment.step(action)
        move_completed, promoted = _validate_step_result(
            environment,
            before,
            step,
            result,
            index,
        )

        if step.is_capture:
            capture_steps += 1
        else:
            simple_steps += 1
        if not move_completed:
            continuation_steps += 1
        else:
            completed_moves += 1
        promotions += int(promoted)

        if step_number % config.snapshot_interval == 0:
            midsequence_snapshot_roundtrips += int(_validate_snapshot(environment, index))
            snapshot_roundtrips += 1

        if environment.terminated:
            games_terminated += 1
            outcome = environment.outcome
            if outcome is None:
                _fail(index, "terminated environment omitted its outcome")
            termination_reasons[outcome.reason.value] += 1
            environment = _new_environment(games_started, fixture_starts)
            games_started += 1

        if progress_callback is not None and (
            step_number % config.progress_interval == 0 or step_number == config.steps
        ):
            progress_callback(
                FuzzProgress(
                    steps_completed=step_number,
                    games_started=games_started,
                    games_terminated=games_terminated,
                )
            )

    _validate_pre_observation(environment, config.steps)
    _validated_actions(environment, config.steps)
    return FuzzResult(
        steps_requested=config.steps,
        steps_completed=config.steps,
        seed=config.seed,
        games_started=games_started,
        games_terminated=games_terminated,
        fixture_starts=fixture_starts,
        termination_reasons=termination_reasons,
        simple_steps=simple_steps,
        capture_steps=capture_steps,
        continuation_steps=continuation_steps,
        completed_moves=completed_moves,
        promotions=promotions,
        snapshot_roundtrips=snapshot_roundtrips,
        midsequence_snapshot_roundtrips=midsequence_snapshot_roundtrips,
        invariant_violations=0,
        mask_disagreements=0,
        empty_nonterminal_masks=0,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_PROGRESS_INTERVAL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _gpu_name() -> str | None:
    try:
        completed = subprocess.run(
            ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else None


def _file_hashes() -> dict[str, str]:
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in SOURCE_PATHS}


def _metadata() -> dict[str, object]:
    uname = platform.uname()
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "goal_sha256": sha256(Path("docs/experiment-contract.md").read_bytes()).hexdigest(),
        "source_sha256": _file_hashes(),
        "python": platform.python_version(),
        "packages": {
            "gymnasium": version("gymnasium"),
            "ppo-checkers": version("ppo-checkers"),
            "numpy": version("numpy"),
        },
        "hardware": {
            "system": uname.system,
            "release": uname.release,
            "machine": uname.machine,
            "processor": uname.processor,
            "logical_cpu_count": os.cpu_count(),
            "gpu": _gpu_name(),
        },
    }


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    """Run the configured fuzz gate and write one immutable report.

    Returns:
        Zero after every requested transition passes all checks.

    Raises:
        FileExistsError: If the immutable output path already exists.
        FuzzInvariantError: On the first invariant failure.
        TypeError: If a budget has the wrong runtime type.
        ValueError: If a budget is outside its valid range.
    """

    args = _parse_args()
    config = FuzzConfig(
        steps=args.steps,
        seed=args.seed,
        snapshot_interval=args.snapshot_interval,
        progress_interval=args.progress_interval,
    )
    started_at = datetime.now().astimezone().isoformat()
    started_clock = time.monotonic()

    def report_progress(progress: FuzzProgress) -> None:
        print(json.dumps(asdict(progress), sort_keys=True), flush=True)

    result = run_environment_fuzz(config, report_progress)
    elapsed = time.monotonic() - started_clock
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "classification": "RANDOMIZED REGRESSION EVIDENCE — NOT A FORMAL PROOF",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "steps_per_second": result.steps_completed / elapsed,
        "config": asdict(config),
        "result": result.as_dict(),
        "metadata": _metadata(),
    }
    _write_new_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "steps_completed": result.steps_completed,
                "invariant_violations": result.invariant_violations,
                "mask_disagreements": result.mask_disagreements,
                "empty_nonterminal_masks": result.empty_nonterminal_masks,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
