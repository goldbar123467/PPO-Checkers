"""Prove that permanent rule tests kill five high-risk checkers-rule mutations.

Each challenge runs in a fresh temporary source tree. A challenge counts as killed only when the
unmodified baseline passes and the named pytest node reports a genuine test failure (exit code 1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
MOVES_PATH = Path("src/checkers/rules/moves.py")


@dataclass(frozen=True, slots=True)
class Replacement:
    """One exact source replacement that defines part of a mutation."""

    old: str
    new: str


@dataclass(frozen=True, slots=True)
class Challenge:
    """A named mutation and the permanent test expected to kill it."""

    name: str
    description: str
    test_node: str
    replacements: tuple[Replacement, ...]


@dataclass(frozen=True, slots=True)
class Execution:
    """Auditable result of one isolated pytest invocation."""

    name: str
    description: str
    test_node: str
    returncode: int
    killed: bool
    baseline_source_sha256: str
    tested_source_sha256: str
    pytest_output_sha256: str
    pytest_output_tail: tuple[str, ...]


FORWARD_DIRECTION_BLOCK = """\
    return (
        RED_FORWARD_DIRECTIONS if state.side_to_move is PlayerId.RED else WHITE_FORWARD_DIRECTIONS
    )
"""

MANDATORY_CAPTURE_BLOCK = """\
    if captures:
        return tuple(sorted(captures, key=_step_sort_key))
"""

CHALLENGES = (
    Challenge(
        name="allow_backward_man_jumps",
        description="Give uncrowned men all four king jump directions.",
        test_node="tests/rules/test_captures.py::test_r4_3_2_man_never_jumps_backward",
        replacements=(Replacement(FORWARD_DIRECTION_BLOCK, "    return KING_DIRECTIONS\n"),),
    ),
    Challenge(
        name="make_captures_optional",
        description="Expose simple moves alongside captures when a capture exists.",
        test_node="tests/rules/test_captures.py::test_r4_2_capture_is_mandatory_across_the_whole_player",
        replacements=(
            Replacement(
                MANDATORY_CAPTURE_BLOCK,
                """\
    if captures:
        optional_steps = [
            step
            for origin in _iter_squares(origins)
            for step in _simple_steps_from(state, origin)
        ]
        return tuple(sorted([*captures, *optional_steps], key=_step_sort_key))
""",
            ),
        ),
    ),
    Challenge(
        name="remove_captured_pieces_immediately",
        description="Clear a jumped piece before the capture sequence finishes.",
        test_node=(
            "tests/rules/test_captures.py::"
            "test_r4_5_marked_piece_remains_occupied_and_cannot_be_jumped_twice"
        ),
        replacements=(
            Replacement(
                """\
    captured = cast(int, step.captured)
    men, kings, was_man = _moved_boards(state, step)
""",
                """\
    captured = cast(int, step.captured)
    men, kings, was_man = _moved_boards(state, step)
    opponent = int(state.side_to_move.opponent)
    mutable_men = [men[0], men[1]]
    mutable_kings = [kings[0], kings[1]]
    mutable_men[opponent] &= ~bit(captured)
    mutable_kings[opponent] &= ~bit(captured)
    men = (mutable_men[0], mutable_men[1])
    kings = (mutable_kings[0], mutable_kings[1])
""",
            ),
            Replacement(
                "        captured_pending=state.captured_pending | bit(captured),\n",
                "        captured_pending=0,\n",
            ),
        ),
    ),
    Challenge(
        name="permit_continuation_after_promotion",
        description="Treat a man on the king row as a king and ignore the promotion stop.",
        test_node=(
            "tests/rules/test_promotion.py::"
            "test_r5_2_promotion_ends_jump_sequence_before_a_new_king_jump"
        ),
        replacements=(
            Replacement(
                "    if state.kings[actor] & bit(square):\n",
                "    if state.kings[actor] & bit(square) or "
                "_is_king_row(state.side_to_move, square):\n",
            ),
            Replacement(
                "    if not promotion_ends_move and "
                "_capture_steps_from(intermediate, step.destination):\n",
                "    if _capture_steps_from(intermediate, step.destination):\n",
            ),
        ),
    ),
    Challenge(
        name="add_majority_capture_rule",
        description="Suppress a one-jump choice whenever another first jump can continue.",
        test_node="tests/rules/test_captures.py::test_r4_6_no_majority_capture_rule",
        replacements=(
            Replacement(
                MANDATORY_CAPTURE_BLOCK,
                """\
    if captures:
        continuing = [
            step for step in captures if not _apply_capture(state, step).move_completed
        ]
        if continuing:
            captures = continuing
        return tuple(sorted(captures, key=_step_sort_key))
""",
            ),
        ),
    ),
)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _prepare_tree(root: Path, test_files: set[Path]) -> None:
    shutil.copytree(REPOSITORY / "src" / "checkers", root / "src" / "checkers")
    for relative_test in test_files:
        destination = root / relative_test
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / relative_test, destination)


def _pytest(root: Path, test_nodes: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--rootdir",
            str(root),
            "--no-cov",
            "-q",
            *test_nodes,
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _test_file(test_node: str) -> Path:
    return Path(test_node.split("::", maxsplit=1)[0])


def _mutate(source: str, challenge: Challenge) -> str:
    mutated = source
    for replacement in challenge.replacements:
        count = mutated.count(replacement.old)
        if count != 1:
            raise RuntimeError(f"{challenge.name}: expected one source marker, found {count}")
        mutated = mutated.replace(replacement.old, replacement.new, 1)
    return mutated


def _tail(output: str, line_count: int = 12) -> tuple[str, ...]:
    return tuple(output.splitlines()[-line_count:])


def _run_baseline(source_sha256: str) -> Execution:
    test_nodes = tuple(challenge.test_node for challenge in CHALLENGES)
    test_files = {_test_file(node) for node in test_nodes}
    with tempfile.TemporaryDirectory(prefix="checkers-mutation-baseline-") as directory:
        root = Path(directory)
        _prepare_tree(root, test_files)
        completed = _pytest(root, test_nodes)
    output = completed.stdout + completed.stderr
    return Execution(
        name="unmodified_baseline",
        description="All five selected permanent tests against unmodified isolated source.",
        test_node=",".join(test_nodes),
        returncode=completed.returncode,
        killed=False,
        baseline_source_sha256=source_sha256,
        tested_source_sha256=source_sha256,
        pytest_output_sha256=_sha256(output),
        pytest_output_tail=_tail(output),
    )


def _run_challenge(challenge: Challenge, source: str) -> Execution:
    test_file = _test_file(challenge.test_node)
    mutated = _mutate(source, challenge)
    with tempfile.TemporaryDirectory(prefix=f"checkers-mutation-{challenge.name}-") as directory:
        root = Path(directory)
        _prepare_tree(root, {test_file})
        (root / MOVES_PATH).write_text(mutated, encoding="utf-8")
        completed = _pytest(root, (challenge.test_node,))
    output = completed.stdout + completed.stderr
    killed = completed.returncode == 1 and "1 failed" in output
    return Execution(
        name=challenge.name,
        description=challenge.description,
        test_node=challenge.test_node,
        returncode=completed.returncode,
        killed=killed,
        baseline_source_sha256=_sha256(source),
        tested_source_sha256=_sha256(mutated),
        pytest_output_sha256=_sha256(output),
        pytest_output_tail=_tail(output),
    )


def run(output_path: Path) -> bool:
    """Run all challenges, write an immutable-style JSON report, and return success."""

    source = (REPOSITORY / MOVES_PATH).read_text(encoding="utf-8")
    source_sha256 = _sha256(source)
    baseline = _run_baseline(source_sha256)
    results = tuple(_run_challenge(challenge, source) for challenge in CHALLENGES)
    all_killed = baseline.returncode == 0 and all(result.killed for result in results)
    report = {
        "schema_version": 1,
        "source": str(MOVES_PATH),
        "source_sha256": source_sha256,
        "baseline": asdict(baseline),
        "challenges": [asdict(result) for result in results],
        "all_killed": all_killed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return all_killed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "reports" / "phase2_rule_mutation_challenges.json",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line mutation challenge gate."""

    arguments = _parse_args()
    succeeded = run(arguments.output)
    print(f"rule mutation challenges: {'PASS' if succeeded else 'FAIL'}")
    print(f"report: {arguments.output}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
