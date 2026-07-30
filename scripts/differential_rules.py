#!/usr/bin/env python3
"""Run and persist the Phase 2 fast/oracle differential gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from checkers.rules.differential import (
    DEFAULT_BFS_DEPTH,
    DEFAULT_DIGEST_INTERVAL,
    DEFAULT_MAX_PLIES,
    DEFAULT_POSITIONS,
    DEFAULT_SEED,
    DifferentialConfig,
    DifferentialMismatchError,
    run_differential,
)
from checkers.rules.notation import serialize_state

REPORT_SCHEMA_VERSION = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--bfs-depth", type=int, default=DEFAULT_BFS_DEPTH)
    parser.add_argument("--digest-interval", type=int, default=DEFAULT_DIGEST_INTERVAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disagreement-dir",
        type=Path,
        default=Path("tests/golden/disagreements"),
    )
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


def _contract_hash() -> str:
    return sha256(Path("docs/experiment-contract.md").read_bytes()).hexdigest()


def _metadata() -> dict[str, object]:
    uname = platform.uname()
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "goal_sha256": _contract_hash(),
        "python": platform.python_version(),
        "packages": {
            "hypothesis": version("hypothesis"),
            "ppo-checkers": version("ppo-checkers"),
            "pytest": version("pytest"),
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


def _write_disagreement(directory: Path, error: DifferentialMismatchError) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    path = directory / f"disagreement-{timestamp}-{error.stage}-{error.index}.json"
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "classification": "HAND ADJUDICATION REQUIRED — NOT GROUND TRUTH",
        "stage": error.stage,
        "index": error.index,
        "state": serialize_state(error.state),
        "fast": [asdict(step) for step in error.fast],
        "oracle": [asdict(step) for step in error.oracle],
        "metadata": _metadata(),
    }
    _write_new_json(path, payload)
    return path


def main() -> int:
    """Run the configured gate and write one immutable JSON report.

    Returns:
        Zero on agreement; two after persisting the first disagreement.

    Raises:
        FileExistsError: If the requested immutable output already exists.
        ValueError: If a configured budget is invalid.
    """

    args = _parse_args()
    config = DifferentialConfig(
        positions=args.positions,
        seed=args.seed,
        max_plies=args.max_plies,
        bfs_depth=args.bfs_depth,
        digest_interval=args.digest_interval,
    )
    started_at = datetime.now().astimezone().isoformat()
    started_clock = time.monotonic()
    metadata = _metadata()
    try:
        result = run_differential(config)
    except DifferentialMismatchError as error:
        disagreement_path = _write_disagreement(args.disagreement_dir, error)
        print(f"differential disagreement saved to {disagreement_path}", file=sys.stderr)
        return 2

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "classification": "INDEPENDENT AGREEMENT — CORROBORATION, NOT EXTERNAL CORRECTNESS",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "config": asdict(config),
        "result": asdict(result),
        "metadata": metadata,
    }
    _write_new_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "playout_positions": result.playout_positions,
                "bfs_positions": result.bfs_positions,
                "disagreements": result.disagreements,
                "state_digest_sha256": result.state_digest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
