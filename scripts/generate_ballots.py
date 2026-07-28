#!/usr/bin/env python3
"""Generate exhaustive three-move evidence and unique-position evaluation ballots."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from checkers.eval.ballots import generate_ballot_artifacts

DEFAULT_SEQUENCES_OUTPUT = Path("data/ballot_sequences_v1.json")
DEFAULT_BALLOTS_OUTPUT = Path("data/ballots_v1.json")


def parse_args() -> argparse.Namespace:
    """Parse explicit paths for both versioned ballot artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences-output", type=Path, default=DEFAULT_SEQUENCES_OUTPUT)
    parser.add_argument("--ballots-output", type=Path, default=DEFAULT_BALLOTS_OUTPUT)
    return parser.parse_args()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> int:
    """Generate, atomically persist, and summarize both canonical artifacts."""

    arguments = parse_args()
    sequences, ballots = generate_ballot_artifacts()
    sequence_payload = sequences.to_json().encode("utf-8")
    ballot_payload = ballots.to_json().encode("utf-8")
    _atomic_write(arguments.sequences_output, sequence_payload)
    _atomic_write(arguments.ballots_output, ballot_payload)
    print(
        f"sequences={sequences.count} positions={ballots.count} "
        f"first_moves={sequences.distinct_first_moves} completed_moves={sequences.completed_moves}"
    )
    print(
        f"sequences_records_sha256={sequences.sha256} "
        f"sequences_file_sha256={hashlib.sha256(sequence_payload).hexdigest()}"
    )
    print(
        f"ballots_records_sha256={ballots.sha256} "
        f"ballots_file_sha256={hashlib.sha256(ballot_payload).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
