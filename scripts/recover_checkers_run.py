#!/usr/bin/env python3
"""Prepare an audited, non-destructive Checkers PPO recovery directory."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from checkers.recovery import RecoveryError, prepare_recovery


def build_parser() -> argparse.ArgumentParser:
    """Return the stable recovery CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare the recovery directory and print one machine-readable result."""

    arguments = build_parser().parse_args(argv)
    command_arguments = sys.argv if argv is None else ["recover_checkers_run.py", *argv]
    try:
        result = prepare_recovery(
            repository=arguments.repository,
            source_run_directory=arguments.source_run,
            recovery_run_directory=arguments.output_dir,
            checkpoint_path=arguments.checkpoint,
            recovery_command=shlex.join(command_arguments),
        )
    except RecoveryError as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "recovery_run_directory": str(result.recovery_run_directory),
                "manifest": str(result.manifest_path),
                "checkpoint": str(result.recovered_checkpoint_path),
                "metrics": str(result.metrics_path),
                "orphaned_metrics": str(result.orphaned_metrics_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
