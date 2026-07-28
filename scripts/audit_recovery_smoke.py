#!/usr/bin/env python3
"""Audit a bounded Checkers PPO recovery continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkers.recovery import RecoveryError, audit_recovery_smoke


def build_parser() -> argparse.ArgumentParser:
    """Return the stable smoke-audit CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit the bounded continuation and print one machine-readable result."""

    arguments = build_parser().parse_args(argv)
    try:
        result = audit_recovery_smoke(
            run_directory=arguments.run_dir,
            expected_updates=arguments.expected_updates,
        )
    except RecoveryError as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "audit": str(result.audit_path),
                "report": str(result.report_path),
                "checkpoint": str(result.checkpoint_path),
                "end_update": result.end_update,
                "logging_step": result.logging_step,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
