#!/usr/bin/env python3
"""Read-only terminal monitor for a Checkers PPO run directory."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from checkers.monitor import (
    collect_monitor_snapshot,
    live_runtime_process_id,
    render_monitor_snapshot,
)
from checkers.system_metrics import SystemTelemetrySampler


def build_parser() -> argparse.ArgumentParser:
    """Return the stable monitoring CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Poll read-only state until interrupted, or print one snapshot."""

    arguments = build_parser().parse_args(argv)
    if arguments.interval < 1.0:
        raise ValueError("monitor interval must be at least one second")
    sampler: SystemTelemetrySampler | None = None
    sampler_pid: int | None = None
    try:
        while True:
            current_pid = live_runtime_process_id(arguments.run_dir)
            if sampler is None or current_pid != sampler_pid:
                sampler = SystemTelemetrySampler(process_pid=current_pid)
                sampler_pid = current_pid
            snapshot = collect_monitor_snapshot(
                run_directory=arguments.run_dir,
                sampler=sampler,
            )
            if not arguments.once:
                sys.stdout.write("\x1b[2J\x1b[H")
            print(render_monitor_snapshot(snapshot), flush=True)
            if arguments.once:
                return 0
            time.sleep(arguments.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
