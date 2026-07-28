#!/usr/bin/env python3
"""Run the measured seed-0 practice acceptance preflight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from checkers.practice_preflight import run_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkers-practice.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            Path("runs")
            / f"practice-preflight-{datetime.now().astimezone():%Y%m%dT%H%M%S%z}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    report = run_preflight(
        repository=repository,
        config_path=arguments.config.resolve(),
        output_directory=arguments.output_dir.resolve(),
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
