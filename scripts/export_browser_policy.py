#!/usr/bin/env python3
"""Export a verified policy bundle for in-browser inference and record parity fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkers.web.browser_export import (
    build_parity_fixture,
    load_browser_policy,
    write_browser_policy,
)
from checkers.web.policy_bundle import load_policy_bundle

WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "checkers"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=WEB_ROOT / "src" / "model")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=WEB_ROOT / "src" / "test" / "fixtures" / "parity.json",
    )
    return parser.parse_args()


def main() -> int:
    """Write browser weights, prove exact reload parity, and record engine fixtures."""

    args = _arguments()
    source = load_policy_bundle(args.bundle)
    manifest = write_browser_policy(source, args.output_dir)
    browser_network, _manifest = load_browser_policy(args.output_dir)
    for source_tensor, browser_tensor in zip(
        source.network.state_dict().values(),
        browser_network.state_dict().values(),
        strict=True,
    ):
        if not torch.equal(source_tensor, browser_tensor):
            raise RuntimeError("browser weights differ from the source bundle")

    fixture = build_parity_fixture(browser_network, manifest)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"weights={args.output_dir / str(manifest['weightsFile'])}")
    print(f"weights_sha256={manifest['weightsSha256']}")
    print(f"parameters={manifest['parameterCount']}")
    print(f"fixture={args.fixture}")
    print(f"fixture_games={len(fixture['games'])}")  # type: ignore[arg-type]
    print(f"fixture_network_cases={len(fixture['network'])}")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
