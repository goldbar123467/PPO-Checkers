#!/usr/bin/env python3
"""Serve the trained checkers model and optional built Vite client on loopback."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from types import FrameType

from checkers.web.game import GameRetention, GameService
from checkers.web.policy_bundle import load_policy_bundle
from checkers.web.server import WebServerConfig, create_server


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-dir", type=Path, default=None)
    parser.add_argument("--max-active-games", type=int, default=256)
    parser.add_argument("--idle-ttl-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--max-concurrent-requests", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    """Load the policy before accepting requests and serve until interrupted."""

    args = _arguments()
    load_started = time.perf_counter()
    loaded_policy = load_policy_bundle(args.bundle)
    model_load_ms = (time.perf_counter() - load_started) * 1_000
    service = GameService(
        loaded_policy,
        retention=GameRetention(
            max_active_games=args.max_active_games,
            idle_ttl_seconds=args.idle_ttl_seconds,
        ),
    )
    config = WebServerConfig(
        port=args.port,
        static_dir=args.static_dir,
        max_concurrent_requests=args.max_concurrent_requests,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    server = create_server(config, service)
    host = config.host
    port = server.server_port
    print(
        json.dumps(
            {
                "event": "model_ready",
                "bundleId": loaded_policy.metadata.bundle_id,
                "bundleSha256": loaded_policy.sha256,
                "bundleSizeBytes": loaded_policy.size_bytes,
                "device": "cpu",
                "modelLoadMs": round(model_load_ms, 3),
                "parameterCount": sum(
                    parameter.numel() for parameter in loaded_policy.network.parameters()
                ),
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {"event": "server_ready", "host": host, "port": port},
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )

    def stop_on_signal(signum: int, frame: FrameType | None) -> None:
        _ = signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('{"event":"server_stopping"}', flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
