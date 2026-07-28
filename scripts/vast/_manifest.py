#!/usr/bin/env python3
"""Append-only, hash-chained Vast run manifests (never stores credentials)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid manifest JSON at line {number}: {exc}") from exc
    return records


def verify_chain(records: list[dict[str, Any]]) -> None:
    previous = None
    for index, record in enumerate(records):
        expected_previous = record.get("previous_record_hash")
        if expected_previous != previous:
            raise SystemExit(f"Manifest hash chain is invalid at record {index + 1}")
        unsigned = {k: v for k, v in record.items() if k != "record_hash"}
        actual = hashlib.sha256(canonical(unsigned)).hexdigest()
        if actual != record.get("record_hash"):
            raise SystemExit(f"Manifest record hash is invalid at record {index + 1}")
        previous = actual


def append(args: argparse.Namespace) -> None:
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = read_records(path)
    verify_chain(records)
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Payload must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Payload must be a JSON object")
    forbidden = {"api_key", "token", "password", "secret", "authorization"}
    if forbidden.intersection(k.lower() for k in payload):
        raise SystemExit("Refusing credential-like top-level manifest fields")
    previous = records[-1]["record_hash"] if records else None
    record = {
        "schema_version": 1,
        "sequence": len(records) + 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "local_run_id": args.run_id,
        "event": args.event,
        "previous_record_hash": previous,
        "payload": payload,
    }
    record["record_hash"] = hashlib.sha256(canonical(record)).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, canonical(record) + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    print(path)


def find(args: argparse.Namespace) -> None:
    matches: list[Path] = []
    for path in sorted(Path(args.directory).glob("*.jsonl")):
        records = read_records(path)
        verify_chain(records)
        if any(str(r.get("payload", {}).get("remote_instance_id", "")) == args.instance_id for r in records):
            matches.append(path)
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one manifest for instance {args.instance_id}; found {len(matches)}")
    print(matches[0])


def has(args: argparse.Namespace) -> None:
    records = read_records(Path(args.manifest))
    verify_chain(records)
    if not any(record.get("event") == args.event for record in records):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(required=True)
    add = subs.add_parser("append")
    add.add_argument("--manifest", required=True)
    add.add_argument("--run-id", required=True)
    add.add_argument("--event", required=True)
    add.add_argument("--payload", default="{}")
    add.set_defaults(func=append)
    finder = subs.add_parser("find")
    finder.add_argument("--directory", required=True)
    finder.add_argument("--instance-id", required=True)
    finder.set_defaults(func=find)
    event = subs.add_parser("has")
    event.add_argument("--manifest", required=True)
    event.add_argument("--event", required=True)
    event.set_defaults(func=has)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
