#!/usr/bin/env python3
"""Content fingerprint and record count for an explicit file or directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path) -> int | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return len(value) if isinstance(value, list) else 1
        if suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error):
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    root = Path(args.path).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    combined = hashlib.sha256()
    total_bytes = 0
    record_total = 0
    record_files = 0
    for file in files:
        relative = file.name if root.is_file() else file.relative_to(root).as_posix()
        digest = hash_file(file)
        size = file.stat().st_size
        combined.update(relative.encode() + b"\0" + digest.encode() + b"\0" + str(size).encode() + b"\n")
        total_bytes += size
        count = records(file)
        if count is not None:
            record_total += count
            record_files += 1
    print(json.dumps({
        "path": str(root),
        "sha256": combined.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "record_count": record_total if record_files else None,
        "record_files": record_files,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
