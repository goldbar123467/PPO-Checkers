#!/usr/bin/env python3
"""Fingerprint only files intentionally synchronized to a Vast worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fixed = [root / "pyproject.toml", root / "uv.lock", root / "README.md"]
    trees = [root / "src", root / "configs", root / "cloud" / "vast" / "accelerate"]
    helpers = [
        root / "scripts" / "configure-env.sh",
        root / "scripts" / "vast" / "_fingerprint.py",
        root / "scripts" / "vast" / "_code_fingerprint.py",
    ]
    files = [p for p in fixed + helpers if p.is_file()]
    for tree in trees:
        if tree.is_dir():
            files.extend(p for p in tree.rglob("*") if p.is_file())
    files = sorted(set(files))
    digest = hashlib.sha256()
    total = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        total += size
        digest.update(rel.encode() + b"\0" + content_hash.encode() + b"\0" + str(size).encode() + b"\n")
    print(json.dumps({"sha256": digest.hexdigest(), "file_count": len(files), "total_bytes": total}, sort_keys=True))


if __name__ == "__main__":
    main()
