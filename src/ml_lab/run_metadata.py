"""Run identity, provenance, incremental metadata, and artifact hashing.

This module deliberately treats metadata as hostile input: likely credential keys are
removed recursively before anything is written to disk.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:token|api_?key|password|passwd|secret|credential|authorization)(?:$|_)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(r"^(?:hf_|sk-|ghp_|github_pat_|vast_|wandb_)[A-Za-z0-9_-]{8,}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_run_id(prefix: str = "run", now: datetime | None = None) -> str:
    """Return a sortable run id with enough entropy for concurrent launchers."""
    instant = now or datetime.now(UTC)
    stamp = instant.strftime("%Y%m%dT%H%M%S.%fZ")
    entropy = hashlib.sha256(f"{os.getpid()}:{stamp}".encode()).hexdigest()[:8]
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", prefix).strip("-.") or "run"
    return f"{safe_prefix}-{stamp}-{entropy}"


def sanitize_metadata(value: Any, key: str | None = None) -> Any:
    """Recursively redact credentials without ever returning their values."""
    if key is not None and SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): sanitize_metadata(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_metadata(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and SECRET_VALUE_RE.match(value):
        return "[REDACTED]"
    return value


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        sanitize_metadata(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_paths(paths: Iterable[str | Path], base: str | Path | None = None) -> str:
    """Hash file names and bytes deterministically (useful for dataset manifests)."""
    root = Path(base).resolve() if base else None
    digest = hashlib.sha256()
    normalized = sorted(Path(item).resolve() for item in paths)
    for path in normalized:
        name = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_state(repo: str | Path) -> dict[str, Any]:
    root = Path(repo)

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1")
    return {
        "commit": commit,
        "is_dirty": bool(status) if status is not None else None,
        "available": commit is not None,
    }


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def base_runtime_metadata(repo: str | Path, packages: Iterable[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "created_at": utc_now(),
        "python": {"version": platform.python_version(), "executable": os.sys.executable},
        "platform": platform.platform(),
        "git": git_state(repo),
        "packages": package_versions(packages),
    }
    try:
        import torch

        data["gpu"] = {
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            data["gpu"].update(
                {
                    "name": props.name,
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "total_vram_bytes": props.total_memory,
                    "count": torch.cuda.device_count(),
                }
            )
    except (ImportError, RuntimeError) as exc:
        data["gpu"] = {"cuda_available": False, "diagnostic": type(exc).__name__}
    return data


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe = sanitize_metadata(value)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(safe, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, destination)


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(sanitize_metadata(value), sort_keys=True, ensure_ascii=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(slots=True)
class RunMetadataStore:
    """Incrementally update a sanitized run metadata file."""

    path: Path
    data: dict[str, Any]

    @classmethod
    def create(cls, path: str | Path, initial: Mapping[str, Any]) -> RunMetadataStore:
        store = cls(Path(path), dict(initial))
        store.flush()
        return store

    def update(self, **changes: Any) -> None:
        self.data.update(changes)
        self.data["updated_at"] = utc_now()
        self.flush()

    def flush(self) -> None:
        atomic_write_json(self.path, self.data)
