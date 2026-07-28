"""Crash-safe checkpoint and deterministic raw-archive support for baseline runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from checkers.eval.arena import MatchResult
from checkers.eval.baseline_eval import (
    BaselineMatchSummary,
    match_result_record,
    parse_match_result_record,
    summarize_match,
)

SCHEMA_VERSION = 1
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PAIR_SIZE = 2


def _mapping(value: object, field_name: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return cast(dict[object, object], value)


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return cast(list[object], value)


def _required(mapping: dict[object, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"missing required field {key!r}")
    return mapping[key]


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _nonnegative_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return checked


def _digest(value: object, field_name: str, pattern: re.Pattern[str]) -> str:
    checked = _string(value, field_name)
    if pattern.fullmatch(checked) is None:
        raise ValueError(f"{field_name} has an invalid digest")
    return checked


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Immutable source/config/goal identity shared by every run artifact."""

    experiment_id: str
    git_commit: str
    config_sha256: str
    goal_sha256: str

    def __post_init__(self) -> None:
        _string(self.experiment_id, "experiment_id")
        _digest(self.git_commit, "git_commit", GIT_SHA_PATTERN)
        _digest(self.config_sha256, "config_sha256", SHA256_PATTERN)
        _digest(self.goal_sha256, "goal_sha256", SHA256_PATTERN)


def _identity_record(identity: RunIdentity) -> dict[str, object]:
    return {
        "experiment_id": identity.experiment_id,
        "git_commit": identity.git_commit,
        "config_sha256": identity.config_sha256,
        "goal_sha256": identity.goal_sha256,
    }


def _parse_identity(value: object) -> RunIdentity:
    root = _mapping(value, "identity")
    return RunIdentity(
        experiment_id=_string(_required(root, "experiment_id"), "experiment_id"),
        git_commit=_string(_required(root, "git_commit"), "git_commit"),
        config_sha256=_string(_required(root, "config_sha256"), "config_sha256"),
        goal_sha256=_string(_required(root, "goal_sha256"), "goal_sha256"),
    )


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes.

    Args:
        value: Exact artifact bytes.

    Returns:
        Lowercase 64-character digest.

    Raises:
        TypeError: If ``value`` is not bytes.
    """

    if not isinstance(value, bytes):
        raise TypeError("value must be bytes")
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one file after fully flushing a sibling temporary file.

    Args:
        path: Exact destination path.
        payload: Complete bytes to persist.

    Raises:
        TypeError: If inputs have invalid runtime types.
        OSError: If directory creation, writing, flushing, or replacement fails.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_summary(match: MatchResult, summary: BaselineMatchSummary) -> None:
    if not isinstance(summary, BaselineMatchSummary):
        raise TypeError("summary must be a BaselineMatchSummary")
    expected = summarize_match(match, elapsed_seconds=summary.elapsed_seconds)
    if summary != expected:
        raise ValueError("summary does not match the replay-complete match")


def build_checkpoint(
    *,
    identity: RunIdentity,
    comparison_index: int,
    match: MatchResult,
    summary: BaselineMatchSummary,
) -> dict[str, object]:
    """Build one self-validating, replay-complete resumable checkpoint.

    Args:
        identity: Immutable source/config/goal identity.
        comparison_index: Zero-based configured pair index.
        match: Complete arena result.
        summary: Digest-bound summary of that exact result.

    Returns:
        JSON-compatible checkpoint record.

    Raises:
        TypeError: If a record has an invalid runtime type.
        ValueError: If the summary and replay content disagree.
    """

    if not isinstance(identity, RunIdentity):
        raise TypeError("identity must be a RunIdentity")
    index = _nonnegative_integer(comparison_index, "comparison_index")
    if not isinstance(match, MatchResult):
        raise TypeError("match must be a MatchResult")
    _validate_summary(match, summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": _identity_record(identity),
        "comparison_index": index,
        "elapsed_seconds": summary.elapsed_seconds,
        "records_sha256": summary.records_sha256,
        "match": match_result_record(match),
    }


def parse_checkpoint(  # noqa: PLR0913
    value: object,
    *,
    identity: RunIdentity,
    comparison_index: int,
    expected_pair: tuple[str, str],
    expected_seed: int,
) -> tuple[MatchResult, BaselineMatchSummary]:
    """Load a checkpoint only if every identity, schedule, and summary invariant agrees.

    Args:
        value: Untrusted checkpoint document.
        identity: Required immutable run identity.
        comparison_index: Required schedule index.
        expected_pair: Required oriented policy pair.
        expected_seed: Required experiment-level root seed.

    Returns:
        Reconstructed match and recomputed summary.

    Raises:
        TypeError: If fields have invalid runtime types.
        ValueError: If identity, schedule, replay, or digest validation fails.
    """

    if not isinstance(identity, RunIdentity):
        raise TypeError("identity must be a RunIdentity")
    root = _mapping(value, "checkpoint")
    if _nonnegative_integer(_required(root, "schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if _parse_identity(_required(root, "identity")) != identity:
        raise ValueError("checkpoint identity does not match this run")
    expected_index = _nonnegative_integer(comparison_index, "comparison_index")
    if (
        _nonnegative_integer(_required(root, "comparison_index"), "comparison_index")
        != expected_index
    ):
        raise ValueError("checkpoint comparison_index does not match")
    if (
        not isinstance(expected_pair, tuple)
        or len(expected_pair) != PAIR_SIZE
        or not all(isinstance(name, str) and name for name in expected_pair)
    ):
        raise ValueError("expected_pair must contain two names")
    elapsed = _nonnegative_number(_required(root, "elapsed_seconds"), "elapsed_seconds")
    expected_records_sha = _digest(
        _required(root, "records_sha256"),
        "records_sha256",
        SHA256_PATTERN,
    )
    match = parse_match_result_record(_required(root, "match"))
    if (match.first_agent, match.second_agent) != expected_pair:
        raise ValueError("checkpoint pair does not match the schedule")
    checked_seed = _nonnegative_integer(expected_seed, "expected_seed")
    if match.seed != checked_seed:
        raise ValueError("checkpoint seed does not match the schedule")
    summary = summarize_match(match, elapsed_seconds=elapsed)
    if summary.records_sha256 != expected_records_sha:
        raise ValueError("checkpoint summary hash does not match replay records")
    return match, summary


def build_raw_archive(
    *,
    identity: RunIdentity,
    matches: tuple[MatchResult, ...],
) -> bytes:
    """Build deterministic gzip bytes containing every action and seed from all matches.

    Args:
        identity: Immutable source/config/goal identity.
        matches: Non-empty ordered replay-complete match tuple.

    Returns:
        Canonical JSON compressed with deterministic gzip metadata.

    Raises:
        TypeError: If identity or matches have invalid runtime types.
    """

    if not isinstance(identity, RunIdentity):
        raise TypeError("identity must be a RunIdentity")
    if (
        not isinstance(matches, tuple)
        or not matches
        or not all(isinstance(match, MatchResult) for match in matches)
    ):
        raise TypeError("matches must be a non-empty tuple of MatchResult values")
    document = {
        "schema_version": SCHEMA_VERSION,
        "identity": _identity_record(identity),
        "matches": [match_result_record(match) for match in matches],
    }
    return gzip.compress(_canonical_json_bytes(document), compresslevel=9, mtime=0)


def parse_raw_archive(
    value: bytes,
    *,
    identity: RunIdentity,
) -> tuple[MatchResult, ...]:
    """Decompress, parse, and invariant-check a deterministic raw-match archive.

    Args:
        value: Exact gzip artifact bytes.
        identity: Required immutable run identity.

    Returns:
        Ordered reconstructed matches.

    Raises:
        TypeError: If inputs or decoded fields have invalid runtime types.
        ValueError: If compression, JSON, identity, schema, or replay validation fails.
    """

    if not isinstance(value, bytes):
        raise TypeError("value must be bytes")
    if not isinstance(identity, RunIdentity):
        raise TypeError("identity must be a RunIdentity")
    try:
        payload = gzip.decompress(value)
    except (EOFError, OSError) as error:
        raise ValueError("value must be a valid gzip archive") from error
    try:
        loaded: object = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("gzip payload must be valid UTF-8 JSON") from error
    root = _mapping(loaded, "raw archive")
    if _nonnegative_integer(_required(root, "schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if _parse_identity(_required(root, "identity")) != identity:
        raise ValueError("raw archive identity does not match this run")
    matches = tuple(
        parse_match_result_record(record) for record in _list(_required(root, "matches"), "matches")
    )
    if not matches:
        raise ValueError("raw archive matches must not be empty")
    return matches
