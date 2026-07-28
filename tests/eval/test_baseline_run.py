"""Crash-safe identity, checkpoint, and raw-archive baseline run support."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from checkers.eval.arena import MatchResult, play_balanced_match
from checkers.eval.baseline_eval import (
    BaselineMatchSummary,
    agent_spec,
    load_baseline_config,
    summarize_match,
)
from checkers.eval.baseline_run import (
    RunIdentity,
    atomic_write_bytes,
    build_checkpoint,
    build_raw_archive,
    parse_checkpoint,
    parse_raw_archive,
    sha256_bytes,
)
from checkers.rules.state import PlayerId, State

CONFIG_PATH = Path("configs/checkers-baselines-v1.yaml")
SHA256 = "a" * 64
GIT_SHA = "b" * 40
SMALL_GAMES = 2
SHA256_LENGTH = 64
EXPECTED_GAMES = 784


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _match(seed: int = 11) -> MatchResult:
    state = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    return play_balanced_match(
        first=agent_spec("greedy", max_plies=512),
        second=agent_spec("random", max_plies=512),
        games=SMALL_GAMES,
        seed=seed,
        initial_state=state,
    )


def _identity() -> RunIdentity:
    return RunIdentity(
        experiment_id="checkers-baselines-v1",
        git_commit=GIT_SHA,
        config_sha256=SHA256,
        goal_sha256="c" * 64,
    )


def test_checkpoint_round_trip_revalidates_match_and_summary() -> None:
    match = _match()
    summary = summarize_match(match, elapsed_seconds=0.25)
    record = build_checkpoint(
        identity=_identity(),
        comparison_index=0,
        match=match,
        summary=summary,
    )

    restored_match, restored_summary = parse_checkpoint(
        record,
        identity=_identity(),
        comparison_index=0,
        expected_pair=("greedy", "random"),
        expected_seed=11,
    )

    assert restored_match == match
    assert restored_summary == summary


def test_raw_archive_is_deterministic_valid_gzip_and_replay_complete() -> None:
    matches = (_match(11), _match(12))

    first = build_raw_archive(identity=_identity(), matches=matches)
    second = build_raw_archive(identity=_identity(), matches=matches)
    restored = parse_raw_archive(first, identity=_identity())

    assert first == second
    assert gzip.decompress(first)
    assert restored == matches
    assert len(sha256_bytes(first)) == SHA256_LENGTH


def test_atomic_bytes_replace_existing_file_without_partial_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.bin"

    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")

    assert target.read_bytes() == b"second"
    assert not list(target.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda _record: [], "mapping"),
        (lambda record: {**record, "schema_version": 2}, "schema_version"),
        (
            lambda record: {
                **record,
                "identity": {
                    **cast(dict[str, object], record["identity"]),
                    "git_commit": "d" * 40,
                },
            },
            "identity",
        ),
        (lambda record: {**record, "comparison_index": 1}, "comparison_index"),
        (lambda record: {**record, "elapsed_seconds": -1.0}, "elapsed"),
        (
            lambda record: {**record, "records_sha256": "0" * 64},
            "summary",
        ),
    ],
)
def test_checkpoint_rejects_corruption(mutate: object, message: str) -> None:
    match = _match()
    record = build_checkpoint(
        identity=_identity(),
        comparison_index=0,
        match=match,
        summary=summarize_match(match, elapsed_seconds=0.25),
    )
    mutation = cast(Callable[[dict[str, object]], object], mutate)

    with pytest.raises((TypeError, ValueError), match=message):
        parse_checkpoint(
            mutation(record),
            identity=_identity(),
            comparison_index=0,
            expected_pair=("greedy", "random"),
            expected_seed=11,
        )


def test_checkpoint_rejects_wrong_expected_pair_or_seed() -> None:
    match = _match()
    record = build_checkpoint(
        identity=_identity(),
        comparison_index=0,
        match=match,
        summary=summarize_match(match, elapsed_seconds=0.1),
    )

    with pytest.raises(ValueError, match="pair"):
        parse_checkpoint(
            record,
            identity=_identity(),
            comparison_index=0,
            expected_pair=("minimax(1)", "random"),
            expected_seed=11,
        )
    with pytest.raises(ValueError, match="seed"):
        parse_checkpoint(
            record,
            identity=_identity(),
            comparison_index=0,
            expected_pair=("greedy", "random"),
            expected_seed=12,
        )


def test_raw_archive_rejects_identity_and_payload_corruption() -> None:
    archive = build_raw_archive(identity=_identity(), matches=(_match(),))
    document = cast(dict[str, object], json.loads(gzip.decompress(archive)))
    identity_record = cast(dict[str, object], document["identity"])
    identity_record["git_commit"] = "d" * 40
    corrupted = gzip.compress(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode(),
        mtime=0,
    )

    with pytest.raises(ValueError, match="identity"):
        parse_raw_archive(corrupted, identity=_identity())
    with pytest.raises(ValueError, match="gzip"):
        parse_raw_archive(b"not gzip", identity=_identity())


def test_run_identity_and_builders_reject_invalid_runtime_values() -> None:
    match = _match()
    summary = summarize_match(match, elapsed_seconds=0.1)

    with pytest.raises(ValueError, match="git_commit"):
        replace(_identity(), git_commit="bad")
    with pytest.raises(TypeError, match="RunIdentity"):
        build_checkpoint(
            identity=cast(RunIdentity, "bad"),
            comparison_index=0,
            match=match,
            summary=summary,
        )
    with pytest.raises(TypeError, match="matches"):
        build_raw_archive(identity=_identity(), matches=cast(tuple[MatchResult, ...], [match]))


def test_run_identity_rejects_empty_and_non_string_experiment_ids() -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        replace(_identity(), experiment_id="")
    with pytest.raises(TypeError, match="experiment_id"):
        replace(_identity(), experiment_id=cast(str, 1))


def test_public_byte_and_atomic_writers_validate_runtime_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="bytes"):
        sha256_bytes(cast(bytes, "bad"))
    with pytest.raises(TypeError, match="Path"):
        atomic_write_bytes(cast(Path, "path"), b"data")
    with pytest.raises(TypeError, match="payload"):
        atomic_write_bytes(tmp_path / "file", cast(bytes, "data"))


def test_atomic_writer_cleans_temporary_file_after_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.bin"

    def fail_replace(_source: Path, _target: Path) -> Path:
        raise OSError("injected replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_bytes(target, b"content")

    assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_builders_reject_bad_types_and_mismatched_summary() -> None:
    match = _match()
    summary = summarize_match(match, elapsed_seconds=0.1)

    with pytest.raises(TypeError, match="integer"):
        build_checkpoint(
            identity=_identity(),
            comparison_index=cast(int, True),
            match=match,
            summary=summary,
        )
    with pytest.raises(ValueError, match="non-negative"):
        build_checkpoint(
            identity=_identity(),
            comparison_index=-1,
            match=match,
            summary=summary,
        )
    with pytest.raises(TypeError, match="MatchResult"):
        build_checkpoint(
            identity=_identity(),
            comparison_index=0,
            match=cast(MatchResult, "bad"),
            summary=summary,
        )
    with pytest.raises(TypeError, match="BaselineMatchSummary"):
        build_checkpoint(
            identity=_identity(),
            comparison_index=0,
            match=match,
            summary=cast(BaselineMatchSummary, "bad"),
        )
    with pytest.raises(ValueError, match="does not match"):
        build_checkpoint(
            identity=_identity(),
            comparison_index=0,
            match=match,
            summary=replace(summary, records_sha256="0" * 64),
        )


def test_checkpoint_parser_rejects_missing_fields_bad_identity_and_pair_shape() -> None:
    match = _match()
    summary = summarize_match(match, elapsed_seconds=0.1)
    record = build_checkpoint(
        identity=_identity(),
        comparison_index=0,
        match=match,
        summary=summary,
    )

    with pytest.raises(ValueError, match="missing required"):
        parse_checkpoint(
            {},
            identity=_identity(),
            comparison_index=0,
            expected_pair=("greedy", "random"),
            expected_seed=11,
        )
    with pytest.raises(TypeError, match="RunIdentity"):
        parse_checkpoint(
            record,
            identity=cast(RunIdentity, "bad"),
            comparison_index=0,
            expected_pair=("greedy", "random"),
            expected_seed=11,
        )
    with pytest.raises(ValueError, match="two names"):
        parse_checkpoint(
            record,
            identity=_identity(),
            comparison_index=0,
            expected_pair=cast(tuple[str, str], ("greedy",)),
            expected_seed=11,
        )
    bad_elapsed = {**record, "elapsed_seconds": "slow"}
    with pytest.raises(TypeError, match="numeric"):
        parse_checkpoint(
            bad_elapsed,
            identity=_identity(),
            comparison_index=0,
            expected_pair=("greedy", "random"),
            expected_seed=11,
        )


def _gzip_json(document: object) -> bytes:
    return gzip.compress(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode(),
        mtime=0,
    )


def test_raw_archive_validates_types_schema_json_and_nonempty_matches() -> None:
    match = _match()
    with pytest.raises(TypeError, match="RunIdentity"):
        build_raw_archive(
            identity=cast(RunIdentity, "bad"),
            matches=(match,),
        )
    for matches in ((), cast(tuple[MatchResult, ...], ("bad",))):
        with pytest.raises(TypeError, match="matches"):
            build_raw_archive(identity=_identity(), matches=matches)
    with pytest.raises(TypeError, match="bytes"):
        parse_raw_archive(cast(bytes, "bad"), identity=_identity())
    with pytest.raises(TypeError, match="RunIdentity"):
        parse_raw_archive(b"data", identity=cast(RunIdentity, "bad"))
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        parse_raw_archive(gzip.compress(b"{"), identity=_identity())
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        parse_raw_archive(gzip.compress(b"\xff"), identity=_identity())

    base = {
        "schema_version": 1,
        "identity": {
            "experiment_id": _identity().experiment_id,
            "git_commit": _identity().git_commit,
            "config_sha256": _identity().config_sha256,
            "goal_sha256": _identity().goal_sha256,
        },
        "matches": [],
    }
    with pytest.raises(ValueError, match="schema_version"):
        parse_raw_archive(_gzip_json({**base, "schema_version": 2}), identity=_identity())
    with pytest.raises(TypeError, match="list"):
        parse_raw_archive(_gzip_json({**base, "matches": {}}), identity=_identity())
    with pytest.raises(ValueError, match="must not be empty"):
        parse_raw_archive(_gzip_json(base), identity=_identity())


def test_reviewed_config_remains_parseable_for_runner() -> None:
    config = load_baseline_config(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config.games_per_match == EXPECTED_GAMES
