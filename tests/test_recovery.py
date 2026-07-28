"""Audited non-destructive recovery tests for interrupted PPO metric tails."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pytest
import yaml

import checkers.recovery as recovery_module
from checkers.config import RunConfig
from checkers.recovery import (
    RECOVERY_SCHEMA,
    RecoveryError,
    analyze_recovery,
    audit_recovery_smoke,
    materialize_recovery,
    prepare_recovery,
    sha256_file,
    validate_recovery_resume_context,
)
from checkers.train import TrainingSession
from checkers.training_cli import run_training

CHECKPOINT_UPDATE = 170
CHECKPOINT_LOGGING_STEP = 187
CHECKPOINT_ELAPSED_SECONDS = 1032.16
CHECKPOINT_GLOBAL_STEP = 170
KNOWN_ORPHAN_COUNT = 2
MULTIPLE_ORPHAN_COUNT = 7
SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class SourceFixture:
    run_directory: Path
    checkpoint: Path
    config_path: Path
    metric_records: tuple[bytes, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="recovery-unit",
        seed=137,
        device="cpu",
        total_timesteps=400,
        duration_seconds=None,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        checkpoint_every=1,
        eval_every=200,
        eval_games=2,
        exploitability_train_games=2,
    )


def _metric_bytes(
    *,
    logging_step: int,
    kind: str,
    update_idx: int,
    elapsed: float,
) -> bytes:
    metric_name = "train/policy_loss" if kind == "training" else "eval/vs_random"
    value = {
        "schema": "CHECKERS_METRIC_HISTORY_1",
        "logging_step": logging_step,
        "kind": kind,
        "update_idx": update_idx,
        "global_step": update_idx,
        "elapsed_training_seconds": elapsed,
        "metrics": {metric_name: 0.0},
    }
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _changed_metric(payload: bytes, **changes: object) -> bytes:
    value = json.loads(payload)
    value.update(changes)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _source_fixture(
    tmp_path: Path,
    *,
    orphan_count: int,
    malformed_suffix: bytes = b"",
) -> SourceFixture:
    run = tmp_path / "source-run"
    run.mkdir()
    config = _config()
    config_path = run / "config.resolved.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    session = TrainingSession.create(config=config)
    session.state.update_idx = CHECKPOINT_UPDATE
    session.state.global_step = CHECKPOINT_GLOBAL_STEP
    session.state.schedule_phase = CHECKPOINT_GLOBAL_STEP / config.total_timesteps
    session.state.elapsed_training_seconds = CHECKPOINT_ELAPSED_SECONDS
    session.state.logging_step = CHECKPOINT_LOGGING_STEP
    session.state.wandb_run_id = "recovery-unit-wandb-id"
    checkpoint = run / "checkpoints" / "update-000170.pt"
    session.save_checkpoint(
        checkpoint,
        git_sha="c8207ca7dbec1001fbd5c8a22e1165bf30d8c507",
        git_dirty=False,
    )
    records: list[bytes] = []
    logging_step = 0
    for update_idx in range(1, CHECKPOINT_UPDATE + 1):
        elapsed = CHECKPOINT_ELAPSED_SECONDS * update_idx / CHECKPOINT_UPDATE
        records.append(
            _metric_bytes(
                logging_step=logging_step,
                kind="training",
                update_idx=update_idx,
                elapsed=elapsed,
            )
        )
        logging_step += 1
        if update_idx % 10 == 0:
            records.append(
                _metric_bytes(
                    logging_step=logging_step,
                    kind="periodic_evaluation",
                    update_idx=update_idx,
                    elapsed=elapsed,
                )
            )
            logging_step += 1
    assert logging_step == CHECKPOINT_LOGGING_STEP
    for offset in range(1, orphan_count + 1):
        records.append(
            _metric_bytes(
                logging_step=logging_step,
                kind="training",
                update_idx=CHECKPOINT_UPDATE + offset,
                elapsed=CHECKPOINT_ELAPSED_SECONDS + 6.0 * offset,
            )
        )
        logging_step += 1
    (run / "metrics.jsonl").write_bytes(b"".join(records) + malformed_suffix)
    evaluations = run / "evaluations"
    evaluations.mkdir()
    (evaluations / "update-000170-periodic.json").write_text(
        json.dumps({"update_idx": CHECKPOINT_UPDATE, "kind": "periodic"}) + "\n",
        encoding="utf-8",
    )
    return SourceFixture(
        run_directory=run,
        checkpoint=checkpoint,
        config_path=config_path,
        metric_records=tuple(records),
    )


def _prepare(tmp_path: Path, source: SourceFixture, name: str = "recovered") -> Path:
    destination = tmp_path / name
    result = prepare_recovery(
        repository=Path.cwd(),
        source_run_directory=source.run_directory,
        recovery_run_directory=destination,
        checkpoint_path=source.checkpoint,
        recovery_command="recover-checkers-unit",
    )
    assert result.status == "PREPARED"
    return destination


def test_known_update_170_case_preserves_orphans_and_resumes_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DISABLE_CODE", "true")
    source = _source_fixture(tmp_path, orphan_count=KNOWN_ORPHAN_COUNT)
    source_files = {
        path.relative_to(source.run_directory): _sha256(path)
        for path in source.run_directory.rglob("*")
        if path.is_file()
    }

    recovered = _prepare(tmp_path, source)

    assert {
        path.relative_to(source.run_directory): _sha256(path)
        for path in source.run_directory.rglob("*")
        if path.is_file()
    } == source_files
    assert (recovered / "metrics.jsonl").read_bytes() == b"".join(
        source.metric_records[:CHECKPOINT_LOGGING_STEP]
    )
    assert (recovered / "recovery" / "orphaned-metrics.jsonl").read_bytes() == b"".join(
        source.metric_records[CHECKPOINT_LOGGING_STEP:]
    )
    manifest = json.loads(
        (recovered / "recovery" / "recovery-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == RECOVERY_SCHEMA
    assert manifest["orphaned_metric_record_count"] == KNOWN_ORPHAN_COUNT
    assert manifest["expected_logging_step"] == CHECKPOINT_LOGGING_STEP
    assert manifest["observed_logging_step"] == CHECKPOINT_LOGGING_STEP + 1
    assert [item["line_number"] for item in manifest["orphaned_records"]] == [188, 189]
    assert manifest["checkpoint_transition_count"] == CHECKPOINT_GLOBAL_STEP
    assert manifest["checkpoint_elapsed_training_seconds"] == CHECKPOINT_ELAPSED_SECONDS
    assert manifest["source_commit"].startswith("c8207ca")
    assert manifest["current_commit"]
    assert manifest["recovery_command"] == "recover-checkers-unit"
    assert manifest["recovery_timestamp"]
    assert manifest["host_information"]
    assert manifest["cuda_device_information"]
    assert manifest["software_dependency_versions"]["torch"]
    assert isinstance(manifest["working_tree_clean"], bool)
    assert manifest["original_run_mutated"] is False
    assert manifest["final_recovery_status"] == "PREPARED"
    assert manifest["orphaned_records"][0]["byte_start"] > 0
    assert len(manifest["orphaned_records"][0]["sha256"]) == SHA256_LENGTH
    copied_paths = {entry["path"] for entry in manifest["files"]}
    assert "evaluations/update-000170-periodic.json" in copied_paths
    assert "recovery/recovery-manifest.json" in copied_paths

    result = run_training(
        config_path=source.config_path,
        output_directory=recovered,
        resume_path=recovered / "checkpoints" / source.checkpoint.name,
        max_updates=1,
    )

    assert result.start_update == CHECKPOINT_UPDATE
    assert result.end_update == CHECKPOINT_UPDATE + 1
    records = [
        json.loads(line)
        for line in (recovered / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["logging_step"] for record in records] == list(range(len(records)))
    training_updates = [record["update_idx"] for record in records if record["kind"] == "training"]
    assert training_updates == list(range(1, CHECKPOINT_UPDATE + 2))
    assert len(training_updates) == len(set(training_updates))
    latest_metrics = records[-1]["metrics"]
    assert "system/cpu_percent" in latest_metrics
    assert "system/ram_used_bytes" in latest_metrics
    invocation_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert invocation_manifest["recovery"]["checkpoint_update"] == CHECKPOINT_UPDATE
    assert invocation_manifest["wandb_run_id"] == "recovery-unit-wandb-id"
    audit = audit_recovery_smoke(run_directory=recovered, expected_updates=1)
    assert audit.status == "PASS"
    assert audit.end_update == CHECKPOINT_UPDATE + 1
    assert audit.audit_path.is_file()
    assert audit.report_path.is_file()


def test_no_orphan_case_copies_the_identical_aligned_history(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=0)
    source_payload = (source.run_directory / "metrics.jsonl").read_bytes()

    recovered = _prepare(tmp_path, source)

    assert (recovered / "metrics.jsonl").read_bytes() == source_payload
    assert (recovered / "recovery" / "orphaned-metrics.jsonl").read_bytes() == b""
    assert (source.run_directory / "metrics.jsonl").read_bytes() == source_payload
    prepared_hashes = {
        path.relative_to(recovered): _sha256(path)
        for path in recovered.rglob("*")
        if path.is_file()
    }

    repeated = _prepare(tmp_path, source)

    assert repeated == recovered
    assert {
        path.relative_to(recovered): _sha256(path)
        for path in recovered.rglob("*")
        if path.is_file()
    } == prepared_hashes


def test_multiple_orphans_are_all_preserved_after_the_unique_boundary(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=MULTIPLE_ORPHAN_COUNT)

    recovered = _prepare(tmp_path, source)

    orphaned = (recovered / "recovery" / "orphaned-metrics.jsonl").read_bytes()
    assert orphaned == b"".join(source.metric_records[CHECKPOINT_LOGGING_STEP:])
    manifest = json.loads(
        (recovered / "recovery" / "recovery-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["orphaned_metric_record_count"] == MULTIPLE_ORPHAN_COUNT
    assert manifest["checkpoint_aligned_record_boundary"]["line_number"] == CHECKPOINT_LOGGING_STEP


def test_ambiguous_history_fails_closed_without_touching_source(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=2)
    metrics_path = source.run_directory / "metrics.jsonl"
    records = metrics_path.read_bytes().splitlines(keepends=True)
    boundary = json.loads(records[CHECKPOINT_LOGGING_STEP - 1])
    boundary["elapsed_training_seconds"] += 1.0
    records[CHECKPOINT_LOGGING_STEP - 1] = (
        json.dumps(boundary, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    metrics_path.write_bytes(b"".join(records))
    source_payload = metrics_path.read_bytes()
    destination = tmp_path / "recovered"

    with pytest.raises(RecoveryError, match="unique checkpoint-aligned boundary"):
        prepare_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=destination,
            checkpoint_path=source.checkpoint,
            recovery_command="ambiguous-unit",
        )

    assert metrics_path.read_bytes() == source_payload
    assert not destination.exists()


def test_malformed_final_record_is_reported_and_preserved_in_source(tmp_path: Path) -> None:
    malformed = b'{"logging_step":189'
    source = _source_fixture(tmp_path, orphan_count=2, malformed_suffix=malformed)
    source_payload = (source.run_directory / "metrics.jsonl").read_bytes()
    destination = tmp_path / "recovered"

    with pytest.raises(RecoveryError, match="malformed JSON at line 190"):
        prepare_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=destination,
            checkpoint_path=source.checkpoint,
            recovery_command="malformed-unit",
        )

    assert source_payload.endswith(malformed)
    assert (source.run_directory / "metrics.jsonl").read_bytes() == source_payload
    assert not destination.exists()


def test_source_hash_change_after_analysis_aborts_materialization(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=2)
    destination = tmp_path / "recovered"
    plan = analyze_recovery(
        repository=Path.cwd(),
        source_run_directory=source.run_directory,
        recovery_run_directory=destination,
        checkpoint_path=source.checkpoint,
    )
    with (source.run_directory / "metrics.jsonl").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(RecoveryError, match="changed after recovery analysis"):
        materialize_recovery(plan, recovery_command="changed-unit")

    assert not destination.exists()


def test_checkpoint_hash_mismatch_aborts_before_creating_destination(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=2)
    source.checkpoint.with_suffix(".pt.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    destination = tmp_path / "recovered"

    with pytest.raises(RecoveryError, match="checkpoint validation failed"):
        prepare_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=destination,
            checkpoint_path=source.checkpoint,
            recovery_command="hash-mismatch-unit",
        )

    assert not destination.exists()


def test_interrupted_partial_output_is_detected_and_recreated(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=2)
    destination = tmp_path / "recovered"
    partial = tmp_path / ".recovered.partial"
    partial.mkdir()
    (partial / "incomplete.bin").write_bytes(b"partial")

    recovered = _prepare(tmp_path, source)

    assert recovered == destination
    assert not partial.exists()
    manifest = json.loads(
        (recovered / "recovery" / "recovery-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["partial_output_detected"] is True
    assert manifest["status"] == "PREPARED"


def test_metric_alignment_parser_rejects_every_unprovable_record_shape() -> None:
    valid = _metric_bytes(logging_step=0, kind="training", update_idx=1, elapsed=1.0)
    invalid_payloads = (
        b"",
        b"\n",
        b"[]\n",
        _changed_metric(valid, schema="BAD"),
        _changed_metric(valid, kind=""),
        _changed_metric(valid, logging_step=True),
        _changed_metric(valid, logging_step=-1),
        _changed_metric(valid, elapsed_training_seconds="bad"),
        _changed_metric(valid, elapsed_training_seconds=-1.0),
        _changed_metric(valid, metrics={}),
        _changed_metric(valid, metrics={"": 0.0}),
        _changed_metric(valid, metrics={"x": "bad"}),
        _changed_metric(valid, metrics={"x": float("inf")}),
        _changed_metric(valid, logging_step=1),
        _changed_metric(valid, global_step=2),
    )
    for payload in invalid_payloads:
        with pytest.raises(RecoveryError):
            recovery_module._parse_metric_records(payload, batch_size=1)

    duplicate = valid + _changed_metric(valid, logging_step=1)
    with pytest.raises(RecoveryError, match="duplicate logical"):
        recovery_module._parse_metric_records(duplicate, batch_size=1)
    later = _metric_bytes(logging_step=0, kind="periodic_evaluation", update_idx=2, elapsed=2.0)
    regressed = later + _metric_bytes(logging_step=1, kind="training", update_idx=1, elapsed=2.0)
    with pytest.raises(RecoveryError, match="progress regresses"):
        recovery_module._parse_metric_records(regressed, batch_size=1)
    elapsed_regression = valid + _metric_bytes(
        logging_step=1, kind="periodic_evaluation", update_idx=1, elapsed=0.5
    )
    with pytest.raises(RecoveryError, match="elapsed time regresses"):
        recovery_module._parse_metric_records(elapsed_regression, batch_size=1)
    assert recovery_module._parse_metric_records(valid.replace(b"\n", b"\r\n"), batch_size=1)


def test_recovery_path_and_evaluation_guards_fail_before_materialization(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=KNOWN_ORPHAN_COUNT)
    with pytest.raises(TypeError, match="paths"):
        analyze_recovery(
            repository=cast(Path, "bad"),
            source_run_directory=source.run_directory,
            recovery_run_directory=tmp_path / "recovered",
            checkpoint_path=source.checkpoint,
        )
    with pytest.raises(RecoveryError, match="must differ"):
        analyze_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=source.run_directory,
            checkpoint_path=source.checkpoint,
        )
    with pytest.raises(RecoveryError, match="sibling"):
        analyze_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=source.run_directory / "nested",
            checkpoint_path=source.checkpoint,
        )
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a checkpoint")
    with pytest.raises(RecoveryError, match="beneath the source run"):
        analyze_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=tmp_path / "recovered",
            checkpoint_path=outside,
        )
    evaluation = source.run_directory / "evaluations" / "update-000170-periodic.json"
    evaluation.write_text("{", encoding="utf-8")
    with pytest.raises(RecoveryError, match="evaluation is malformed"):
        analyze_recovery(
            repository=Path.cwd(),
            source_run_directory=source.run_directory,
            recovery_run_directory=tmp_path / "recovered",
            checkpoint_path=source.checkpoint,
        )


def test_recovery_resume_context_rechecks_commit_cleanliness_and_paths(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path, orphan_count=KNOWN_ORPHAN_COUNT)
    recovered = _prepare(tmp_path, source)
    manifest = json.loads(
        (recovered / "recovery" / "recovery-manifest.json").read_text(encoding="utf-8")
    )
    commit = cast(str, manifest["current_commit"])
    clean = cast(bool, manifest["working_tree_clean"])
    checkpoint = recovered / "checkpoints" / source.checkpoint.name

    context = validate_recovery_resume_context(
        output_directory=recovered,
        resume_path=checkpoint,
        current_commit=commit,
        working_tree_clean=clean,
    )

    assert context is not None
    assert context.wandb_summary()["recovery/is_recovery"] is True
    with pytest.raises(RecoveryError, match="explicit checkpoint"):
        validate_recovery_resume_context(
            output_directory=recovered,
            resume_path=None,
            current_commit=commit,
            working_tree_clean=clean,
        )
    with pytest.raises(RecoveryError, match="source commit"):
        validate_recovery_resume_context(
            output_directory=recovered,
            resume_path=checkpoint,
            current_commit="different",
            working_tree_clean=clean,
        )
    with pytest.raises(RecoveryError, match="cleanliness"):
        validate_recovery_resume_context(
            output_directory=recovered,
            resume_path=checkpoint,
            current_commit=commit,
            working_tree_clean=not clean,
        )
    outside = tmp_path / "outside-resume.pt"
    outside.write_bytes(b"outside")
    with pytest.raises(RecoveryError, match="checkpoint directory"):
        validate_recovery_resume_context(
            output_directory=recovered,
            resume_path=outside,
            current_commit=commit,
            working_tree_clean=clean,
        )


def test_sha256_file_rejects_non_paths_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Path"):
        sha256_file(cast(Path, "bad"))
    with pytest.raises(RecoveryError, match="not a file"):
        sha256_file(tmp_path / "missing")
