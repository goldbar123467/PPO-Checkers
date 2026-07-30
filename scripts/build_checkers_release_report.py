#!/usr/bin/env python3
"""Build a compact, deterministic release report from authoritative run artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

from checkers.eval.baseline_run import atomic_write_bytes
from checkers.web.policy_bundle import load_policy_bundle, sha256_file

CHECKPOINT_PATTERN = re.compile(r"update-(\d{6})\.pt$")
EXPECTED_EVALUATION_SCHEMA = "CHECKERS_PRACTICE_EVALUATION_1"
EXPECTED_MANIFEST_SCHEMA = "CHECKERS_TRAINING_RUN_1"
SCORE_TOLERANCE = 1e-12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _evaluation(path: Path) -> dict[str, Any]:
    value = _json_object(path)
    if value.get("schema") != EXPECTED_EVALUATION_SCHEMA:
        raise ValueError(f"unsupported evaluation schema in {path}")
    rows = value.get("game_rows")
    metrics = value.get("metrics")
    if not isinstance(rows, list) or not isinstance(metrics, dict):
        raise ValueError(f"malformed evaluation in {path}")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _match_summary(evaluation: dict[str, Any], match: str) -> dict[str, object]:
    rows = evaluation["game_rows"]
    selected = [row for row in rows if isinstance(row, dict) and row.get("match") == match]
    outcomes = Counter(str(row.get("perspective_result")) for row in selected)
    games = len(selected)
    if games < 1 or set(outcomes) - {"win", "draw", "loss"}:
        raise ValueError(f"invalid {match} game rows")
    score = (outcomes["win"] + 0.5 * outcomes["draw"]) / games
    metric_name = f"eval/{match}"
    recorded = float(evaluation["metrics"][metric_name])
    if abs(score - recorded) > SCORE_TOLERANCE:
        raise ValueError(f"{match} rows disagree with the recorded score")
    return {
        "games": games,
        "wins": outcomes["win"],
        "draws": outcomes["draw"],
        "losses": outcomes["loss"],
        "score": recorded,
        "ci95_low": float(evaluation["metrics"][f"{metric_name}_ci_low"]),
        "ci95_high": float(evaluation["metrics"][f"{metric_name}_ci_high"]),
    }


def _evaluated_checkpoints(run_dir: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for checkpoint in sorted((run_dir / "checkpoints").glob("update-*.pt")):
        match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if match is None:
            continue
        update = int(match.group(1))
        candidates = (
            run_dir / "evaluations" / f"update-{update:06d}-periodic.json",
            run_dir / "evaluations" / f"update-{update:06d}-final.json",
            run_dir / "evaluations" / f"update-{update:06d}-approval_gate.json",
        )
        evaluation_path = next((path for path in candidates if path.is_file()), None)
        if evaluation_path is None:
            continue
        evaluation = _evaluation(evaluation_path)
        result.append(
            {
                "update": update,
                "vs_minimax2_score": float(evaluation["metrics"]["eval/vs_minimax2"]),
                "evaluation_sha256": sha256_file(evaluation_path),
            }
        )
    if not result:
        raise ValueError("run has no evaluated persisted checkpoints")
    return result


def _resource_summary(metrics_path: Path) -> dict[str, object]:
    keys = (
        "system/gpu_memory_used_mib",
        "system/gpu_power_draw_watts",
        "system/gpu_temperature_celsius",
        "system/gpu_utilization_percent",
        "system/process_rss_bytes",
        "charts/SPS",
    )
    values: dict[str, list[float]] = {key: [] for key in keys}
    training_rows = 0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "training":
            continue
        training_rows += 1
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("training metric record has no metric mapping")
        for key in keys:
            if key in metrics:
                values[key].append(float(metrics[key]))
    if training_rows < 1 or any(len(series) != training_rows for series in values.values()):
        raise ValueError("resource metric history is incomplete")
    return {
        "training_rows": training_rows,
        "gpu_memory_used_mib": {
            "mean": statistics.fmean(values["system/gpu_memory_used_mib"]),
            "max": max(values["system/gpu_memory_used_mib"]),
        },
        "gpu_power_draw_watts": {
            "mean": statistics.fmean(values["system/gpu_power_draw_watts"]),
            "max": max(values["system/gpu_power_draw_watts"]),
        },
        "gpu_temperature_celsius": {
            "mean": statistics.fmean(values["system/gpu_temperature_celsius"]),
            "max": max(values["system/gpu_temperature_celsius"]),
        },
        "gpu_utilization_percent": {
            "mean": statistics.fmean(values["system/gpu_utilization_percent"]),
            "max": max(values["system/gpu_utilization_percent"]),
        },
        "process_rss_bytes": {
            "mean": statistics.fmean(values["system/process_rss_bytes"]),
            "max": max(values["system/process_rss_bytes"]),
        },
        "steps_per_second": {
            "mean": statistics.fmean(values["charts/SPS"]),
            "min": min(values["charts/SPS"]),
            "max": max(values["charts/SPS"]),
        },
    }


def main() -> int:
    """Validate source evidence and write its deliberately compact public projection."""

    args = _arguments()
    run_dir = args.run_dir.resolve()
    bundle = load_policy_bundle(args.bundle.resolve())
    manifest_path = run_dir / "manifest-006144.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("practice run manifest is not a completed supported run")

    selected_path = (
        run_dir / "evaluations" / f"update-{bundle.metadata.update_idx:06d}-periodic.json"
    )
    final_update = _integer(manifest["end_update"], name="manifest end_update")
    final_path = run_dir / "evaluations" / f"update-{final_update:06d}-final.json"
    selected = _evaluation(selected_path)
    final = _evaluation(final_path)
    candidates = _evaluated_checkpoints(run_dir)
    best = max(
        candidates,
        key=lambda row: _number(row["vs_minimax2_score"], name="candidate score"),
    )
    if _integer(best["update"], name="candidate update") != bundle.metadata.update_idx:
        raise ValueError("bundle is not the highest-scoring evaluated persisted checkpoint")
    if bundle.metadata.source_checkpoint_sha256 != sha256_file(
        run_dir / "checkpoints" / f"update-{bundle.metadata.update_idx:06d}.pt"
    ):
        raise ValueError("bundle checkpoint digest does not match the source checkpoint")

    approval_manifest_path = run_dir / "manifest-001024.json"
    approval_manifest = _json_object(approval_manifest_path)
    invocation_wall_seconds = float(approval_manifest["wall_seconds"]) + float(
        manifest["wall_seconds"]
    )
    report = {
        "schema": "CHECKERS_PUBLIC_RELEASE_REPORT_1",
        "experiment": {
            "id": manifest["experiment_id"],
            "seed": manifest["seed"],
            "source_git_sha": manifest["git_sha"],
            "device": manifest["device"],
            "deterministic": manifest["deterministic"],
            "final_update": manifest["end_update"],
            "global_step": manifest["global_step"],
            "measured_training_seconds": manifest["elapsed_training_seconds"],
            "invocation_wall_seconds": invocation_wall_seconds,
        },
        "selection": {
            "criterion": (
                "highest vs_minimax2 score among persisted checkpoints with a full ballot "
                "evaluation"
            ),
            "selected_update": bundle.metadata.update_idx,
            "evaluated_persisted_checkpoints": candidates,
            "selection_bias_warning": (
                "The ballot suite was reused for checkpoint selection; this is not sealed test "
                "evidence."
            ),
        },
        "selected_evaluation": {
            "ballot_count": selected["ballot_count"],
            "ballot_sha256": selected["ballot_sha256"],
            "vs_random": _match_summary(selected, "vs_random"),
            "vs_minimax2": _match_summary(selected, "vs_minimax2"),
            "evaluation_sha256": sha256_file(selected_path),
        },
        "final_evaluation": {
            "update": final["update_idx"],
            "vs_random": _match_summary(final, "vs_random"),
            "vs_minimax2": _match_summary(final, "vs_minimax2"),
            "evaluation_sha256": sha256_file(final_path),
        },
        "bundle": {
            "id": bundle.metadata.bundle_id,
            "sha256": bundle.sha256,
            "size_bytes": bundle.size_bytes,
            "parameter_count": sum(parameter.numel() for parameter in bundle.network.parameters()),
            "source_checkpoint_sha256": bundle.metadata.source_checkpoint_sha256,
            "source_checkpoint_size_bytes": bundle.metadata.source_checkpoint_size_bytes,
            "config_sha256": bundle.metadata.config_sha256,
            "parity_validation_positions": 12,
        },
        "resources": _resource_summary(run_dir / "metrics.jsonl"),
        "source_evidence": {
            "manifest_sha256": sha256_file(manifest_path),
            "approval_manifest_sha256": sha256_file(approval_manifest_path),
            "resolved_config_sha256": sha256_file(run_dir / "config.resolved.yaml"),
            "metrics_history_sha256": sha256_file(run_dir / "metrics.jsonl"),
        },
        "limitations": [
            (
                "One practice-run seed is reported; no between-seed uncertainty is available for "
                "this checkpoint."
            ),
            (
                "Minimax-2 is a shallow project baseline, not an external rating or "
                "expert-strength reference."
            ),
            (
                "The selected checkpoint has not been evaluated on a sealed external suite or "
                "against humans."
            ),
        ],
    }
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(args.output, payload)
    print(f"report={args.output}")
    print(f"sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
