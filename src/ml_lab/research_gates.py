"""Validate the evidence gate that must pass before the 8B research phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REQUIRED_GATES = {
    "frozen_dataset_version",
    "dataset_card",
    "dataset_validation_tests",
    "pinned_base_model_revision",
    "reproducible_baseline",
    "substantial_sft_run",
    "held_out_evaluation_suite",
    "manual_evaluation_rubric",
    "generation_sample_set",
    "preference_data_schema",
    "adapter_save_reload",
    "local_inference",
    "serving_backend",
    "credential_leak_audit",
    "train_test_contamination_audit",
    "experiment_report",
    "documented_4b_weaknesses",
}


def load_gate(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("promotion gate must be a YAML mapping")
    return value


def validate_gate(path: Path, lab: Path | None = None) -> dict[str, Any]:
    lab = (lab or path.resolve().parents[2]).resolve()
    value = load_gate(path)
    gates = value.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("promotion gate requires a gates mapping")
    missing = sorted(REQUIRED_GATES - set(gates))
    unknown = sorted(set(gates) - REQUIRED_GATES)
    results: dict[str, Any] = {}
    for name in sorted(REQUIRED_GATES):
        entry = gates.get(name)
        if not isinstance(entry, Mapping):
            results[name] = {"passed": False, "reason": "missing structured gate entry"}
            continue
        declared = entry.get("status") == "passed"
        evidence = entry.get("evidence")
        evidence_ok = False
        if isinstance(evidence, list) and evidence:
            evidence_ok = True
            for item in evidence:
                candidate = (lab / str(item)).resolve()
                if not candidate.is_relative_to(lab) or not candidate.exists():
                    evidence_ok = False
                    break
        results[name] = {
            "passed": declared and evidence_ok,
            "declared_status": entry.get("status"),
            "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        }
    passed = not missing and not unknown and all(item["passed"] for item in results.values())
    return {
        "schema_version": value.get("schema_version"),
        "primary_platform": value.get("primary_platform"),
        "missing_gates": missing,
        "unknown_gates": unknown,
        "gates": results,
        "eligible_for_8b": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", type=Path, default=Path("configs/research/4b-to-8b-gate.yaml")
    )
    args = parser.parse_args(argv)
    result = validate_gate(args.gate)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["eligible_for_8b"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

