"""Comparative evaluation with separate automated, model, and human-grade fields."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .data_validation import read_json_records
from .inference import generate, load_model
from .run_metadata import append_jsonl, atomic_write_json, make_run_id, utc_now

ALLOWED_CANDIDATE_ROLES = {
    "base_4b",
    "sft_4b",
    "preference_4b",
    "base_8b",
    "sft_8b",
    "preference_8b",
}


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    role: str
    model: str
    revision: str
    adapter: str | None = None
    load_in_4bit: bool = True


def load_evaluation_config(path: Path) -> tuple[list[Candidate], str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("evaluation config must be a mapping")
    prompts = value.get("prompts")
    if not isinstance(prompts, str):
        raise ValueError("evaluation config requires prompts path")
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("evaluation config requires at least one candidate")
    candidates: list[Candidate] = []
    names: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate must be a mapping")
        candidate = Candidate(**dict(raw))
        if candidate.role not in ALLOWED_CANDIDATE_ROLES:
            raise ValueError(f"invalid candidate role: {candidate.role}")
        if candidate.name in names:
            raise ValueError(f"duplicate candidate name: {candidate.name}")
        if candidate.revision in {"", "main"}:
            raise ValueError(f"candidate {candidate.name} requires a pinned revision")
        names.add(candidate.name)
        candidates.append(candidate)
    grader = value.get("model_grader")
    if grader and any(grader == candidate.name for candidate in candidates):
        raise ValueError("a candidate model cannot be its own model grader")
    options = {
        "max_new_tokens": int(value.get("max_new_tokens", 256)),
        "temperature": float(value.get("temperature", 0.0)),
        "model_grader": grader,
        "rubric_version": str(value.get("rubric_version", "unversioned")),
    }
    return candidates, prompts, options


def validate_prompt(record: Mapping[str, Any], index: int) -> None:
    required = {"id", "prompt", "category", "split"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"evaluation record {index} missing: {', '.join(missing)}")
    if record["split"] not in {"validation", "test"}:
        raise ValueError(f"evaluation record {index} must be held-out validation or test")
    if not isinstance(record["prompt"], str) or not record["prompt"].strip():
        raise ValueError(f"evaluation record {index} has empty prompt")


def automated_grade(record: Mapping[str, Any], response: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected = record.get("expected_contains", [])
    if expected:
        checks["expected_contains"] = all(str(item).casefold() in response.casefold() for item in expected)
    forbidden = record.get("forbidden_contains", [])
    if forbidden:
        checks["forbidden_absent"] = all(str(item).casefold() not in response.casefold() for item in forbidden)
    pattern = record.get("expected_regex")
    if pattern:
        checks["regex"] = re.search(str(pattern), response, flags=re.IGNORECASE | re.MULTILINE) is not None
    max_words = record.get("max_words")
    if max_words is not None:
        checks["length"] = len(response.split()) <= int(max_words)
    return {
        "checks": checks,
        "passed": all(checks.values()) if checks else None,
        "response_words": len(response.split()),
    }


def _extract_python(response: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response.strip()


def execute_code_test(response: str, test_code: str, timeout: int = 5) -> dict[str, Any]:
    """Run an explicitly opted-in generated-code test in an isolated interpreter.

    Python ``-I`` and a timeout reduce ambient state, but this is not a security
    sandbox. Callers must use only trusted evaluation assets on a trusted machine.
    """
    source = _extract_python(response) + "\n\n" + test_code + "\n"
    with tempfile.TemporaryDirectory(prefix="ml-lab-eval-") as directory:
        path = Path(directory) / "candidate_test.py"
        path.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "executed": True,
                "passed": completed.returncode == 0,
                "exit_code": completed.returncode,
                # Do not persist generated stdout/stderr; arbitrary model output can
                # accidentally reproduce sensitive prompt content.
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
            }
        except subprocess.TimeoutExpired:
            return {"executed": True, "passed": False, "timed_out": True}


def evaluate(
    config_path: Path,
    *,
    execute_code: bool = False,
) -> Path:
    candidates, prompt_path, options = load_evaluation_config(config_path)
    prompts = read_json_records(prompt_path)
    for index, prompt in enumerate(prompts):
        validate_prompt(prompt, index)
    run_id = make_run_id("evaluation")
    lab = Path(__file__).resolve().parents[2]
    output = lab / "runs" / run_id
    output.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "created_at": utc_now(),
        "evaluation_config": str(config_path.resolve()),
        "candidate_count": len(candidates),
        "prompt_count": len(prompts),
        "rubric_version": options["rubric_version"],
        "grade_channels": ["automated", "model_based", "human"],
        "model_grader": options["model_grader"],
        "warning": "Model-based grades are never the sole evidence; human grades remain independent.",
    }
    atomic_write_json(output / "metadata.json", metadata)
    results_path = output / "results.jsonl"
    for candidate in candidates:
        model, tokenizer = load_model(
            candidate.model,
            adapter=candidate.adapter,
            revision=candidate.revision,
            load_in_4bit=candidate.load_in_4bit,
        )
        for prompt in prompts:
            response, performance = generate(
                model,
                tokenizer,
                prompt["prompt"],
                max_new_tokens=options["max_new_tokens"],
                temperature=options["temperature"],
            )
            automated = automated_grade(prompt, response)
            if execute_code and prompt.get("code_test"):
                automated["code_test"] = execute_code_test(response, str(prompt["code_test"]))
            result = {
                "candidate": candidate.name,
                "candidate_role": candidate.role,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "response": response,
                "performance": performance,
                "automated_grade": automated,
                "model_based_grade": None,
                "human_grade": {
                    "grader": None,
                    "rubric_version": options["rubric_version"],
                    "scores": None,
                    "teacher_preference": None,
                    "notes": None,
                },
            }
            append_jsonl(results_path, result)
        del model
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute-code-tests", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    candidates, prompts, options = load_evaluation_config(args.config)
    records = read_json_records(prompts)
    for index, record in enumerate(records):
        validate_prompt(record, index)
    if args.validate_only:
        print(json.dumps({"valid": True, "candidates": len(candidates), "prompts": len(records), **options}))
        return 0
    output = evaluate(args.config, execute_code=args.execute_code_tests)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

