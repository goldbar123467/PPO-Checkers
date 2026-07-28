"""Prepare pinned SynthGSM8K training subsets and the untouched GSM8K benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .run_metadata import atomic_write_json, sha256_file, utc_now

SYNTH_DATASET_ID = "clarkkitchen22/SynthGSM8K-50K"
SYNTH_DATASET_REVISION = "ebf8f270d82680fc8b31c15bd1535eafa972da07"
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
QWEN_MODEL_ID = "Qwen/Qwen3-4B"
QWEN_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

CALCULATION_ANNOTATION_RE = re.compile(r"<<[^<>]*>>")
FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)")


def canonicalize_number(value: object) -> str:
    """Return a stable decimal representation without unnecessary trailing zeroes."""
    try:
        number = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric answer: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"answer must be finite: {value!r}")
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def extract_final_answer(text: str) -> Decimal | None:
    """Parse only the last explicit GSM8K final-answer marker."""
    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def clean_solution(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("solution must be a non-empty string")
    cleaned = CALCULATION_ANNOTATION_RE.sub("", value)
    cleaned = "\n".join(line.rstrip() for line in cleaned.strip().splitlines())
    if not cleaned:
        raise ValueError("solution became empty after annotation removal")
    return cleaned


def transform_record(
    record: Mapping[str, Any], *, source_row_index: int | None = None
) -> dict[str, Any]:
    identifier = record.get("id")
    question = record.get("question")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("record requires a non-empty string id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"record {identifier}: question must be non-empty")
    solution = clean_solution(record.get("solution"))
    answer = canonicalize_number(record.get("answer"))
    completion = f"<think>\n{solution}\n</think>\n\n#### {answer}"
    source_record_key = (
        f"{source_row_index:06d}:{identifier}"
        if source_row_index is not None
        else f"unindexed:{identifier}"
    )
    content_digest = hashlib.sha256(
        f"{question.strip()}\0{solution}\0{answer}".encode()
    ).hexdigest()
    return {
        "messages": [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": completion},
        ],
        "source_id": identifier,
        "source_row_index": source_row_index,
        "source_record_key": source_record_key,
        "expected_answer": answer,
        "source_provenance": f"hf://datasets/{SYNTH_DATASET_ID}@{SYNTH_DATASET_REVISION}",
        "license": "mit",
        "author": "clarkkitchen22; synthetic generation attributed in dataset card",
        "creation_method": (
            "synthetic generation plus source filtering; deterministic local transform"
        ),
        "review_status": "reviewed_for_hardware_benchmark",
        "grade_band": "grade-school math; dataset-provided scope",
        "safety_category": "general",
        "subject_category": "mathematics",
        "difficulty": "dataset-unspecified",
        "duplicate_group": content_digest,
        "split": "unassigned",
        "quality_flags": {
            "well_posed": None,
            "answer_matches_question": None,
            "discrete_quantity_valid": None,
            "unit_consistent": None,
            "rounding_consistent": None,
            "real_world_assumptions_valid": None,
        },
    }


def _selection_key(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{identifier}".encode()).hexdigest()


def _ordered_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _token_length(tokenizer: Any, record: Mapping[str, Any]) -> int:
    token_ids = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return len(token_ids)


def prepare_training_subset(
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if train_count <= 0 or validation_count < 0 or max_length <= 0:
        raise ValueError(
            "train_count/max_length must be positive and validation_count non-negative"
        )
    dataset = load_dataset(
        SYNTH_DATASET_ID,
        revision=SYNTH_DATASET_REVISION,
        split="train",
    )
    expected_columns = {"id", "question", "solution", "answer"}
    if set(dataset.column_names) != expected_columns:
        raise RuntimeError(
            f"unexpected source schema: {sorted(dataset.column_names)}; "
            f"expected {sorted(expected_columns)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, revision=QWEN_MODEL_REVISION)
    identifiers = [str(record["id"]) for record in dataset]
    identifier_counts: dict[str, int] = {}
    for identifier in identifiers:
        identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1
    duplicate_identifiers = {
        identifier: count for identifier, count in identifier_counts.items() if count > 1
    }

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_row_index, source in enumerate(dataset):
        try:
            transformed = transform_record(source, source_row_index=source_row_index)
            length = _token_length(tokenizer, transformed)
            if length > max_length:
                rejected.append(
                    {
                        "source_id": transformed["source_id"],
                        "reason": "over_max_length",
                        "tokens": length,
                    }
                )
                continue
            transformed["token_count"] = length
            accepted.append(transformed)
        except (TypeError, ValueError) as exc:
            rejected.append(
                {"source_id": str(source.get("id", "unknown")), "reason": type(exc).__name__}
            )
    accepted.sort(key=lambda item: _selection_key(str(item["source_record_key"]), seed))
    needed = train_count + validation_count
    if len(accepted) < needed:
        raise RuntimeError(f"only {len(accepted)} accepted records; {needed} requested")
    selected = accepted[:needed]
    train = selected[:train_count]
    validation = selected[train_count:]
    for record in train:
        record["split"] = "train"
    for record in validation:
        record["split"] = "validation"

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    combined_path = output_dir / "dataset.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    _write_jsonl(combined_path, [*train, *validation])
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "purpose": "hardware benchmark and controlled learning test; not a classroom dataset",
        "source": {
            "dataset": SYNTH_DATASET_ID,
            "revision": SYNTH_DATASET_REVISION,
            "split": "train",
            "record_count": len(dataset),
            "columns": sorted(dataset.column_names),
            "unique_id_count": len(identifier_counts),
            "duplicate_id_value_count": len(duplicate_identifiers),
            "rows_in_duplicate_id_groups": sum(duplicate_identifiers.values()),
            "max_id_multiplicity": max(duplicate_identifiers.values(), default=1),
            "row_identity": "zero-based source row index plus source id",
            "ordered_row_identity_sha256": hashlib.sha256(
                "\n".join(
                    f"{index:06d}:{identifier}" for index, identifier in enumerate(identifiers)
                ).encode("utf-8")
            ).hexdigest(),
        },
        "tokenizer": {
            "model": QWEN_MODEL_ID,
            "revision": QWEN_MODEL_REVISION,
            "class": type(tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(
                (tokenizer.chat_template or "").encode("utf-8")
            ).hexdigest(),
            "enable_thinking": True,
            "max_length": max_length,
            "overlength_policy": "exclude; never truncate",
        },
        "selection": {
            "algorithm": "sort by sha256(seed + NUL + source_row_index:source_id)",
            "seed": seed,
            "train_count": len(train),
            "validation_count": len(validation),
            "accepted_before_selection": len(accepted),
            "rejected_count": len(rejected),
            "rejection_reasons": {
                reason: sum(1 for item in rejected if item["reason"] == reason)
                for reason in sorted({str(item["reason"]) for item in rejected})
            },
        },
        "format": {
            "assistant_only": True,
            "completion_shape": "<think>solution</think> followed by #### canonical_answer",
            "calculation_annotations_removed": True,
            "semantic_quality_flags": "unknown pending separate v2-clean review",
        },
        "artifacts": {
            "train": {
                "path": str(train_path.resolve()),
                "sha256": sha256_file(train_path),
                "ordered_records_sha256": _ordered_digest(train),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "sha256": sha256_file(validation_path),
                "ordered_records_sha256": _ordered_digest(validation),
            },
            "combined": {
                "path": str(combined_path.resolve()),
                "sha256": sha256_file(combined_path),
                "ordered_records_sha256": _ordered_digest([*train, *validation]),
            },
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def prepare_benchmark(output_dir: Path) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        GSM8K_DATASET_ID,
        "main",
        revision=GSM8K_DATASET_REVISION,
        split="test",
    )
    records: list[dict[str, Any]] = []
    for index, source in enumerate(dataset):
        raw_answer = str(source["answer"])
        parsed = extract_final_answer(raw_answer)
        if parsed is None:
            raise RuntimeError(f"GSM8K test record {index} has no canonical answer marker")
        records.append(
            {
                "id": f"gsm8k-test-{index:04d}",
                "question": str(source["question"]),
                "expected_answer": canonicalize_number(parsed),
                "source_answer_sha256": hashlib.sha256(raw_answer.encode()).hexdigest(),
            }
        )
    output = output_dir / "test.jsonl"
    _write_jsonl(output, records)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "benchmark": GSM8K_DATASET_ID,
        "configuration": "main",
        "split": "test",
        "revision": GSM8K_DATASET_REVISION,
        "record_count": len(records),
        "training_use_forbidden": True,
        "artifact": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "ordered_records_sha256": _ordered_digest(records),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("prepare-training")
    training.add_argument("--output-dir", required=True, type=Path)
    training.add_argument("--train-count", required=True, type=int)
    training.add_argument("--validation-count", type=int, default=0)
    training.add_argument("--max-length", type=int, default=1024)
    training.add_argument("--seed", type=int, default=42)
    benchmark = subparsers.add_parser("prepare-benchmark")
    benchmark.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare-training":
        result = prepare_training_subset(
            args.output_dir,
            train_count=args.train_count,
            validation_count=args.validation_count,
            max_length=args.max_length,
            seed=args.seed,
        )
    else:
        result = prepare_benchmark(args.output_dir)
    summary = {
        "created_at": result["created_at"],
        "manifest": str((args.output_dir / "manifest.json").resolve()),
    }
    if "selection" in result:
        summary.update(result["selection"])
    else:
        summary["record_count"] = result["record_count"]
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
