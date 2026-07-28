"""Dataset linting, deterministic splitting, and safe ChatML rendering."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
REQUIRED_PROVENANCE_FIELDS = {
    "source_provenance",
    "license",
    "author",
    "creation_method",
    "review_status",
    "grade_band",
    "safety_category",
    "subject_category",
    "difficulty",
    "duplicate_group",
    "split",
}
REQUIRED_PREFERENCE_FIELDS = {
    "prompt",
    "system_policy",
    "chosen_response",
    "rejected_response",
    "human_grader",
    "rubric_version",
    "category_scores",
    "overall_preference",
    "rationale",
    "safety_flags",
    "timestamp",
    "chosen_model_version",
    "rejected_model_version",
}
REVIEWED_STATUSES = {"reviewed", "approved", "reviewed_for_hardware_benchmark"}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    index: int | None
    code: str
    message: str
    severity: str = "error"


@dataclass(slots=True)
class ValidationReport:
    records: int = 0
    valid_records: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    duplicate_groups: dict[str, list[int]] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "valid_records": self.valid_records,
            "ok": self.ok,
            "issues": [asdict(issue) for issue in self.issues],
            "duplicate_groups": self.duplicate_groups,
        }


def canonical_record_text(record: Mapping[str, Any]) -> str:
    text = record.get("text")
    if isinstance(text, str):
        return re.sub(r"\s+", " ", text).strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        pieces = []
        for message in messages:
            if isinstance(message, Mapping):
                pieces.append(f"{message.get('role', '')}:{message.get('content', '')}")
        return "\n".join(pieces).strip()
    return ""


def content_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_record_text(record).encode("utf-8")).hexdigest()


def render_chatml(
    messages: Sequence[Mapping[str, Any]], add_generation_prompt: bool = False
) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def validate_record(
    record: Any,
    index: int,
    *,
    require_reviewed: bool = False,
    require_provenance: bool = False,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(record, Mapping):
        return [ValidationIssue(index, "record_type", "record must be a JSON object")]
    has_text = "text" in record
    has_messages = "messages" in record
    if has_text == has_messages:
        issues.append(
            ValidationIssue(
                index, "representation", "record must contain exactly one of text or messages"
            )
        )
    if has_text and (not isinstance(record.get("text"), str) or not record.get("text", "").strip()):
        issues.append(ValidationIssue(index, "empty_text", "text must be a non-empty string"))
    if has_messages:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            issues.append(
                ValidationIssue(index, "messages_type", "messages must be a non-empty list")
            )
        else:
            assistant_seen = False
            previous_role: str | None = None
            for message_index, message in enumerate(messages):
                if not isinstance(message, Mapping):
                    issues.append(
                        ValidationIssue(
                            index, "message_type", f"message {message_index} must be an object"
                        )
                    )
                    continue
                role = message.get("role")
                content = message.get("content")
                if role not in ALLOWED_ROLES:
                    issues.append(
                        ValidationIssue(
                            index, "message_role", f"message {message_index} has invalid role"
                        )
                    )
                if not isinstance(content, str) or not content.strip():
                    issues.append(
                        ValidationIssue(
                            index, "message_content", f"message {message_index} has empty content"
                        )
                    )
                if role == "system" and message_index != 0:
                    issues.append(
                        ValidationIssue(
                            index, "system_position", "system message should be first", "warning"
                        )
                    )
                if role == "assistant":
                    assistant_seen = True
                if role == previous_role and role in {"user", "assistant"}:
                    issues.append(
                        ValidationIssue(
                            index,
                            "role_alternation",
                            f"consecutive {role} messages at {message_index}",
                            "warning",
                        )
                    )
                previous_role = str(role)
            if not assistant_seen:
                issues.append(
                    ValidationIssue(index, "missing_assistant", "no assistant response present")
                )
    if require_provenance:
        missing = sorted(field for field in REQUIRED_PROVENANCE_FIELDS if not record.get(field))
        if missing:
            issues.append(
                ValidationIssue(
                    index, "missing_provenance", f"missing fields: {', '.join(missing)}"
                )
            )
    if require_reviewed and record.get("review_status") not in REVIEWED_STATUSES:
        issues.append(ValidationIssue(index, "unreviewed", "record is not reviewed or approved"))
    return issues


def validate_records(
    records: Sequence[Any],
    *,
    require_reviewed: bool = False,
    require_provenance: bool = False,
    detect_duplicates: bool = True,
) -> ValidationReport:
    report = ValidationReport(records=len(records))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        record_issues = validate_record(
            record,
            index,
            require_reviewed=require_reviewed,
            require_provenance=require_provenance,
        )
        report.issues.extend(record_issues)
        if not any(issue.severity == "error" for issue in record_issues):
            report.valid_records += 1
        if detect_duplicates and isinstance(record, Mapping):
            digest = content_hash(record)
            if canonical_record_text(record):
                groups[digest].append(index)
    report.duplicate_groups = {
        digest: indexes for digest, indexes in groups.items() if len(indexes) > 1
    }
    for indexes in report.duplicate_groups.values():
        report.issues.append(
            ValidationIssue(
                indexes[0], "duplicate", f"duplicate content at records {indexes}", "warning"
            )
        )
    return report


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number}: expected JSON object")
                records.append(value)
        return records
    if source.suffix.lower() == ".json":
        with source.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            value = value["data"]
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{source}: expected a JSON array of objects")
        return value
    raise ValueError(f"unsupported local dataset extension: {source.suffix}")


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_json_records(path)


def deterministic_split_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, list[int]]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    indexes = list(range(len(records)))
    random.Random(seed).shuffle(indexes)
    validation_count = round(len(indexes) * validation_fraction)
    test_count = round(len(indexes) * test_fraction)
    # Preserve at least one training record for any non-empty dataset.
    overflow = max(0, validation_count + test_count - max(0, len(indexes) - 1))
    validation_count = max(0, validation_count - overflow)
    validation = sorted(indexes[:validation_count])
    test = sorted(indexes[validation_count : validation_count + test_count])
    train = sorted(indexes[validation_count + test_count :])
    return {"train": train, "validation": validation, "test": test}


def check_split_contamination(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ValidationIssue]:
    seen: dict[str, set[str]] = defaultdict(set)
    for split_name, records in splits.items():
        for record in records:
            seen[content_hash(record)].add(split_name)
    return [
        ValidationIssue(None, "split_contamination", f"content appears in splits: {sorted(names)}")
        for names in seen.values()
        if len(names) > 1
    ]


def dataset_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = [content_hash(record) for record in records]
    split_counts = Counter(str(record.get("split", "unspecified")) for record in records)
    aggregate = hashlib.sha256("\n".join(sorted(hashes)).encode("ascii")).hexdigest()
    return {
        "record_count": len(records),
        "content_sha256": aggregate,
        "unique_content_count": len(set(hashes)),
        "split_counts": dict(sorted(split_counts.items())),
    }


def validate_preference_record(record: Any, index: int = 0) -> list[ValidationIssue]:
    """Validate rich offline-preference data; unexplained scalar ratings are rejected."""
    if not isinstance(record, Mapping):
        return [ValidationIssue(index, "preference_type", "preference record must be an object")]
    issues: list[ValidationIssue] = []
    missing = sorted(field for field in REQUIRED_PREFERENCE_FIELDS if field not in record)
    if missing:
        issues.append(
            ValidationIssue(index, "preference_fields", f"missing fields: {', '.join(missing)}")
        )
    for name in ("prompt", "chosen_response", "rejected_response", "rationale"):
        if name in record and (not isinstance(record[name], str) or not record[name].strip()):
            issues.append(ValidationIssue(index, f"empty_{name}", f"{name} must be non-empty"))
    if record.get("chosen_response") == record.get("rejected_response"):
        issues.append(
            ValidationIssue(
                index, "identical_preference", "chosen and rejected responses are identical"
            )
        )
    if "rationale" in record and len(str(record.get("rationale", "")).split()) < 3:
        issues.append(ValidationIssue(index, "thin_rationale", "preference rationale is too short"))
    scores = record.get("category_scores")
    if scores is not None and not isinstance(scores, Mapping):
        issues.append(
            ValidationIssue(index, "category_scores", "category_scores must be an object")
        )
    if "safety_flags" in record and not isinstance(record.get("safety_flags"), list):
        issues.append(ValidationIssue(index, "safety_flags", "safety_flags must be a list"))
    return issues
