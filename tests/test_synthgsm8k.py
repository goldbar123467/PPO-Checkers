from __future__ import annotations

from decimal import Decimal

import pytest
import torch
from datasets import Dataset

from ml_lab.synthgsm8k import (
    canonicalize_number,
    clean_solution,
    extract_final_answer,
    transform_record,
)
from ml_lab.train_sft import TokenAccountingCollator, _split_dataset


@pytest.mark.parametrize(
    ("value", "expected"),
    [(108.0, "108"), ("1,250.500", "1250.5"), (-0.25, "-0.25")],
)
def test_canonicalize_number(value: object, expected: str) -> None:
    assert canonicalize_number(value) == expected


def test_extract_final_answer_uses_only_explicit_marker() -> None:
    assert extract_final_answer("The last calculation is 99") is None
    assert extract_final_answer("work 12\n#### 1,250.5") == Decimal("1250.5")


def test_clean_solution_removes_calculation_annotations() -> None:
    assert clean_solution("Multiply <<3*4=12>> to get 12.") == "Multiply  to get 12."


def test_transform_record_uses_thinking_and_canonical_marker() -> None:
    transformed = transform_record(
        {"id": "row-1", "question": "What is 2 + 2?", "solution": "Add 2 and 2.", "answer": 4.0}
    )
    assistant = transformed["messages"][-1]["content"]
    assert assistant.startswith("<think>\n")
    assert assistant.endswith("#### 4")
    assert transformed["review_status"] == "reviewed_for_hardware_benchmark"


def test_explicit_train_validation_split_is_honored() -> None:
    dataset = Dataset.from_list(
        [
            {"text": "a", "split": "train"},
            {"text": "b", "split": "validation"},
            {"text": "c", "split": "train"},
        ]
    )
    train, validation = _split_dataset(dataset, fraction=0.9, seed=999)
    assert train["text"] == ["a", "c"]
    assert validation is not None
    assert validation["text"] == ["b"]


def test_token_accounting_counts_padding_and_supervised_tokens() -> None:
    def collate(features: object) -> dict[str, torch.Tensor]:
        del features
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "labels": torch.tensor([[-100, 2, 3], [-100, 5, -100]]),
        }

    accounting = TokenAccountingCollator(collate)
    accounting([{}, {}])
    snapshot = accounting.snapshot()
    assert snapshot["raw_input_tokens"] == 6
    assert snapshot["non_padding_tokens"] == 5
    assert snapshot["supervised_tokens"] == 3
    assert snapshot["padding_fraction"] == pytest.approx(1 / 6)
