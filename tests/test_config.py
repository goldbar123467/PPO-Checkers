from __future__ import annotations

import pytest

from ml_lab.config import TrainingConfig


def minimal_config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "model_name_or_path": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "dataset": "data/processed/example.jsonl",
    }
    values.update(overrides)
    return TrainingConfig.from_mapping(values)


def test_wandb_reporting_requires_project() -> None:
    with pytest.raises(ValueError, match="wandb_project"):
        minimal_config(report_to=["tensorboard", "wandb"])


def test_wandb_and_dataset_revisions_are_preserved() -> None:
    config = minimal_config(
        dataset_revision="ebf8f270d82680fc8b31c15bd1535eafa972da07",
        report_to=["tensorboard", "wandb"],
        wandb_project="qwen3-4b-synthgsm8k-5070",
        chat_template_enable_thinking=True,
        warmup_steps=10,
        warmup_ratio=0.0,
    )
    assert config.dataset_revision == "ebf8f270d82680fc8b31c15bd1535eafa972da07"
    assert config.chat_template_enable_thinking is True
    assert config.warmup_steps == 10


def test_unknown_reporter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported report_to"):
        minimal_config(report_to=["not-a-reporter"])
