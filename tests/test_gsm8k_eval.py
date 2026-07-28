from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab.gsm8k_eval import EvaluationConfig


def test_evaluation_config_requires_pinned_sources(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        """\
model: Qwen/Qwen3-4B
model_revision: main
benchmark: benchmark.jsonl
benchmark_dataset: openai/gsm8k
benchmark_revision: 740312add88f781978c0658806c59bc2815b9866
track: deterministic
seeds: [42]
wandb_project: test
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinned Qwen3-4B"):
        EvaluationConfig.load(path)
