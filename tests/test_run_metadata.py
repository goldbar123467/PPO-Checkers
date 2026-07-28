from __future__ import annotations

from ml_lab.run_metadata import sanitize_metadata


def test_wandb_style_values_are_redacted() -> None:
    assert sanitize_metadata("wandb_examplefakevalue") == "[REDACTED]"
