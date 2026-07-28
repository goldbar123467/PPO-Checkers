"""Pure acceptance accounting for the measured practice preflight."""

from __future__ import annotations

import math

import pytest

from checkers.practice_preflight import loss_equivalence

LOSS_VALUE_COUNT = 2


def _history(policy: float, value: float) -> dict[int, dict[str, float]]:
    return {
        1: {
            "train/policy_loss": policy,
            "train/value_loss": value,
        }
    }


def test_loss_equivalence_uses_float64_bits_and_reports_numeric_delta() -> None:
    first = _history(0.0, 1.0)
    same = _history(0.0, 1.0)
    different = _history(-0.0, math.nextafter(1.0, 2.0))

    assert loss_equivalence(first, same) == (LOSS_VALUE_COUNT, 0, 0.0)
    compared, mismatches, max_delta = loss_equivalence(first, different)

    assert compared == LOSS_VALUE_COUNT
    assert mismatches == LOSS_VALUE_COUNT
    assert max_delta == math.nextafter(1.0, 2.0) - 1.0


def test_loss_equivalence_rejects_different_update_coverage() -> None:
    with pytest.raises(ValueError, match="update indices"):
        loss_equivalence(_history(0.0, 1.0), {})
