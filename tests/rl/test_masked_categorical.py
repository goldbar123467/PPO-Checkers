"""Numerical and contract tests for the sole legal-action distribution."""

from __future__ import annotations

import pytest
import torch

from checkers.rl.masked_categorical import MaskedCategorical


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sample_log_prob_and_entropy_are_finite_and_legal(dtype: torch.dtype) -> None:
    logits = torch.tensor(
        [[-2.0, 0.0, 2.0, 4.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=dtype,
    )
    legal_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
    )
    distribution = MaskedCategorical(logits=logits, legal_mask=legal_mask)

    torch.manual_seed(20260728)
    samples = distribution.sample((2048,))
    selected_masks = legal_mask.expand(2048, -1, -1).gather(
        dim=-1,
        index=samples.unsqueeze(-1),
    )

    assert selected_masks.all()
    assert torch.isfinite(distribution.log_prob(samples)).all()
    assert torch.isfinite(distribution.entropy()).all()
    assert distribution.probs.dtype == dtype
    assert distribution.legal_mask is legal_mask
    assert torch.isfinite(distribution.logits).all()
    assert torch.equal(distribution.probs[~legal_mask], torch.zeros(4, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_legal_action_is_deterministic_with_zero_entropy(dtype: torch.dtype) -> None:
    logits = torch.tensor([[100.0, -100.0, 0.0]], dtype=dtype)
    legal_mask = torch.tensor([[False, True, False]])
    distribution = MaskedCategorical(logits=logits, legal_mask=legal_mask)

    assert distribution.sample().item() == 1
    assert distribution.log_prob(torch.tensor([1])).item() == pytest.approx(0.0, abs=0.0)
    assert distribution.entropy().item() == pytest.approx(0.0, abs=0.0)


def test_illegal_logits_receive_exactly_zero_gradient() -> None:
    logits = torch.tensor(
        [[0.5, -0.25, 1.5, 3.0], [-1.0, 2.0, 0.0, -2.0]],
        requires_grad=True,
    )
    legal_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
    )
    distribution = MaskedCategorical(logits=logits, legal_mask=legal_mask)
    actions = torch.tensor([2, 1])

    loss = -(distribution.log_prob(actions) + 0.1 * distribution.entropy()).mean()
    loss.backward()  # type: ignore[no-untyped-call]

    assert logits.grad is not None
    assert torch.equal(logits.grad[~legal_mask], torch.zeros(4))
    assert torch.count_nonzero(logits.grad[legal_mask]) == legal_mask.sum()


@pytest.mark.parametrize(
    ("logits", "legal_mask", "error", "message"),
    [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[True, False], [False, False]]),
            ValueError,
            "at least one legal action",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([False, False]),
            ValueError,
            "at least one legal action",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[True, False]]),
            ValueError,
            "same shape",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([1, 0]),
            TypeError,
            "bool",
        ),
        (
            torch.tensor([1, 2]),
            torch.tensor([True, False]),
            TypeError,
            "floating",
        ),
        (
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
            ValueError,
            "action dimension",
        ),
        (
            torch.tensor([0.0, float("nan")]),
            torch.tensor([True, False]),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_distribution_inputs_raise(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        MaskedCategorical(logits=logits, legal_mask=legal_mask)


def test_non_tensor_inputs_raise() -> None:
    with pytest.raises(TypeError, match="logits must be a Tensor"):
        MaskedCategorical(logits=[1.0, 2.0], legal_mask=torch.tensor([True, False]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="legal_mask must be a Tensor"):
        MaskedCategorical(logits=torch.tensor([1.0, 2.0]), legal_mask=[True, False])  # type: ignore[arg-type]


def test_mask_and_logits_must_share_a_device() -> None:
    logits = torch.tensor([1.0, 2.0], device="meta")
    legal_mask = torch.tensor([True, False])
    with pytest.raises(ValueError, match="same device"):
        MaskedCategorical(logits=logits, legal_mask=legal_mask)
