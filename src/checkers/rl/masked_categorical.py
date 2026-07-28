"""Dtype-aware categorical distribution with legality in the gradient path."""

from __future__ import annotations

from typing import cast

import torch
from torch.distributions import Categorical


class MaskedCategorical:
    """Categorical distribution whose illegal logits have exactly zero probability."""

    def __init__(self, *, logits: torch.Tensor, legal_mask: torch.Tensor) -> None:
        """Construct one validated masked distribution.

        Args:
            logits: Floating unnormalized log probabilities with actions on the last axis.
            legal_mask: Boolean tensor of the same shape; true entries are legal actions.

        Raises:
            TypeError: If either input is not a tensor or has the wrong dtype.
            ValueError: If shapes/devices disagree, the action axis is empty, a logit is non-finite,
                or any batch row has no legal action.
        """

        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a Tensor")
        if not isinstance(legal_mask, torch.Tensor):
            raise TypeError("legal_mask must be a Tensor")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating")
        if legal_mask.dtype is not torch.bool:
            raise TypeError("legal_mask must have dtype bool")
        if logits.shape != legal_mask.shape:
            raise ValueError("logits and legal_mask must have the same shape")
        if logits.device != legal_mask.device:
            raise ValueError("logits and legal_mask must be on the same device")
        if logits.ndim < 1 or logits.shape[-1] < 1:
            raise ValueError("action dimension must contain at least one action")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("logits must be finite")
        if not bool(legal_mask.any(dim=-1).all().item()):
            raise ValueError("every distribution row must contain at least one legal action")

        self._legal_mask = legal_mask
        minimum = torch.finfo(logits.dtype).min
        self._distribution = Categorical(logits=logits.masked_fill(~legal_mask, minimum))

    @property
    def legal_mask(self) -> torch.Tensor:
        """Return the boolean legality tensor used to construct the distribution."""

        return self._legal_mask

    @property
    def logits(self) -> torch.Tensor:
        """Return normalized masked logits from the underlying categorical distribution."""

        return self._distribution.logits

    @property
    def probs(self) -> torch.Tensor:
        """Return probabilities with exact zeros at illegal entries."""

        return self._distribution.probs

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Sample legal action indices.

        Args:
            sample_shape: Optional leading sample dimensions.

        Returns:
            Integer action indices with the requested leading dimensions and batch shape.
        """

        sampled = self._distribution.sample(torch.Size(sample_shape))  # type: ignore[no-untyped-call]
        return cast(torch.Tensor, sampled)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return masked log probabilities for action indices.

        Args:
            value: Integer action indices compatible with the batch shape.

        Returns:
            Log probability for each supplied action.
        """

        log_probability = self._distribution.log_prob(value)  # type: ignore[no-untyped-call]
        return cast(torch.Tensor, log_probability)

    def entropy(self) -> torch.Tensor:
        """Return categorical entropy over legal actions only."""

        entropy = self._distribution.entropy()  # type: ignore[no-untyped-call]
        return cast(torch.Tensor, entropy)
