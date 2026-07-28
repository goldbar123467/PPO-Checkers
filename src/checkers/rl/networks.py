"""GroupNorm residual policy/value network for canonical checkers observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

OBSERVATION_SHAPE = (8, 8, 8)
TRUNK_CHANNELS = 64
TRUNK_GROUPS = 8
RESIDUAL_BLOCK_COUNT = 6
ACTION_COUNT = 128
OBSERVATION_RANK = len(OBSERVATION_SHAPE) + 1


def _initialize_affine(module: nn.Conv2d | nn.Linear, *, gain: float) -> None:
    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.zeros_(cast(torch.Tensor, module.bias))


def _initialize_group_norm(module: nn.GroupNorm) -> None:
    nn.init.ones_(cast(torch.Tensor, module.weight))
    nn.init.zeros_(cast(torch.Tensor, module.bias))


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    """Unmasked policy logits and actor-relative bounded values."""

    logits: torch.Tensor
    value: torch.Tensor


class ResidualBlock(nn.Module):
    """Two-convolution GroupNorm residual block with a post-addition ReLU."""

    def __init__(self) -> None:
        """Initialize one 64-channel residual block with orthogonal weights."""

        super().__init__()
        self.conv1 = nn.Conv2d(TRUNK_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(TRUNK_GROUPS, TRUNK_CHANNELS)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(TRUNK_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(TRUNK_GROUPS, TRUNK_CHANNELS)
        self.relu2 = nn.ReLU()
        _initialize_affine(self.conv1, gain=math.sqrt(2.0))
        _initialize_affine(self.conv2, gain=math.sqrt(2.0))
        _initialize_group_norm(self.norm1)
        _initialize_group_norm(self.norm2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the residual transform.

        Args:
            features: Tensor shaped ``(batch, 64, 8, 8)``.

        Returns:
            Tensor with the same shape.
        """

        transformed = self.relu1(self.norm1(self.conv1(features)))
        transformed = self.norm2(self.conv2(transformed))
        return cast(torch.Tensor, self.relu2(transformed + features))


class PolicyHead(nn.Module):
    """One-by-one convolutional head producing 128 unmasked action logits."""

    def __init__(self) -> None:
        """Initialize the policy head with the declared small output gain."""

        super().__init__()
        self.conv = nn.Conv2d(TRUNK_CHANNELS, 2, kernel_size=1)
        self.norm = nn.GroupNorm(1, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Linear(ACTION_COUNT, ACTION_COUNT)
        _initialize_affine(self.conv, gain=math.sqrt(2.0))
        _initialize_group_norm(self.norm)
        _initialize_affine(self.output, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map trunk features to unmasked action logits.

        Args:
            features: Tensor shaped ``(batch, 64, 8, 8)``.

        Returns:
            Tensor shaped ``(batch, 128)``.
        """

        logits = self.output(self.flatten(self.relu(self.norm(self.conv(features)))))
        return cast(torch.Tensor, logits)


class ValueHead(nn.Module):
    """One-by-one convolutional head producing actor-relative values in [-1, 1]."""

    def __init__(self) -> None:
        """Initialize the two-layer value head with declared orthogonal gains."""

        super().__init__()
        self.conv = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=1)
        self.norm = nn.GroupNorm(1, 1)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        _initialize_affine(self.conv, gain=math.sqrt(2.0))
        _initialize_group_norm(self.norm)
        _initialize_affine(self.hidden, gain=math.sqrt(2.0))
        _initialize_affine(self.output, gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map trunk features to bounded scalar values.

        Args:
            features: Tensor shaped ``(batch, 64, 8, 8)``.

        Returns:
            Tensor shaped ``(batch,)`` in ``[-1, 1]``.
        """

        hidden = self.flatten(self.relu1(self.norm(self.conv(features))))
        hidden = self.relu2(self.hidden(hidden))
        return torch.tanh(self.output(hidden)).squeeze(-1)


class CheckersNetwork(nn.Module):
    """Shared six-block GroupNorm trunk with separate policy and value heads."""

    def __init__(self) -> None:
        """Initialize the exact N1–N6 baseline architecture."""

        super().__init__()
        stem_conv = nn.Conv2d(8, TRUNK_CHANNELS, kernel_size=3, padding=1)
        stem_norm = nn.GroupNorm(TRUNK_GROUPS, TRUNK_CHANNELS)
        _initialize_affine(stem_conv, gain=math.sqrt(2.0))
        _initialize_group_norm(stem_norm)
        self.stem = nn.Sequential(stem_conv, stem_norm, nn.ReLU())
        self.residual_blocks = nn.ModuleList(ResidualBlock() for _ in range(RESIDUAL_BLOCK_COUNT))
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    @staticmethod
    def _validate_observation(observation: object) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a Tensor")
        if (
            observation.ndim != OBSERVATION_RANK
            or tuple(observation.shape[1:]) != OBSERVATION_SHAPE
        ):
            raise ValueError("observation shape must be (batch, 8, 8, 8)")
        if observation.shape[0] < 1:
            raise ValueError("observation must contain a non-empty batch")
        if not observation.is_floating_point():
            raise TypeError("observation must be floating")
        if not bool(torch.isfinite(observation).all().item()):
            raise ValueError("observation must be finite")
        return observation

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode canonical observations with the shared GroupNorm trunk.

        Args:
            observation: Floating tensor shaped ``(batch, 8, 8, 8)``.

        Returns:
            Trunk features shaped ``(batch, 64, 8, 8)``.

        Raises:
            TypeError: If the input is not a floating tensor.
            ValueError: If its shape/content is invalid or its batch is empty.
        """

        features = cast(torch.Tensor, self.stem(self._validate_observation(observation)))
        for block in self.residual_blocks:
            features = cast(torch.Tensor, block(features))
        return features

    def forward(self, observation: torch.Tensor) -> NetworkOutput:
        """Return unmasked policy logits and actor-relative values.

        Args:
            observation: Floating tensor shaped ``(batch, 8, 8, 8)``.

        Returns:
            Policy logits shaped ``(batch, 128)`` and values shaped ``(batch,)``.

        Raises:
            TypeError: If the input is not a floating tensor.
            ValueError: If its shape/content is invalid or its batch is empty.
        """

        features = self.encode(observation)
        return NetworkOutput(
            logits=self.policy_head(features),
            value=self.value_head(features),
        )
