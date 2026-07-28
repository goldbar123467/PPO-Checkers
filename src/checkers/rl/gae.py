"""Perspective-aware generalized advantage estimation for step-wise checkers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GAEOutput:
    """Advantages and their internally consistent value targets."""

    advantages: torch.Tensor
    returns: torch.Tensor


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return checked


def _validate_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _validate_inputs(  # noqa: PLR0912, PLR0913, PLR0917
    rewards: object,
    values: object,
    dones: object,
    sigmas: object,
    bootstrap_value: object,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    checked_rewards = _validate_tensor(rewards, "rewards")
    checked_values = _validate_tensor(values, "values")
    checked_dones = _validate_tensor(dones, "dones")
    checked_sigmas = _validate_tensor(sigmas, "sigmas")
    checked_bootstrap = _validate_tensor(bootstrap_value, "bootstrap_value")
    if not checked_rewards.is_floating_point():
        raise TypeError("rewards must be floating")
    if not checked_values.is_floating_point():
        raise TypeError("values must be floating")
    if checked_dones.dtype is not torch.bool:
        raise TypeError("dones must have dtype bool")
    if checked_sigmas.dtype is torch.bool or checked_sigmas.is_complex():
        raise TypeError("sigmas must be real numeric values")
    if not checked_bootstrap.is_floating_point():
        raise TypeError("bootstrap_value must be floating")
    if checked_rewards.ndim < 1 or checked_rewards.shape[0] < 1:
        raise ValueError("rewards must contain at least one transition")
    if not (
        checked_values.shape == checked_dones.shape == checked_sigmas.shape == checked_rewards.shape
    ):
        raise ValueError("rewards, values, dones, and sigmas must have the same shape")
    if checked_bootstrap.shape != checked_values.shape[1:]:
        raise ValueError("bootstrap_value shape must match the non-time dimensions")
    if not (
        checked_rewards.dtype == checked_values.dtype == checked_bootstrap.dtype
        and checked_rewards.device
        == checked_values.device
        == checked_dones.device
        == checked_sigmas.device
        == checked_bootstrap.device
    ):
        raise ValueError("all GAE tensors must share floating dtype and device")
    if not bool(torch.isfinite(checked_rewards).all().item()):
        raise ValueError("rewards must be finite")
    if not bool(torch.isfinite(checked_values).all().item()):
        raise ValueError("values must be finite")
    if not bool(torch.isfinite(checked_bootstrap).all().item()):
        raise ValueError("bootstrap_value must be finite")
    if not bool(((checked_sigmas == 1) | (checked_sigmas == -1)).all().item()):
        raise ValueError("sigmas must contain only -1 or +1")
    return (
        checked_rewards,
        checked_values,
        checked_dones,
        checked_sigmas,
        checked_bootstrap,
        _unit_interval(gamma, "gamma"),
        _unit_interval(gae_lambda, "gae_lambda"),
    )


def compute_two_player_gae(  # noqa: PLR0913
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    sigmas: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> GAEOutput:
    """Compute signed GAE over one or more time-aligned environment lanes.

    Time is the first dimension. Interior next-state values come from the following time row; the
    final non-terminal row uses ``bootstrap_value`` and has no recursively propagated advantage
    beyond the rollout boundary. Both bootstrap terms are multiplied by the stored perspective
    sign, while terminal rows mask both terms.

    Args:
        rewards: Floating transition rewards shaped ``(time, ...)``.
        values: Floating actor-relative values for each stored state, same shape as ``rewards``.
        dones: Boolean true terminal indicators, same shape as ``rewards``.
        sigmas: Perspective signs containing only ``-1`` and ``+1``, same shape as ``rewards``.
        bootstrap_value: Value of the state after the final transition, shaped like one time row.
        gamma: Discount factor in ``[0, 1]``.
        gae_lambda: GAE trace factor in ``[0, 1]``.

    Returns:
        Advantages and ``advantages + values`` targets with the same shape/dtype/device as values.

    Raises:
        TypeError: If tensors or scalar parameters have invalid types or dtypes.
        ValueError: If shapes, devices, values, signs, or parameter ranges are invalid.
    """

    (
        checked_rewards,
        checked_values,
        checked_dones,
        checked_sigmas,
        checked_bootstrap,
        checked_gamma,
        checked_lambda,
    ) = _validate_inputs(
        rewards,
        values,
        dones,
        sigmas,
        bootstrap_value,
        gamma,
        gae_lambda,
    )
    advantages = torch.empty_like(checked_values)
    next_advantage = torch.zeros_like(checked_bootstrap)
    final_index = checked_rewards.shape[0] - 1
    for index in range(final_index, -1, -1):
        next_value = checked_bootstrap if index == final_index else checked_values[index + 1]
        nonterminal = (~checked_dones[index]).to(dtype=checked_values.dtype)
        sigma = checked_sigmas[index].to(dtype=checked_values.dtype)
        delta = (
            checked_rewards[index]
            + checked_gamma * nonterminal * sigma * next_value
            - checked_values[index]
        )
        next_advantage = (
            delta + checked_gamma * checked_lambda * nonterminal * sigma * next_advantage
        )
        advantages[index] = next_advantage
    return GAEOutput(advantages=advantages, returns=advantages + checked_values)
