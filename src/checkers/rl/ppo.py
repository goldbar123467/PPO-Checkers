"""Stored-mask PPO-Clip losses and one-minibatch optimizer updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from checkers.rl.buffer import RolloutBatch
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import CheckersNetwork

LOGIT_RANK = 2


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return checked


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Validated scalar choices for a PPO minibatch loss and optimizer step."""

    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    advantage_eps: float = 1e-8
    target_kl: float = 0.02

    def __post_init__(self) -> None:
        _positive(self.clip_coef, "clip_coef")
        _nonnegative(self.vf_coef, "vf_coef")
        _nonnegative(self.ent_coef, "ent_coef")
        _positive(self.max_grad_norm, "max_grad_norm")
        if not isinstance(self.normalize_advantages, bool):
            raise TypeError("normalize_advantages must be bool")
        _positive(self.advantage_eps, "advantage_eps")
        _positive(self.target_kl, "target_kl")


@dataclass(frozen=True, slots=True)
class PPOMinibatch:
    """Trainable-only PPO tensors derived from one consumed rollout."""

    obs: torch.Tensor
    legal_mask: torch.Tensor
    actions: torch.Tensor
    old_logprob: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    source_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class PPOLosses:
    """Differentiable PPO objective components and diagnostic vectors."""

    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    total_loss: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: torch.Tensor
    ratios: torch.Tensor
    normalized_advantages: torch.Tensor
    kl_early_stop: bool


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    """Detached scalar evidence from one optimizer step."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clipfrac: float
    grad_norm: float
    kl_early_stop: bool
    transitions: int


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _validate_loss_inputs(  # noqa: PLR0912, PLR0913, PLR0917
    logits: object,
    legal_mask: object,
    actions: object,
    old_logprob: object,
    values: object,
    advantages: object,
    returns: object,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    checked_logits = _tensor(logits, "logits")
    checked_mask = _tensor(legal_mask, "legal_mask")
    checked_actions = _tensor(actions, "actions")
    checked_old_logprob = _tensor(old_logprob, "old_logprob")
    checked_values = _tensor(values, "values")
    checked_advantages = _tensor(advantages, "advantages")
    checked_returns = _tensor(returns, "returns")
    if checked_logits.ndim != LOGIT_RANK or checked_logits.shape[1] < 1:
        raise ValueError("logits must have shape (batch, actions) with a non-empty action axis")
    batch_size = checked_actions.shape[0] if checked_actions.ndim == 1 else -1
    if checked_logits.shape[0] != batch_size:
        raise ValueError("logits batch size must match the action vector")
    if checked_mask.shape != checked_logits.shape:
        raise ValueError("legal_mask shape must equal logits shape")
    vector_shape = (batch_size,)
    vectors = (
        (checked_actions, "actions"),
        (checked_old_logprob, "old_logprob"),
        (checked_values, "values"),
        (checked_advantages, "advantages"),
        (checked_returns, "returns"),
    )
    for tensor, name in vectors:
        if tensor.shape != vector_shape:
            raise ValueError(f"{name} shape must be {vector_shape}")
    if not checked_logits.is_floating_point():
        raise TypeError("logits must be floating")
    if checked_mask.dtype is not torch.bool:
        raise TypeError("legal_mask must have dtype bool")
    if checked_actions.dtype is not torch.int64:
        raise TypeError("actions must have dtype int64")
    floating_vectors = (
        (checked_old_logprob, "old_logprob"),
        (checked_values, "values"),
        (checked_advantages, "advantages"),
        (checked_returns, "returns"),
    )
    for tensor, name in floating_vectors:
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating")
    if not (
        checked_logits.dtype
        == checked_old_logprob.dtype
        == checked_values.dtype
        == checked_advantages.dtype
        == checked_returns.dtype
        and checked_logits.device
        == checked_mask.device
        == checked_actions.device
        == checked_old_logprob.device
        == checked_values.device
        == checked_advantages.device
        == checked_returns.device
    ):
        raise ValueError("all PPO tensors must share floating dtype and device")
    finite_tensors = (
        (checked_logits, "logits"),
        (checked_old_logprob, "old_logprob"),
        (checked_values, "values"),
        (checked_advantages, "advantages"),
        (checked_returns, "returns"),
    )
    for tensor, name in finite_tensors:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must be finite")
    in_range = (checked_actions >= 0) & (checked_actions < checked_logits.shape[1])
    if not bool(in_range.all().item()):
        raise ValueError("actions must be legal action indices")
    selected_legal = checked_mask.gather(1, checked_actions.unsqueeze(1))
    if not bool(selected_legal.all().item()):
        raise ValueError("actions must be legal under the stored legal_mask")
    return (
        checked_logits,
        checked_mask,
        checked_actions,
        checked_old_logprob,
        checked_values,
        checked_advantages,
        checked_returns,
    )


def build_ppo_minibatch(rollout: RolloutBatch) -> PPOMinibatch:
    """Build the default trainable-only PPO view from a finalized rollout.

    Args:
        rollout: Full chronology with advantages already computed before filtering.

    Returns:
        Aligned policy/value tensors and their original flattened source indices.

    Raises:
        TypeError: If ``rollout`` is not a ``RolloutBatch``.
        ValueError: If there are no trainable transitions.
    """

    if not isinstance(rollout, RolloutBatch):
        raise TypeError("rollout must be a RolloutBatch")
    policy = rollout.policy_view()
    value = rollout.value_view()
    return PPOMinibatch(
        obs=policy.obs,
        legal_mask=policy.legal_mask,
        actions=policy.action,
        old_logprob=policy.behaviour_logprob,
        old_values=value.values,
        advantages=policy.advantages,
        returns=value.returns,
        source_indices=policy.source_indices,
    )


def compute_ppo_loss(  # noqa: PLR0913
    *,
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    actions: torch.Tensor,
    old_logprob: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> PPOLosses:
    """Compute the clipped policy, plain-MSE value, entropy, and diagnostic terms.

    Args:
        logits: Current unmasked action logits.
        legal_mask: Stored boolean legal masks from collection time.
        actions: Stored trainable-policy actions.
        old_logprob: Stored behavior-policy log probabilities.
        values: Current actor-relative value predictions.
        advantages: Full-chronology signed GAE advantages.
        returns: Internally consistent ``advantages + old_values`` targets.
        config: Validated PPO scalar configuration.

    Returns:
        Differentiable losses, likelihood ratios, normalized advantages, and k3 diagnostics.

    Raises:
        TypeError: If config/tensors/dtypes are invalid.
        ValueError: If shapes, devices, finite values, actions, or masks disagree.
    """

    if not isinstance(config, PPOConfig):
        raise TypeError("config must be a PPOConfig")
    (
        checked_logits,
        checked_mask,
        checked_actions,
        checked_old_logprob,
        checked_values,
        checked_advantages,
        checked_returns,
    ) = _validate_loss_inputs(
        logits,
        legal_mask,
        actions,
        old_logprob,
        values,
        advantages,
        returns,
    )
    normalized_advantages = checked_advantages
    if config.normalize_advantages:
        normalized_advantages = (checked_advantages - checked_advantages.mean()) / (
            checked_advantages.std(unbiased=False) + config.advantage_eps
        )
    distribution = MaskedCategorical(logits=checked_logits, legal_mask=checked_mask)
    new_logprob = distribution.log_prob(checked_actions)
    log_ratio = new_logprob - checked_old_logprob
    ratios = torch.exp(log_ratio)
    unclipped = ratios * normalized_advantages
    clipped = (
        torch.clamp(
            ratios,
            1.0 - config.clip_coef,
            1.0 + config.clip_coef,
        )
        * normalized_advantages
    )
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(checked_values, checked_returns)
    entropy = distribution.entropy().mean()
    total_loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    approx_kl = ((ratios - 1.0) - log_ratio).mean()
    clipfrac = ((ratios - 1.0).abs() > config.clip_coef).to(checked_logits.dtype).mean()
    return PPOLosses(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        total_loss=total_loss,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        ratios=ratios,
        normalized_advantages=normalized_advantages,
        kl_early_stop=float(approx_kl.detach()) > config.target_kl,
    )


def ppo_minibatch_update(
    *,
    network: CheckersNetwork,
    optimizer: torch.optim.Optimizer,
    minibatch: PPOMinibatch,
    config: PPOConfig,
) -> PPOUpdateMetrics:
    """Apply one stored-mask PPO minibatch update with global gradient clipping.

    Args:
        network: Trainable policy/value network.
        optimizer: Optimizer owning the network parameters.
        minibatch: Trainable-only, source-audited PPO tensors.
        config: Validated loss and gradient settings.

    Returns:
        Detached losses, diagnostics, pre-clipping global norm, and transition count.

    Raises:
        TypeError: If a supplied object has the wrong runtime type.
        RuntimeError: If gradients are non-finite.
        ValueError: If minibatch tensor invariants fail.
    """

    if not isinstance(network, CheckersNetwork):
        raise TypeError("network must be a CheckersNetwork")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not isinstance(minibatch, PPOMinibatch):
        raise TypeError("minibatch must be a PPOMinibatch")
    if not isinstance(config, PPOConfig):
        raise TypeError("config must be a PPOConfig")
    optimizer.zero_grad(set_to_none=True)
    output = network(minibatch.obs)
    calculation_dtype = minibatch.old_logprob.dtype
    losses = compute_ppo_loss(
        logits=output.logits.to(dtype=calculation_dtype),
        legal_mask=minibatch.legal_mask,
        actions=minibatch.actions,
        old_logprob=minibatch.old_logprob,
        values=output.value.to(dtype=calculation_dtype),
        advantages=minibatch.advantages,
        returns=minibatch.returns,
        config=config,
    )
    losses.total_loss.backward()  # type: ignore[no-untyped-call]
    norm = torch.nn.utils.clip_grad_norm_(
        network.parameters(),
        max_norm=config.max_grad_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()

    def scalar(tensor: torch.Tensor) -> float:
        return float(tensor.detach())

    return PPOUpdateMetrics(
        policy_loss=scalar(losses.policy_loss),
        value_loss=scalar(losses.value_loss),
        entropy=scalar(losses.entropy),
        total_loss=scalar(losses.total_loss),
        approx_kl=scalar(losses.approx_kl),
        clipfrac=scalar(losses.clipfrac),
        grad_norm=scalar(norm),
        kl_early_stop=losses.kl_early_stop,
        transitions=int(minibatch.actions.shape[0]),
    )
