"""Full-chronology rollout storage for vectorized two-player GAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from checkers.rl.gae import compute_two_player_gae


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """One complete lockstep vector-environment transition row."""

    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: torch.Tensor
    behaviour_logprob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    actor: torch.Tensor
    sigma: torch.Tensor
    trainable: torch.Tensor
    policy_id: tuple[str, ...]
    env_id: torch.Tensor
    move_completed: torch.Tensor


@dataclass(frozen=True, slots=True)
class PolicyView:
    """Only transitions whose actions were sampled by the trainable policy."""

    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: torch.Tensor
    behaviour_logprob: torch.Tensor
    advantages: torch.Tensor
    policy_id: tuple[str, ...]
    source_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class ValueView:
    """Value-regression states and internally consistent GAE targets."""

    obs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    source_indices: torch.Tensor


@dataclass(frozen=True, slots=True, weakref_slot=True)
class RolloutBatch:
    """Flattened time-major rollout with GAE computed before any filtering."""

    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: torch.Tensor
    behaviour_logprob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    actor: torch.Tensor
    sigma: torch.Tensor
    trainable: torch.Tensor
    policy_id: tuple[str, ...]
    env_id: torch.Tensor
    move_completed: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @property
    def transitions(self) -> int:
        """Return the number of flattened transitions."""

        return int(self.action.shape[0])

    def _trainable_indices(self) -> torch.Tensor:
        indices = torch.nonzero(self.trainable, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("rollout contains no trainable transitions")
        return indices

    def policy_view(self) -> PolicyView:
        """Return the policy-loss view with all opponent transitions excluded.

        Returns:
            Stored observations, masks, actions, old log-probabilities, and advantages for only
            transitions sampled by the current trainable policy.

        Raises:
            ValueError: If the rollout contains no trainable transition.
        """

        indices = self._trainable_indices()
        integer_indices = cast(list[int], indices.tolist())
        return PolicyView(
            obs=self.obs[indices],
            legal_mask=self.legal_mask[indices],
            action=self.action[indices],
            behaviour_logprob=self.behaviour_logprob[indices],
            advantages=self.advantages[indices],
            policy_id=tuple(self.policy_id[index] for index in integer_indices),
            source_indices=indices,
        )

    def value_view(self, *, include_nontrainable: bool = False) -> ValueView:
        """Return default trainable-only or explicit all-state value targets.

        Args:
            include_nontrainable: Include frozen-opponent states for the declared ablation.

        Returns:
            Observations, old values, targets, and original flattened indices.

        Raises:
            TypeError: If ``include_nontrainable`` is not boolean.
            ValueError: If the default view has no trainable transition.
        """

        if not isinstance(include_nontrainable, bool):
            raise TypeError("include_nontrainable must be bool")
        indices = (
            torch.arange(self.transitions, device=self.action.device)
            if include_nontrainable
            else self._trainable_indices()
        )
        return ValueView(
            obs=self.obs[indices],
            values=self.value[indices],
            returns=self.returns[indices],
            source_indices=indices,
        )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _observation_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("observation_shape must be a tuple of integers")
    if not value:
        raise ValueError("observation_shape must not be empty")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in value):
        raise TypeError("observation_shape must contain only integers")
    checked = cast(tuple[int, ...], value)
    if any(size < 1 for size in checked):
        raise ValueError("observation_shape dimensions must be positive")
    return checked


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    return value


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


class RolloutBuffer:
    """Fixed-size vector rollout that can be finalized exactly once."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_steps: int,
        observation_shape: tuple[int, ...],
        action_count: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Create an empty lockstep rollout.

        Args:
            num_envs: Number of stable vector lanes in every appended row.
            num_steps: Number of environment-step rows in the rollout.
            observation_shape: Shape after the environment-lane dimension.
            action_count: Fixed legal-mask width.
            device: Required device for every stored tensor.

        Raises:
            TypeError: If dimensions have invalid runtime types.
            ValueError: If a dimension is empty or non-positive.
        """

        self.num_envs = _positive_integer(num_envs, "num_envs")
        self.num_steps = _positive_integer(num_steps, "num_steps")
        self.observation_shape = _observation_shape(observation_shape)
        self.action_count = _positive_integer(action_count, "action_count")
        self.device = torch.device(device)
        self._steps: list[RolloutStep] = []
        self._finalized = False

    @property
    def size(self) -> int:
        """Return the number of complete vector rows currently stored."""

        return len(self._steps)

    @property
    def full(self) -> bool:
        """Return whether the configured rollout capacity has been reached."""

        return self.size == self.num_steps

    def _validate_shapes(self, step: RolloutStep) -> None:
        expected_obs = (self.num_envs, *self.observation_shape)
        if step.obs.shape != expected_obs:
            raise ValueError(f"obs shape must be {expected_obs}")
        expected_mask = (self.num_envs, self.action_count)
        if step.legal_mask.shape != expected_mask:
            raise ValueError(f"legal_mask shape must be {expected_mask}")
        expected_vector = (self.num_envs,)
        vector_fields = (
            (step.action, "action"),
            (step.behaviour_logprob, "behaviour_logprob"),
            (step.value, "value"),
            (step.reward, "reward"),
            (step.done, "done"),
            (step.actor, "actor"),
            (step.sigma, "sigma"),
            (step.trainable, "trainable"),
            (step.env_id, "env_id"),
            (step.move_completed, "move_completed"),
        )
        for tensor, name in vector_fields:
            if tensor.shape != expected_vector:
                raise ValueError(f"{name} shape must be {expected_vector}")

    def _validate_step(self, step: RolloutStep) -> None:  # noqa: PLR0912, PLR0915
        fields = (
            (step.obs, "obs"),
            (step.legal_mask, "legal_mask"),
            (step.action, "action"),
            (step.behaviour_logprob, "behaviour_logprob"),
            (step.value, "value"),
            (step.reward, "reward"),
            (step.done, "done"),
            (step.actor, "actor"),
            (step.sigma, "sigma"),
            (step.trainable, "trainable"),
            (step.env_id, "env_id"),
            (step.move_completed, "move_completed"),
        )
        for value, name in fields:
            _tensor(value, name)
        self._validate_shapes(step)
        if any(tensor.device != self.device for tensor, _ in fields):
            raise ValueError("all step tensors must be on the buffer device")
        if not step.obs.is_floating_point():
            raise TypeError("obs must be floating")
        if step.legal_mask.dtype is not torch.bool:
            raise TypeError("legal_mask must have dtype bool")
        if step.action.dtype is not torch.int64:
            raise TypeError("action must have dtype int64")
        scalar_fields = (
            (step.behaviour_logprob, "behaviour_logprob"),
            (step.value, "value"),
            (step.reward, "reward"),
        )
        for tensor, name in scalar_fields:
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating")
        if not (step.behaviour_logprob.dtype == step.value.dtype == step.reward.dtype):
            raise ValueError("behaviour_logprob, value, and reward must share a dtype")
        if step.done.dtype is not torch.bool:
            raise TypeError("done must have dtype bool")
        if step.actor.dtype is not torch.int64:
            raise TypeError("actor must have dtype int64")
        if step.sigma.dtype is not torch.int8:
            raise TypeError("sigma must have dtype int8")
        if step.trainable.dtype is not torch.bool:
            raise TypeError("trainable must have dtype bool")
        if step.env_id.dtype is not torch.int64:
            raise TypeError("env_id must have dtype int64")
        if step.move_completed.dtype is not torch.bool:
            raise TypeError("move_completed must have dtype bool")
        if not bool(torch.isfinite(step.obs).all().item()):
            raise ValueError("obs must be finite")
        for tensor, name in scalar_fields:
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} must be finite")
        if not bool(((step.action >= 0) & (step.action < self.action_count)).all().item()):
            raise ValueError(f"action must be in [0, {self.action_count})")
        selected_legal = step.legal_mask.gather(dim=1, index=step.action.unsqueeze(1))
        if not bool(selected_legal.all().item()):
            raise ValueError("every stored action must be legal under its stored legal_mask")
        if not bool(((step.reward == -1) | (step.reward == 0) | (step.reward == 1)).all().item()):
            raise ValueError("reward must contain only -1, 0, or +1")
        if not bool(((step.actor == 0) | (step.actor == 1)).all().item()):
            raise ValueError("actor must contain only player IDs 0 or 1")
        if not bool(((step.sigma == -1) | (step.sigma == 1)).all().item()):
            raise ValueError("sigma must contain only -1 or +1")
        if not isinstance(step.policy_id, tuple) or len(step.policy_id) != self.num_envs:
            raise ValueError("policy_id length must equal num_envs")
        if not all(isinstance(policy_id, str) for policy_id in step.policy_id):
            raise TypeError("policy_id entries must be strings")
        if not all(step.policy_id):
            raise ValueError("policy_id entries must not be empty")
        expected_env_ids = torch.arange(self.num_envs, device=self.device)
        if not torch.equal(step.env_id, expected_env_ids):
            raise ValueError("env_id must equal the stable lane order [0, num_envs)")

    @staticmethod
    def _clone_step(step: RolloutStep) -> RolloutStep:
        return RolloutStep(
            obs=_clone(step.obs),
            legal_mask=_clone(step.legal_mask),
            action=_clone(step.action),
            behaviour_logprob=_clone(step.behaviour_logprob),
            value=_clone(step.value),
            reward=_clone(step.reward),
            done=_clone(step.done),
            actor=_clone(step.actor),
            sigma=_clone(step.sigma),
            trainable=_clone(step.trainable),
            policy_id=tuple(step.policy_id),
            env_id=_clone(step.env_id),
            move_completed=_clone(step.move_completed),
        )

    def append(self, step: RolloutStep) -> None:
        """Append one validated, cloned lockstep vector row.

        Args:
            step: Every required field for exactly one transition in every stable environment lane.

        Raises:
            TypeError: If the row or one field has an invalid type/dtype.
            ValueError: If shapes, devices, finite values, IDs, signs, or legality disagree.
            OverflowError: If the configured rollout is already full.
        """

        if not isinstance(step, RolloutStep):
            raise TypeError("step must be a RolloutStep")
        if self.full:
            raise OverflowError("rollout buffer is already full")
        self._validate_step(step)
        self._steps.append(self._clone_step(step))

    def _stack(self, name: str) -> torch.Tensor:
        tensors = [_tensor(getattr(step, name), name) for step in self._steps]
        return torch.stack(tensors, dim=0)

    def finalize(
        self,
        *,
        bootstrap_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        """Compute full-chronology GAE and consume this rollout exactly once.

        Args:
            bootstrap_value: Actor-relative value after the last row in each environment lane.
            gamma: Discount factor passed to two-player GAE.
            gae_lambda: Trace factor passed to two-player GAE.

        Returns:
            A flattened time-major rollout with unfiltered advantages and returns.

        Raises:
            TypeError: If the bootstrap has an invalid type or dtype.
            ValueError: If the buffer is incomplete or bootstrap shape/device/content disagrees.
            RuntimeError: If this rollout has already been finalized.
        """

        if self._finalized:
            raise RuntimeError("rollout buffer was already finalized")
        if not self.full:
            raise ValueError("rollout buffer is not full")
        checked_bootstrap = _tensor(bootstrap_value, "bootstrap_value")
        if checked_bootstrap.shape != (self.num_envs,):
            raise ValueError(f"bootstrap_value shape must be {(self.num_envs,)}")
        if not checked_bootstrap.is_floating_point():
            raise TypeError("bootstrap_value must be floating")
        if checked_bootstrap.device != self.device:
            raise ValueError("bootstrap_value must be on the buffer device")
        values = self._stack("value")
        rewards = self._stack("reward")
        if checked_bootstrap.dtype != values.dtype:
            raise ValueError("bootstrap_value dtype must match stored values")
        if not bool(torch.isfinite(checked_bootstrap).all().item()):
            raise ValueError("bootstrap_value must be finite")
        gae = compute_two_player_gae(
            rewards=rewards,
            values=values,
            dones=self._stack("done"),
            sigmas=self._stack("sigma"),
            bootstrap_value=checked_bootstrap,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self._finalized = True

        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(self.num_steps * self.num_envs, *tensor.shape[2:])

        return RolloutBatch(
            obs=flatten(self._stack("obs")),
            legal_mask=flatten(self._stack("legal_mask")),
            action=flatten(self._stack("action")),
            behaviour_logprob=flatten(self._stack("behaviour_logprob")),
            value=flatten(values),
            reward=flatten(rewards),
            done=flatten(self._stack("done")),
            actor=flatten(self._stack("actor")),
            sigma=flatten(self._stack("sigma")),
            trainable=flatten(self._stack("trainable")),
            policy_id=tuple(policy_id for step in self._steps for policy_id in step.policy_id),
            env_id=flatten(self._stack("env_id")),
            move_completed=flatten(self._stack("move_completed")),
            advantages=flatten(gae.advantages),
            returns=flatten(gae.returns),
        )
