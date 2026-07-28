"""Hand oracles and directional tests for PPO-Clip."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from checkers.rl.buffer import RolloutBuffer, RolloutStep
from checkers.rl.gae import compute_two_player_gae
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import CheckersNetwork
from checkers.rl.ppo import (
    PPOConfig,
    PPOMinibatch,
    build_ppo_minibatch,
    compute_ppo_loss,
    ppo_minibatch_update,
)

CLIPPED_POSITIVE_PROBABILITY = 0.6
CLIPPED_NEGATIVE_PROBABILITY = 0.4
UPDATE_TRANSITIONS = 4


def test_t3_four_transition_ppo_loss_matches_literal_hand_oracle() -> None:
    gae = compute_two_player_gae(
        rewards=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
        values=torch.tensor([0.2, -0.1, 0.3, 0.4], dtype=torch.float64),
        dones=torch.tensor([False, False, False, True]),
        sigmas=torch.tensor([1, -1, -1, 1], dtype=torch.int8),
        bootstrap_value=torch.tensor(0.7, dtype=torch.float64),
        gamma=0.9,
        gae_lambda=0.8,
    )
    chosen_probabilities = torch.tensor([0.55, 0.45, 0.65, 0.35], dtype=torch.float64)
    logits = torch.log(torch.stack((chosen_probabilities, 1.0 - chosen_probabilities), dim=1))
    losses = compute_ppo_loss(
        logits=logits,
        legal_mask=torch.ones((4, 2), dtype=torch.bool),
        actions=torch.zeros(4, dtype=torch.int64),
        old_logprob=torch.full((4,), -math.log(2.0), dtype=torch.float64),
        values=torch.tensor([0.25, -0.2, 0.0, 0.5], dtype=torch.float64),
        advantages=gae.advantages,
        returns=gae.returns,
        config=PPOConfig(
            clip_coef=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            normalize_advantages=False,
            advantage_eps=1e-8,
            target_kl=0.02,
        ),
    )

    expected = {
        "policy_loss": 0.06898048,
        "value_loss": 0.35025398359296,
        "entropy": 0.6677927263741105,
        "approx_kl": 0.02609025383118571,
        "clipfrac": 0.5,
        "total_loss": 0.23742954453273896,
    }
    torch.testing.assert_close(
        losses.ratios,
        torch.tensor([1.1, 0.9, 1.3, 0.7], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.equal(losses.normalized_advantages, gae.advantages)
    for name, value in expected.items():
        actual = getattr(losses, name)
        assert float(actual) == pytest.approx(value, abs=1e-12)
    assert losses.kl_early_stop


def test_t4_probabilities_move_monotonically_until_both_ratios_are_clipped() -> None:
    logits = nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    optimizer = torch.optim.SGD((logits,), lr=0.2)
    config = PPOConfig(
        clip_coef=0.2,
        vf_coef=0.0,
        ent_coef=0.0,
        max_grad_norm=1.0,
        normalize_advantages=False,
        advantage_eps=1e-8,
        target_kl=1.0,
    )
    positive_probabilities: list[float] = []
    negative_probabilities: list[float] = []
    clipfractions: list[float] = []

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_ppo_loss(
            logits=logits,
            legal_mask=torch.ones((2, 2), dtype=torch.bool),
            actions=torch.zeros(2, dtype=torch.int64),
            old_logprob=torch.full((2,), -math.log(2.0), dtype=torch.float64),
            values=torch.zeros(2, dtype=torch.float64),
            advantages=torch.tensor([1.0, -1.0], dtype=torch.float64),
            returns=torch.zeros(2, dtype=torch.float64),
            config=config,
        )
        probabilities = torch.softmax(logits.detach(), dim=-1)[:, 0]
        positive_probabilities.append(float(probabilities[0]))
        negative_probabilities.append(float(probabilities[1]))
        clipfractions.append(float(losses.clipfrac))
        losses.total_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    assert all(
        after >= before
        for before, after in zip(
            positive_probabilities,
            positive_probabilities[1:],
            strict=False,
        )
    )
    assert all(
        after <= before
        for before, after in zip(
            negative_probabilities,
            negative_probabilities[1:],
            strict=False,
        )
    )
    assert positive_probabilities[-1] >= CLIPPED_POSITIVE_PROBABILITY
    assert negative_probabilities[-1] <= CLIPPED_NEGATIVE_PROBABILITY
    assert 1.0 in clipfractions
    assert positive_probabilities[-1] == positive_probabilities[-2]
    assert negative_probabilities[-1] == negative_probabilities[-2]


def test_default_advantage_normalization_uses_population_variance_and_epsilon() -> None:
    losses = compute_ppo_loss(
        logits=torch.zeros((4, 2)),
        legal_mask=torch.ones((4, 2), dtype=torch.bool),
        actions=torch.zeros(4, dtype=torch.int64),
        old_logprob=torch.full((4,), -math.log(2.0)),
        values=torch.zeros(4),
        advantages=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        returns=torch.zeros(4),
        config=PPOConfig(),
    )
    expected = torch.tensor([-1.3416407, -0.4472136, 0.4472136, 1.3416407])

    torch.testing.assert_close(losses.normalized_advantages, expected)
    assert float(losses.normalized_advantages.mean()) == pytest.approx(0.0, abs=1e-7)


def _single_transition_rollout(*, trainable: bool = True) -> object:
    buffer = RolloutBuffer(
        num_envs=1,
        num_steps=1,
        observation_shape=(8, 8, 8),
        action_count=128,
    )
    legal_mask = torch.zeros((1, 128), dtype=torch.bool)
    legal_mask[0, 17] = True
    buffer.append(
        RolloutStep(
            obs=torch.zeros((1, 8, 8, 8)),
            legal_mask=legal_mask,
            action=torch.tensor([17]),
            behaviour_logprob=torch.tensor([0.0]),
            value=torch.tensor([0.25]),
            reward=torch.tensor([1.0]),
            done=torch.tensor([True]),
            actor=torch.tensor([0]),
            sigma=torch.tensor([1], dtype=torch.int8),
            trainable=torch.tensor([trainable]),
            policy_id=(("current" if trainable else "frozen"),),
            env_id=torch.tensor([0]),
            move_completed=torch.tensor([True]),
        )
    )
    return buffer.finalize(
        bootstrap_value=torch.tensor([0.0]),
        gamma=1.0,
        gae_lambda=0.95,
    )


def test_build_minibatch_uses_stored_mask_and_trainable_indices() -> None:
    rollout = _single_transition_rollout()
    minibatch = build_ppo_minibatch(rollout)  # type: ignore[arg-type]

    assert isinstance(minibatch, PPOMinibatch)
    assert minibatch.source_indices.tolist() == [0]
    assert minibatch.legal_mask[0].nonzero().flatten().tolist() == [17]
    assert minibatch.actions.tolist() == [17]
    assert minibatch.old_logprob.tolist() == [0.0]
    assert minibatch.old_values.tolist() == [0.25]
    assert minibatch.advantages.tolist() == [0.75]
    assert minibatch.returns.tolist() == [1.0]


def test_minibatch_update_changes_parameters_and_clips_global_gradient() -> None:
    torch.manual_seed(9)
    network = CheckersNetwork()
    observations = torch.rand((4, 8, 8, 8))
    masks = torch.zeros((4, 128), dtype=torch.bool)
    masks[:, :4] = True
    actions = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        initial = network(observations)
        old_logprob = MaskedCategorical(logits=initial.logits, legal_mask=masks).log_prob(actions)
    minibatch = PPOMinibatch(
        obs=observations,
        legal_mask=masks,
        actions=actions,
        old_logprob=old_logprob,
        old_values=initial.value.detach(),
        advantages=torch.tensor([1.0, -1.0, 0.5, -0.5]),
        returns=torch.tensor([1.0, -1.0, 0.75, -0.75]),
        source_indices=torch.arange(4),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
    first_parameter = next(network.parameters())
    before = first_parameter.detach().clone()
    config = replace(PPOConfig(), max_grad_norm=1e-4)

    metrics = ppo_minibatch_update(
        network=network,
        optimizer=optimizer,
        minibatch=minibatch,
        config=config,
    )

    assert not torch.equal(first_parameter.detach(), before)
    assert metrics.grad_norm > config.max_grad_norm
    clipped_norm = torch.linalg.vector_norm(
        torch.cat(
            [
                parameter.grad.detach().flatten()
                for parameter in network.parameters()
                if parameter.grad is not None
            ]
        )
    )
    assert float(clipped_norm) <= config.max_grad_norm * 1.001
    assert metrics.transitions == UPDATE_TRANSITIONS
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.total_loss,
            metrics.approx_kl,
            metrics.clipfrac,
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("clip_coef", 0.0, ValueError, "clip_coef must be positive"),
        ("clip_coef", True, TypeError, "clip_coef must be numeric"),
        ("vf_coef", True, TypeError, "vf_coef must be numeric"),
        ("vf_coef", -1.0, ValueError, "vf_coef must be non-negative"),
        ("ent_coef", float("nan"), ValueError, "ent_coef must be non-negative"),
        ("max_grad_norm", 0.0, ValueError, "max_grad_norm must be positive"),
        ("normalize_advantages", 1, TypeError, "normalize_advantages must be bool"),
        ("advantage_eps", 0.0, ValueError, "advantage_eps must be positive"),
        ("target_kl", 0.0, ValueError, "target_kl must be positive"),
    ],
)
def test_invalid_ppo_config_raises(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        replace(PPOConfig(), **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("logits", object(), TypeError, "logits must be a Tensor"),
        ("logits", torch.zeros(2, 2), ValueError, "batch size"),
        ("logits", torch.zeros(3), ValueError, "shape"),
        ("logits", torch.zeros(3, 0), ValueError, "shape"),
        ("logits", torch.zeros(3, 2, dtype=torch.int64), TypeError, "logits must be floating"),
        ("legal_mask", torch.ones(3, 3, dtype=torch.bool), ValueError, "legal_mask shape"),
        ("legal_mask", torch.ones(3, 2), TypeError, "legal_mask must have dtype bool"),
        ("actions", torch.zeros(3), TypeError, "actions must have dtype int64"),
        ("actions", torch.tensor([0, 0, 2]), ValueError, "actions must be legal"),
        ("actions", torch.tensor([0, 0, 0]), ValueError, "stored legal_mask"),
        ("old_logprob", torch.zeros(2), ValueError, "shape"),
        ("values", torch.zeros(3, dtype=torch.int64), TypeError, "values must be floating"),
        ("advantages", torch.tensor([0.0, float("nan"), 0.0]), ValueError, "finite"),
        ("returns", torch.zeros(3, dtype=torch.float64), ValueError, "dtype and device"),
    ],
)
def test_invalid_ppo_loss_tensors_raise(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    arguments = {
        "logits": torch.zeros((3, 2)),
        "legal_mask": torch.tensor([[True, True], [True, False], [False, True]]),
        "actions": torch.tensor([0, 0, 1]),
        "old_logprob": torch.zeros(3),
        "values": torch.zeros(3),
        "advantages": torch.ones(3),
        "returns": torch.zeros(3),
        "config": PPOConfig(),
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        compute_ppo_loss(**arguments)  # type: ignore[arg-type]


def test_build_minibatch_rejects_wrong_type_and_no_trainable_rows() -> None:
    with pytest.raises(TypeError, match="rollout must be a RolloutBatch"):
        build_ppo_minibatch(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no trainable transitions"):
        build_ppo_minibatch(_single_transition_rollout(trainable=False))  # type: ignore[arg-type]


def _valid_loss_arguments() -> dict[str, object]:
    return {
        "logits": torch.zeros((2, 2)),
        "legal_mask": torch.ones((2, 2), dtype=torch.bool),
        "actions": torch.zeros(2, dtype=torch.int64),
        "old_logprob": torch.full((2,), -math.log(2.0)),
        "values": torch.zeros(2),
        "advantages": torch.tensor([1.0, -1.0]),
        "returns": torch.zeros(2),
        "config": PPOConfig(),
    }


def test_loss_rejects_non_config() -> None:
    arguments = _valid_loss_arguments()
    arguments["config"] = object()
    with pytest.raises(TypeError, match="config must be a PPOConfig"):
        compute_ppo_loss(**arguments)  # type: ignore[arg-type]


def test_update_rejects_wrong_runtime_objects() -> None:
    network = CheckersNetwork()
    optimizer = torch.optim.Adam(network.parameters())
    rollout = _single_transition_rollout()
    minibatch = build_ppo_minibatch(rollout)  # type: ignore[arg-type]
    config = PPOConfig()

    with pytest.raises(TypeError, match="network must be a CheckersNetwork"):
        ppo_minibatch_update(
            network=object(),  # type: ignore[arg-type]
            optimizer=optimizer,
            minibatch=minibatch,
            config=config,
        )
    with pytest.raises(TypeError, match="optimizer must be a torch Optimizer"):
        ppo_minibatch_update(
            network=network,
            optimizer=object(),  # type: ignore[arg-type]
            minibatch=minibatch,
            config=config,
        )
    with pytest.raises(TypeError, match="minibatch must be a PPOMinibatch"):
        ppo_minibatch_update(
            network=network,
            optimizer=optimizer,
            minibatch=object(),  # type: ignore[arg-type]
            config=config,
        )
    with pytest.raises(TypeError, match="config must be a PPOConfig"):
        ppo_minibatch_update(
            network=network,
            optimizer=optimizer,
            minibatch=minibatch,
            config=object(),  # type: ignore[arg-type]
        )
