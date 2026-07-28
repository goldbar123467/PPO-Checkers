"""Independent numerical oracles for perspective-aware GAE."""

from __future__ import annotations

import pytest
import torch

from checkers.rl.gae import GAEOutput, compute_two_player_gae


def test_four_transition_hand_calculation_has_both_perspective_signs() -> None:
    output = compute_two_player_gae(
        rewards=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
        values=torch.tensor([0.2, -0.1, 0.3, 0.4], dtype=torch.float64),
        dones=torch.tensor([False, False, False, True]),
        sigmas=torch.tensor([1, -1, -1, 1], dtype=torch.int8),
        bootstrap_value=torch.tensor(0.7, dtype=torch.float64),
        gamma=0.9,
        gae_lambda=0.8,
    )

    expected_advantages = torch.tensor(
        [0.1536928, 0.61624, -1.092, 0.6],
        dtype=torch.float64,
    )
    expected_returns = torch.tensor(
        [0.3536928, 0.51624, -0.792, 1.0],
        dtype=torch.float64,
    )

    assert isinstance(output, GAEOutput)
    torch.testing.assert_close(output.advantages, expected_advantages, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(output.returns, expected_returns, atol=1e-12, rtol=0.0)


def test_rollout_truncation_bootstraps_with_last_sigma_and_stops_advantage_tail() -> None:
    output = compute_two_player_gae(
        rewards=torch.tensor([0.0]),
        values=torch.tensor([0.25]),
        dones=torch.tensor([False]),
        sigmas=torch.tensor([1], dtype=torch.int8),
        bootstrap_value=torch.tensor(0.75),
        gamma=1.0,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(output.advantages, torch.tensor([0.5]))
    torch.testing.assert_close(output.returns, torch.tensor([0.75]))


def test_terminal_transition_ignores_bootstrap_and_sigma() -> None:
    positive = compute_two_player_gae(
        rewards=torch.tensor([-1.0]),
        values=torch.tensor([0.4]),
        dones=torch.tensor([True]),
        sigmas=torch.tensor([1], dtype=torch.int8),
        bootstrap_value=torch.tensor(999.0),
        gamma=1.0,
        gae_lambda=0.95,
    )
    negative = compute_two_player_gae(
        rewards=torch.tensor([-1.0]),
        values=torch.tensor([0.4]),
        dones=torch.tensor([True]),
        sigmas=torch.tensor([-1], dtype=torch.int8),
        bootstrap_value=torch.tensor(-999.0),
        gamma=1.0,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(positive.advantages, torch.tensor([-1.4]))
    torch.testing.assert_close(positive.returns, torch.tensor([-1.0]))
    assert torch.equal(positive.advantages, negative.advantages)
    assert torch.equal(positive.returns, negative.returns)


def test_sigma_degenerate_case_is_bitwise_equal_to_reference_gae() -> None:
    rewards = torch.tensor([0.1, -0.2, 0.3, 0.0, 1.0])
    values = torch.tensor([0.2, 0.4, -0.1, 0.3, 0.25])
    dones = torch.tensor([False, False, False, False, True])
    gamma = 0.97
    gae_lambda = 0.91
    bootstrap = torch.tensor(-0.75)

    reference = torch.zeros_like(values)
    next_advantage = torch.zeros_like(bootstrap)
    for index in range(len(rewards) - 1, -1, -1):
        next_value = bootstrap if index == len(rewards) - 1 else values[index + 1]
        nonterminal = (~dones[index]).to(values.dtype)
        delta = rewards[index] + gamma * nonterminal * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        reference[index] = next_advantage

    output = compute_two_player_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        sigmas=torch.ones_like(dones, dtype=torch.int8),
        bootstrap_value=bootstrap,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert torch.equal(output.advantages, reference)
    assert torch.equal(output.returns, reference + values)


def test_colour_swap_negates_advantages_and_returns() -> None:
    rewards = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    values = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)
    dones = torch.tensor([False, False, True])
    sigmas = torch.tensor([-1, 1, -1], dtype=torch.int8)
    bootstrap = torch.tensor(0.0, dtype=torch.float64)
    original = compute_two_player_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        sigmas=sigmas,
        bootstrap_value=bootstrap,
        gamma=1.0,
        gae_lambda=0.95,
    )
    swapped = compute_two_player_gae(
        rewards=-rewards,
        values=-values,
        dones=dones,
        sigmas=sigmas,
        bootstrap_value=-bootstrap,
        gamma=1.0,
        gae_lambda=0.95,
    )

    assert torch.equal(swapped.advantages, -original.advantages)
    assert torch.equal(swapped.returns, -original.returns)


def test_batched_environment_lanes_are_independent() -> None:
    output = compute_two_player_gae(
        rewards=torch.tensor([[0.0, 0.0], [1.0, -1.0]]),
        values=torch.zeros((2, 2)),
        dones=torch.tensor([[False, False], [True, True]]),
        sigmas=torch.tensor([[-1, 1], [1, 1]], dtype=torch.int8),
        bootstrap_value=torch.tensor([5.0, 7.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(output.advantages, torch.tensor([[-1.0, -1.0], [1.0, -1.0]]))
    assert torch.equal(output.returns, output.advantages)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"rewards": [0.0]}, TypeError, "rewards must be a Tensor"),
        ({"rewards": torch.tensor([0])}, TypeError, "rewards must be floating"),
        ({"values": torch.tensor([0])}, TypeError, "values must be floating"),
        ({"dones": torch.tensor([0.0])}, TypeError, "dones must have dtype bool"),
        ({"sigmas": torch.tensor([True])}, TypeError, "sigmas must be real"),
        ({"sigmas": torch.tensor([1.0 + 0.0j])}, TypeError, "sigmas must be real"),
        ({"bootstrap_value": torch.tensor(0)}, TypeError, "bootstrap_value must be floating"),
        ({"sigmas": torch.tensor([0])}, ValueError, "sigmas must contain only"),
        ({"bootstrap_value": torch.tensor([0.0])}, ValueError, "bootstrap_value shape"),
        ({"rewards": torch.tensor([])}, ValueError, "at least one transition"),
        ({"rewards": torch.tensor(0.0)}, ValueError, "at least one transition"),
        ({"values": torch.tensor([0.0, 1.0])}, ValueError, "same shape"),
        ({"values": torch.tensor([0.0], dtype=torch.float64)}, ValueError, "dtype and device"),
        ({"rewards": torch.tensor([float("nan")])}, ValueError, "rewards must be finite"),
        ({"values": torch.tensor([float("inf")])}, ValueError, "values must be finite"),
        (
            {"bootstrap_value": torch.tensor(float("nan"))},
            ValueError,
            "bootstrap_value must be finite",
        ),
        ({"gamma": "1.0"}, TypeError, "gamma must be numeric"),
        ({"gae_lambda": True}, TypeError, "gae_lambda must be numeric"),
        ({"gamma": -0.1}, ValueError, "gamma must be in"),
        ({"gamma": float("nan")}, ValueError, "gamma must be in"),
        ({"gae_lambda": 1.1}, ValueError, "gae_lambda must be in"),
    ],
)
def test_invalid_gae_inputs_raise(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "rewards": torch.tensor([0.0]),
        "values": torch.tensor([0.0]),
        "dones": torch.tensor([False]),
        "sigmas": torch.tensor([1], dtype=torch.int8),
        "bootstrap_value": torch.tensor(0.0),
        "gamma": 1.0,
        "gae_lambda": 0.95,
    }
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        compute_two_player_gae(**arguments)  # type: ignore[arg-type]
