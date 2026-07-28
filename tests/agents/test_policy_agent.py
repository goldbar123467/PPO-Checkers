"""Learned policy adapter tests for greedy and privately seeded sampled arena play."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from checkers.agents.base import NoLegalActionError
from checkers.agents.policy_agent import PolicyAgent, PolicyMode
from checkers.env.masking import legal_action_map
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import PlayerId, State


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _forced_state() -> State:
    return State(
        men=(_mask(9), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _terminal_state() -> State:
    return State(men=(0, _mask(32)), kings=(0, 0), side_to_move=PlayerId.RED)


@pytest.mark.parametrize("mode", ["greedy", "sampled"])
def test_policy_agent_returns_exact_sole_legal_action_and_restores_mode(mode: str) -> None:
    network = CheckersNetwork()
    network.train()
    agent = PolicyAgent(network=network, mode=cast(PolicyMode, mode), seed=7)
    state = _forced_state()

    action = agent.select_action(state)

    assert action == next(iter(legal_action_map(state)))
    assert network.training
    assert agent.name == f"policy-{mode}"


def test_sampled_policy_uses_private_replayable_generator_without_global_rng_drift() -> None:
    torch.manual_seed(11)
    network = CheckersNetwork()
    first = PolicyAgent(network=network, mode="sampled", seed=99)
    second = PolicyAgent(network=network, mode="sampled", seed=99)
    before = torch.get_rng_state()

    first_actions = tuple(first.select_action(State.initial()) for _ in range(8))
    second_actions = tuple(second.select_action(State.initial()) for _ in range(8))

    assert first_actions == second_actions
    assert torch.equal(torch.get_rng_state(), before)
    assert all(action in legal_action_map(State.initial()) for action in first_actions)


def test_policy_agent_rejects_terminal_state_and_invalid_constructor_values() -> None:
    network = CheckersNetwork()
    with pytest.raises(NoLegalActionError, match="no legal action"):
        PolicyAgent(network=network, mode="greedy", seed=1).select_action(_terminal_state())
    with pytest.raises(TypeError, match="network"):
        PolicyAgent(network=object(), mode="greedy", seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        PolicyAgent(network=network, mode=cast(PolicyMode, "invalid"), seed=1)
    with pytest.raises(TypeError, match="seed"):
        PolicyAgent(network=network, mode="greedy", seed=True)
