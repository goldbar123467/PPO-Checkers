"""Deterministic baseline-agent behavior and search-unit tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from checkers.agents.base import NoLegalActionError
from checkers.agents.greedy_agent import GreedyAgent, material_score
from checkers.agents.minimax_agent import MinimaxAgent, SearchStats
from checkers.agents.random_agent import RandomAgent
from checkers.env.masking import action_to_step, legal_action_mask, step_to_action
from checkers.rules.moves import Step, apply_step
from checkers.rules.state import PlayerId, State

KING_VALUE = 2
EXPECTED_SEARCH_NODES = 3
THREE_JUMP_CONTINUATIONS = 2
THREE_JUMP_ENVIRONMENT_DEPTH = 3


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def _terminal_state() -> State:
    return State(
        men=(_mask(9), 0),
        kings=(0, 0),
        side_to_move=PlayerId.WHITE,
    )


def _two_jump_state() -> State:
    return State(
        men=(_mask(9, 11), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _three_jump_state() -> State:
    return State(
        men=(_mask(10, 12, 19), _mask(13, 21)),
        kings=(_mask(27), _mask(6)),
        side_to_move=PlayerId.WHITE,
        no_progress=(0, 1),
        ply=55,
    )


def test_random_agent_is_seed_reproducible_and_always_legal() -> None:
    first = RandomAgent(seed=20260728)
    second = RandomAgent(seed=20260728)
    state = State.initial()

    first_actions = [first.select_action(state) for _ in range(100)]
    second_actions = [second.select_action(state) for _ in range(100)]

    assert first_actions == second_actions
    assert len(set(first_actions)) > 1
    assert all(legal_action_mask(state)[action] for action in first_actions)
    assert first.name == "random"


def test_random_agent_handles_forced_continuation_mask() -> None:
    continuation = apply_step(_two_jump_state(), _step(9, 18, 14)).after
    agent = RandomAgent(seed=1)

    actions = {agent.select_action(continuation) for _ in range(20)}

    assert actions == {step_to_action(continuation, _step(18, 25, 22))}


def test_material_score_uses_man_one_king_two_and_explicit_perspective() -> None:
    state = State(
        men=(_mask(9, 10), _mask(24)),
        kings=(_mask(14), _mask(19, 20)),
        side_to_move=PlayerId.RED,
    )

    assert material_score(state, PlayerId.RED) == -1
    assert material_score(state, PlayerId.WHITE) == 1


def test_material_score_treats_pending_capture_as_already_won_material() -> None:
    continuation = apply_step(_two_jump_state(), _step(9, 18, 14)).after

    assert continuation.captured_pending == _mask(14)
    assert material_score(continuation, PlayerId.RED) == 0
    assert material_score(continuation, PlayerId.WHITE) == 0


def test_greedy_prefers_capturing_a_king_over_a_man() -> None:
    state = State(
        men=(0, _mask(18)),
        kings=(_mask(14), _mask(17)),
        side_to_move=PlayerId.RED,
    )
    agent = GreedyAgent(seed=4)

    selected = action_to_step(state, agent.select_action(state))

    assert selected == _step(14, 21, 17)


def test_greedy_seeded_tie_break_is_reproducible_and_legal() -> None:
    first = GreedyAgent(seed=9)
    second = GreedyAgent(seed=9)
    state = State.initial()

    first_actions = [first.select_action(state) for _ in range(30)]
    second_actions = [second.select_action(state) for _ in range(30)]

    assert first_actions == second_actions
    assert len(set(first_actions)) > 1
    assert all(legal_action_mask(state)[action] for action in first_actions)
    assert first.name == "greedy"


def test_minimax_depth_one_matches_best_material_successor() -> None:
    state = State(
        men=(0, _mask(18)),
        kings=(_mask(14), _mask(17)),
        side_to_move=PlayerId.RED,
    )
    agent = MinimaxAgent(depth=1, seed=4)

    selected = action_to_step(state, agent.select_action(state))

    assert selected == _step(14, 21, 17)
    assert agent.name == "minimax(1)"


def test_minimax_searches_forced_sequence_to_boundary_without_spending_extra_depth() -> None:
    state = _two_jump_state()
    agent = MinimaxAgent(depth=1, seed=2)

    action = agent.select_action(state)

    selected = action_to_step(state, action)
    first_transition = apply_step(state, selected)
    assert first_transition.move_completed is False
    assert agent.last_stats.completed_move_edges > 0
    assert agent.last_stats.continuation_edges > 0
    assert agent.last_stats.max_environment_step_depth > agent.depth


def test_minimax_returns_the_only_forced_continuation() -> None:
    continuation = apply_step(_two_jump_state(), _step(9, 18, 14)).after
    agent = MinimaxAgent(depth=2, seed=5)

    assert action_to_step(continuation, agent.select_action(continuation)) == _step(18, 25, 22)


def test_minimax_counts_nested_continuations_inside_a_three_jump_move() -> None:
    agent = MinimaxAgent(depth=1, seed=0)

    selected = action_to_step(
        _three_jump_state(),
        agent.select_action(_three_jump_state()),
    )

    assert selected == _step(6, 15, 10)
    assert agent.last_stats.continuation_edges >= THREE_JUMP_CONTINUATIONS
    assert agent.last_stats.max_environment_step_depth >= THREE_JUMP_ENVIRONMENT_DEPTH


def test_minimax_reaches_terminal_child_before_material_leaf() -> None:
    state = State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )
    agent = MinimaxAgent(depth=1, seed=0)

    assert action_to_step(state, agent.select_action(state)) == _step(9, 18, 14)
    assert agent.last_stats.nodes == 1


def test_minimax_transposition_cache_is_exercised_by_depth_three_opening() -> None:
    agent = MinimaxAgent(depth=3, seed=0)

    agent.select_action(State.initial())

    assert agent.last_stats.cache_hits > 0
    assert agent.last_stats.completed_move_edges == agent.last_stats.nodes


def test_search_stats_are_immutable_nonnegative_counts() -> None:
    stats = SearchStats(
        nodes=3,
        cache_hits=1,
        completed_move_edges=2,
        continuation_edges=1,
        max_environment_step_depth=2,
    )

    assert stats.nodes == EXPECTED_SEARCH_NODES
    with pytest.raises((TypeError, ValueError), match="nodes"):
        SearchStats(
            nodes=-1,
            cache_hits=0,
            completed_move_edges=0,
            continuation_edges=0,
            max_environment_step_depth=0,
        )


@pytest.mark.parametrize(
    "agent_factory",
    [
        lambda: RandomAgent(seed=0),
        lambda: GreedyAgent(seed=0),
        lambda: MinimaxAgent(depth=1, seed=0),
    ],
)
def test_all_baseline_agents_reject_terminal_state(
    agent_factory: Callable[[], RandomAgent | GreedyAgent | MinimaxAgent],
) -> None:
    with pytest.raises(NoLegalActionError, match="legal"):
        agent_factory().select_action(_terminal_state())


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RandomAgent(seed=cast(int, True)), "seed"),
        (lambda: GreedyAgent(seed=cast(int, True)), "seed"),
        (lambda: MinimaxAgent(depth=0), "depth"),
        (lambda: MinimaxAgent(depth=cast(int, True)), "depth"),
        (
            lambda: SearchStats(
                nodes=0,
                cache_hits=1,
                completed_move_edges=0,
                continuation_edges=0,
                max_environment_step_depth=0,
            ),
            "cache_hits",
        ),
    ],
)
def test_agent_configuration_rejects_invalid_values(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


@pytest.mark.parametrize(
    "score_player",
    [cast(PlayerId, 0), cast(PlayerId, "RED")],
)
def test_material_score_requires_player_enum(score_player: PlayerId) -> None:
    with pytest.raises(TypeError, match="PlayerId"):
        material_score(State.initial(), score_player)


def test_agent_select_action_requires_state() -> None:
    agents = (RandomAgent(), GreedyAgent(), MinimaxAgent(depth=1))
    for agent in agents:
        with pytest.raises(TypeError, match="State"):
            agent.select_action(cast(State, "state"))


def test_king_weight_constant_matches_documented_simple_evaluator() -> None:
    state = State(
        men=(0, _mask(24, 25)),
        kings=(_mask(14), 0),
        side_to_move=PlayerId.RED,
    )

    assert material_score(state, PlayerId.RED) == KING_VALUE - 2
