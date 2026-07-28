"""Learned-policy fixed-anchor and two-policy population evaluation tests."""

from __future__ import annotations

import torch

from checkers.eval.policy_eval import ExploitabilityEvidence, evaluate_development_policy
from checkers.metrics import EVALUATION_METRIC_KEYS
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import PlayerId, State

SMOKE_GAMES = 2
PAYOFF_CELLS = 4


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _one_jump_state() -> State:
    return State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def test_learned_policy_eval_reports_every_scalar_and_literal_payoff_rows() -> None:
    torch.manual_seed(83)
    network = CheckersNetwork()
    network.train()
    initial = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()

    result = evaluate_development_policy(
        network=network,
        initial_model_state=initial,
        games=SMOKE_GAMES,
        seed=101,
        max_plies=16,
        repetition_draws=True,
        initial_state=_one_jump_state(),
        exploitability=ExploitabilityEvidence.not_evaluated(),
    )

    assert set(result.scalar_metrics) == EVALUATION_METRIC_KEYS - {"eval/payoff_matrix"}
    for anchor in ("random", "greedy", "minimax2"):
        assert result.scalar_metrics[f"eval/vs_{anchor}_games"] == SMOKE_GAMES
        assert 0.0 <= result.scalar_metrics[f"eval/vs_{anchor}"] <= 1.0
        assert (
            result.scalar_metrics[f"eval/vs_{anchor}_ci_low"]
            <= result.scalar_metrics[f"eval/vs_{anchor}"]
        )
        assert (
            result.scalar_metrics[f"eval/vs_{anchor}_ci_high"]
            >= result.scalar_metrics[f"eval/vs_{anchor}"]
        )
    assert result.scalar_metrics["eval/exploitability_proxy"] == -1.0
    assert result.exploitability_status == "NOT_EVALUATED"
    assert len(result.payoff_rows) == PAYOFF_CELLS
    assert {row["row_agent"] for row in result.payoff_rows} == {"current", "initial"}
    assert {row["column_agent"] for row in result.payoff_rows} == {"current", "initial"}
    assert result.scalar_metrics["eval/three_cycle_count"] == 0.0
    assert -1.0 <= result.scalar_metrics["eval/greedy_vs_sampled_delta"] <= 1.0
    assert len(result.game_rows) == SMOKE_GAMES * 5
    assert {row["match"] for row in result.game_rows} == {
        "vs_random",
        "vs_greedy",
        "vs_minimax2",
        "sampled_vs_random",
        "current_vs_initial",
    }
    assert {row["perspective_result"] for row in result.game_rows} <= {
        "win",
        "draw",
        "loss",
    }
    assert all(isinstance(row["moves"], str) for row in result.game_rows)
    assert network.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
