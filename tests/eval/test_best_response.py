"""Short-budget policy-gradient best-response evidence tests."""

from __future__ import annotations

import torch

from checkers.eval.best_response import train_short_best_response
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import PlayerId, State

TRAINING_GAMES = 2
EVALUATION_GAMES = 2


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _one_jump_state() -> State:
    return State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def test_short_best_response_trains_only_clone_and_returns_measured_match() -> None:
    torch.manual_seed(211)
    frozen = CheckersNetwork()
    frozen.train()
    weights_before = {name: tensor.detach().clone() for name, tensor in frozen.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()

    result = train_short_best_response(
        frozen_network=frozen,
        training_games=TRAINING_GAMES,
        evaluation_games=EVALUATION_GAMES,
        seed=223,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        max_plies=16,
        repetition_draws=True,
        initial_state=_one_jump_state(),
    )

    assert result.evidence.status == "MEASURED"
    assert result.evidence.training_steps == result.training_decisions
    assert 0.0 <= result.evidence.score <= 1.0
    assert result.match.score.score == result.evidence.score
    assert result.match.games == EVALUATION_GAMES
    assert result.training_games == TRAINING_GAMES
    assert result.training_decisions > 0
    assert result.optimizer_steps > 0
    assert result.frozen_sha256_before == result.frozen_sha256_after
    assert all(
        torch.equal(weights_before[name], tensor) for name, tensor in frozen.state_dict().items()
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert frozen.training
