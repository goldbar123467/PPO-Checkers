"""Short-budget policy-gradient best-response proxy against a frozen checkpoint."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass

import torch

from checkers.agents.policy_agent import PolicyAgent
from checkers.env.checkers_env import CheckersEnv
from checkers.env.encoding import encode_observation
from checkers.env.masking import legal_action_mask
from checkers.eval.arena import AgentSpec, MatchResult, play_balanced_match
from checkers.eval.policy_eval import ExploitabilityEvidence
from checkers.rl.determinism import derive_stream_seed
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import CheckersNetwork
from checkers.rules.state import PlayerId, State

MIN_BALANCED_GAMES = 2
UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class BestResponseResult:
    """Training audit, frozen-policy digests, and the balanced proxy match."""

    evidence: ExploitabilityEvidence
    match: MatchResult
    training_games: int
    training_decisions: int
    optimizer_steps: int
    frozen_sha256_before: str
    frozen_sha256_after: str
    best_response_sha256: str


def _balanced_games(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < MIN_BALANCED_GAMES or value % MIN_BALANCED_GAMES:
        raise ValueError(f"{name} must be a positive even colour-balanced count")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return checked


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    if not 0 <= value <= UINT64_MAX:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return value


def _model_sha256(network: CheckersNetwork) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sample_best_response_action(
    *,
    network: CheckersNetwork,
    state: State,
    generator: torch.Generator,
) -> tuple[int, torch.Tensor]:
    device = next(network.parameters()).device
    observation = torch.as_tensor(
        encode_observation(state),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    mask = torch.as_tensor(
        legal_action_mask(state),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    output = network(observation)
    distribution = MaskedCategorical(logits=output.logits, legal_mask=mask)
    action = torch.multinomial(
        distribution.probs,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(1)
    return int(action.item()), distribution.log_prob(action).squeeze(0)


def _train_game(  # noqa: PLR0913
    *,
    best_response: CheckersNetwork,
    frozen_network: CheckersNetwork,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    best_response_side: PlayerId,
    seed: int,
    max_grad_norm: float,
    max_plies: int,
    repetition_draws: bool,
    initial_state: State,
) -> tuple[int, bool]:
    environment = CheckersEnv(
        max_plies=max_plies,
        repetition_draws=repetition_draws,
        initial_state=initial_state,
    )
    opponent = PolicyAgent(
        network=frozen_network,
        mode="greedy",
        seed=derive_stream_seed(seed, 1),
        name="frozen",
    )
    log_probabilities: list[torch.Tensor] = []
    try:
        environment.reset(seed=derive_stream_seed(seed, 2))
        while not environment.terminated:
            if environment.state.side_to_move is best_response_side:
                action, log_probability = _sample_best_response_action(
                    network=best_response,
                    state=environment.state,
                    generator=generator,
                )
                log_probabilities.append(log_probability)
            else:
                action = opponent.select_action(environment.state)
            environment.step(action)
        outcome = environment.outcome
        if outcome is None:
            raise RuntimeError("best-response training game terminated without an outcome")
        game_score = outcome.score_for(best_response_side)
        if not log_probabilities or game_score == 0:
            return len(log_probabilities), False
        loss = -float(game_score) * torch.stack(log_probabilities).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(best_response.parameters(), max_grad_norm)
        optimizer.step()
        return len(log_probabilities), True
    finally:
        environment.close()


def train_short_best_response(  # noqa: PLR0913
    *,
    frozen_network: CheckersNetwork,
    training_games: int,
    evaluation_games: int,
    seed: int,
    learning_rate: float,
    max_grad_norm: float,
    max_plies: int,
    repetition_draws: bool,
    initial_state: State | None = None,
) -> BestResponseResult:
    """Train a no-shaping REINFORCE clone against one frozen policy and score it.

    Args:
        frozen_network: Checkpoint policy held fixed throughout training and evaluation.
        training_games: Even number of games, alternating which colour the clone controls.
        evaluation_games: Even number of post-training colour-balanced arena games.
        seed: Root seed for private sampling, opponent, environment, and arena streams.
        learning_rate: Adam learning rate for the cloned policy only.
        max_grad_norm: Global gradient clipping threshold.
        max_plies: Declared engine-variant game bound.
        repetition_draws: Whether arena threefold repetition is enabled.
        initial_state: Optional completed-move start state, primarily for tiny validation.

    Returns:
        Measured score, full arena records, step counts, and before/after policy hashes.

    Raises:
        TypeError: If an input has an invalid runtime type.
        ValueError: If counts, seed, bounds, or the initial state are invalid.
        RuntimeError: If no trainable decision was observed or the frozen policy changed.
    """

    if not isinstance(frozen_network, CheckersNetwork):
        raise TypeError("frozen_network must be a CheckersNetwork")
    checked_training_games = _balanced_games(training_games, "training_games")
    checked_evaluation_games = _balanced_games(evaluation_games, "evaluation_games")
    checked_seed = _seed(seed)
    checked_learning_rate = _positive(learning_rate, "learning_rate")
    checked_max_grad_norm = _positive(max_grad_norm, "max_grad_norm")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    if not isinstance(repetition_draws, bool):
        raise TypeError("repetition_draws must be bool")
    configured_initial = State.initial() if initial_state is None else initial_state
    if not isinstance(configured_initial, State):
        raise TypeError("initial_state must be a State or None")
    if configured_initial.capture_in_progress:
        raise ValueError("initial_state must be a completed-move boundary")

    frozen_before = _model_sha256(frozen_network)
    best_response = deepcopy(frozen_network)
    best_response.train()
    device = next(best_response.parameters()).device
    generator = torch.Generator(device=device).manual_seed(derive_stream_seed(checked_seed, 0))
    optimizer = torch.optim.Adam(best_response.parameters(), lr=checked_learning_rate)
    training_decisions = 0
    optimizer_steps = 0
    for game_index in range(checked_training_games):
        decisions, stepped = _train_game(
            best_response=best_response,
            frozen_network=frozen_network,
            optimizer=optimizer,
            generator=generator,
            best_response_side=(
                PlayerId.RED if game_index % MIN_BALANCED_GAMES == 0 else PlayerId.WHITE
            ),
            seed=derive_stream_seed(checked_seed, 10 + game_index),
            max_grad_norm=checked_max_grad_norm,
            max_plies=max_plies,
            repetition_draws=repetition_draws,
            initial_state=configured_initial,
        )
        training_decisions += decisions
        optimizer_steps += int(stepped)
    if training_decisions < 1:
        raise RuntimeError("best-response budget observed no trainable decisions")

    match = play_balanced_match(
        first=AgentSpec(
            name="best-response",
            factory=lambda value: PolicyAgent(
                network=best_response,
                mode="greedy",
                seed=value,
                name="best-response",
            ),
        ),
        second=AgentSpec(
            name="frozen",
            factory=lambda value: PolicyAgent(
                network=frozen_network,
                mode="greedy",
                seed=value,
                name="frozen",
            ),
        ),
        games=checked_evaluation_games,
        seed=derive_stream_seed(checked_seed, 100_000),
        initial_state=configured_initial,
        max_plies=max_plies,
        repetition_draws=repetition_draws,
    )
    frozen_after = _model_sha256(frozen_network)
    if frozen_after != frozen_before:
        raise RuntimeError("frozen policy changed during best-response measurement")
    evidence = ExploitabilityEvidence.measured(
        score=match.score.score,
        training_steps=training_decisions,
    )
    return BestResponseResult(
        evidence=evidence,
        match=match,
        training_games=checked_training_games,
        training_decisions=training_decisions,
        optimizer_steps=optimizer_steps,
        frozen_sha256_before=frozen_before,
        frozen_sha256_after=frozen_after,
        best_response_sha256=_model_sha256(best_response),
    )
