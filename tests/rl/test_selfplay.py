"""Production-engine self-play collection and chronology integration tests."""

from __future__ import annotations

import copy

import torch

from checkers.config import RunConfig
from checkers.rl.determinism import seed_everything
from checkers.rl.networks import CheckersNetwork
from checkers.rl.selfplay import SelfPlayCollector
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import TerminationReason

EXPECTED_TRANSITIONS = 2


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _config() -> RunConfig:
    return RunConfig(
        experiment_id="selfplay-unit",
        seed=77,
        device="cpu",
        total_updates=1,
        schedule_horizon_updates=1,
        duration_seconds=None,
        num_envs=2,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        eval_games=2,
    )


def _multijump_state() -> State:
    return State(
        men=(_mask(9), _mask(14, 15, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def _one_jump_win() -> State:
    return State(
        men=(_mask(9), _mask(14)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
    )


def test_s1_s2_s3_collector_preserves_midjump_and_resets_terminal_lane() -> None:
    config = _config()
    seed_everything(config.seed, num_envs=config.num_envs, deterministic=True)
    network = CheckersNetwork()
    collector = SelfPlayCollector(
        config=config,
        network=network,
        initial_states=(_multijump_state(), _one_jump_win()),
    )

    result = collector.collect()

    assert result.rollout.transitions == EXPECTED_TRANSITIONS
    assert result.rollout.actor.tolist() == [int(PlayerId.RED), int(PlayerId.RED)]
    assert result.rollout.sigma.tolist() == [1, -1]
    assert result.rollout.done.tolist() == [False, True]
    assert result.rollout.move_completed.tolist() == [False, True]
    assert result.rollout.reward.tolist() == [0.0, 1.0]
    assert result.rollout.trainable.tolist() == [True, True]
    assert result.rollout.policy_id == ("current", "current")
    assert collector.vector_env.envs[0].state.capture_in_progress
    assert not collector.vector_env.envs[0].terminated
    assert collector.vector_env.envs[1].state == _one_jump_win()
    assert not collector.vector_env.envs[1].terminated
    assert collector.episode_indices == (0, 1)
    assert collector.active_episode_steps == (1, 0)

    assert result.metrics["mask/sample_legality_violations"] == 0.0
    assert result.metrics["mask/oracle_disagreements"] == 0.0
    assert result.metrics["mask/empty_mask_count"] == 0.0
    assert result.metrics["mask/mean_legal_actions"] == 1.0
    assert result.metrics["policy/frac_states_k_eq_1"] == 1.0
    assert result.metrics["env/mean_game_len_steps"] == 1.0
    assert result.metrics["env/mean_game_len_moves"] == 1.0
    assert result.metrics["env/captures_per_game"] == 1.0
    assert result.metrics["env/mean_sequence_len"] == 1.0
    assert result.metrics["env/first_player_win_rate"] == 1.0
    assert result.metrics["env/first_player_frac"] == 2.0 / 3.0
    assert result.completed_games[0].reason is TerminationReason.NO_PIECES


def test_s2_second_rollout_finishes_continuation_with_same_actor() -> None:
    config = _config()
    seed_everything(config.seed, num_envs=config.num_envs, deterministic=True)
    collector = SelfPlayCollector(
        config=config,
        network=CheckersNetwork(),
        initial_states=(_multijump_state(), _one_jump_win()),
    )
    collector.collect()

    second = collector.collect()

    assert second.rollout.actor[0].item() == int(PlayerId.RED)
    assert second.rollout.move_completed[0].item()
    assert second.rollout.sigma[0].item() == -1
    assert not collector.vector_env.envs[0].state.capture_in_progress


def test_s2_collector_record_resumes_midsequence_and_replays_next_rollout() -> None:
    config = _config()
    seed_everything(config.seed, num_envs=config.num_envs, deterministic=True)
    network = CheckersNetwork()
    collector = SelfPlayCollector(
        config=config,
        network=network,
        initial_states=(_multijump_state(), _one_jump_win()),
    )
    collector.collect()
    record = collector.to_record()
    resumed = SelfPlayCollector.from_record(
        config=config,
        network=copy.deepcopy(network),
        record=record,
    )
    torch_state = torch.get_rng_state()

    uninterrupted = collector.collect()
    torch.set_rng_state(torch_state)
    replayed = resumed.collect()

    assert resumed.episode_indices == collector.episode_indices
    assert resumed.active_episode_steps == collector.active_episode_steps
    assert resumed.vector_env.serialize() == collector.vector_env.serialize()
    for field in (
        "obs",
        "legal_mask",
        "action",
        "behaviour_logprob",
        "value",
        "reward",
        "done",
        "actor",
        "sigma",
        "trainable",
        "env_id",
        "move_completed",
        "advantages",
        "returns",
    ):
        assert torch.equal(getattr(replayed.rollout, field), getattr(uninterrupted.rollout, field))
    assert replayed.rollout.policy_id == uninterrupted.rollout.policy_id
    assert replayed.metrics == uninterrupted.metrics
