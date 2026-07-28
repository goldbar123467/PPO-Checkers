"""T7 engine-derived forced-win oracles for actor-relative value targets."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from checkers.env.checkers_env import CheckersEnv
from checkers.env.masking import step_to_action
from checkers.rl.gae import compute_two_player_gae
from checkers.rules.moves import Step, legal_steps
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome, TerminationReason

MIN_MULTI_JUMP_STEPS = 2


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


@dataclass(frozen=True, slots=True)
class _ForcedWin:
    name: str
    initial: State
    steps: tuple[Step, ...]
    actors: tuple[PlayerId, ...]
    move_completed: tuple[bool, ...]
    sigmas: tuple[int, ...]
    expected_targets: tuple[float, ...]
    outcome: Outcome


FORCED_WINS = (
    _ForcedWin(
        name="length-3-r6.2",
        initial=State(
            men=(_mask(13, 32), _mask(26)),
            kings=(_mask(10), _mask(7, 29)),
            side_to_move=PlayerId.WHITE,
        ),
        steps=(
            _step(7, 14, 10),
            _step(13, 17),
            _step(14, 21, 17),
        ),
        actors=(PlayerId.WHITE, PlayerId.RED, PlayerId.WHITE),
        move_completed=(True, True, True),
        sigmas=(-1, -1, -1),
        expected_targets=(1.0, -1.0, 1.0),
        outcome=Outcome(winner=PlayerId.WHITE, reason=TerminationReason.NO_LEGAL_MOVE),
    ),
    _ForcedWin(
        name="length-5-terminal-multijump",
        initial=State(
            men=(0, 0),
            kings=(_mask(6, 30), _mask(1, 11, 18, 26)),
            side_to_move=PlayerId.WHITE,
        ),
        steps=(
            _step(1, 10, 6),
            _step(30, 23, 26),
            _step(23, 14, 18),
            _step(14, 7, 10),
            _step(7, 16, 11),
        ),
        actors=(
            PlayerId.WHITE,
            PlayerId.RED,
            PlayerId.RED,
            PlayerId.RED,
            PlayerId.RED,
        ),
        move_completed=(True, False, False, False, True),
        sigmas=(-1, 1, 1, 1, -1),
        expected_targets=(-1.0, 1.0, 1.0, 1.0, 1.0),
        outcome=Outcome(winner=PlayerId.RED, reason=TerminationReason.NO_PIECES),
    ),
    _ForcedWin(
        name="length-7-forced-win",
        initial=State(
            men=(_mask(5, 10, 11), _mask(6)),
            kings=(_mask(1), _mask(16, 21)),
            side_to_move=PlayerId.WHITE,
        ),
        steps=(
            _step(16, 7, 11),
            _step(7, 14, 10),
            _step(1, 10, 6),
            _step(10, 17, 14),
            _step(21, 14, 17),
            _step(5, 9),
            _step(14, 5, 9),
        ),
        actors=(
            PlayerId.WHITE,
            PlayerId.WHITE,
            PlayerId.RED,
            PlayerId.RED,
            PlayerId.WHITE,
            PlayerId.RED,
            PlayerId.WHITE,
        ),
        move_completed=(False, True, False, True, True, True, True),
        sigmas=(1, -1, 1, -1, -1, -1, -1),
        expected_targets=(1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0),
        outcome=Outcome(winner=PlayerId.WHITE, reason=TerminationReason.NO_PIECES),
    ),
)


@pytest.mark.parametrize("fixture", FORCED_WINS, ids=lambda fixture: fixture.name)
def test_t7_forced_win_targets_are_exact_in_each_actor_frame(fixture: _ForcedWin) -> None:
    env = CheckersEnv(initial_state=fixture.initial)
    rewards: list[float] = []
    dones: list[bool] = []
    observed_sigmas: list[int] = []
    observed_actors: list[PlayerId] = []
    observed_completion: list[bool] = []

    assert not env.terminated
    for expected_step, expected_actor in zip(fixture.steps, fixture.actors, strict=True):
        assert legal_steps(env.state) == (expected_step,)
        actor = env.state.side_to_move
        _, reward, terminated, truncated, info = env.step(step_to_action(env.state, expected_step))
        observed_actors.append(actor)
        rewards.append(reward)
        dones.append(terminated)
        observed_sigmas.append(1 if env.state.side_to_move is actor else -1)
        observed_completion.append(bool(info["move_completed"]))

        assert actor is expected_actor
        assert info["actor"] is expected_actor
        assert truncated is False

    expected_rewards = (0.0,) * (len(fixture.steps) - 1) + (1.0,)
    expected_dones = (False,) * (len(fixture.steps) - 1) + (True,)
    assert tuple(observed_actors) == fixture.actors
    assert tuple(observed_completion) == fixture.move_completed
    assert tuple(observed_sigmas) == fixture.sigmas
    assert tuple(rewards) == expected_rewards
    assert tuple(dones) == expected_dones
    assert env.outcome == fixture.outcome
    assert fixture.expected_targets == tuple(
        float(fixture.outcome.score_for(actor)) for actor in fixture.actors
    )

    output = compute_two_player_gae(
        rewards=torch.tensor(rewards, dtype=torch.float64),
        values=torch.zeros(len(fixture.steps), dtype=torch.float64),
        dones=torch.tensor(dones),
        sigmas=torch.tensor(observed_sigmas, dtype=torch.int8),
        bootstrap_value=torch.tensor(0.0, dtype=torch.float64),
        gamma=1.0,
        gae_lambda=1.0,
    )
    expected = torch.tensor(fixture.expected_targets, dtype=torch.float64)

    assert torch.equal(output.advantages, expected)
    assert torch.equal(output.returns, expected)


def test_t7_fixture_set_covers_distinct_terminal_multijump_and_r6_2_paths() -> None:
    r6_2 = next(
        fixture
        for fixture in FORCED_WINS
        if fixture.outcome.reason is TerminationReason.NO_LEGAL_MOVE
    )
    terminal_multijump = next(
        fixture
        for fixture in FORCED_WINS
        if len(fixture.move_completed) >= MIN_MULTI_JUMP_STEPS
        and not fixture.move_completed[-2]
        and fixture.move_completed[-1]
        and fixture.actors[-2] is fixture.actors[-1]
    )

    assert r6_2.name != terminal_multijump.name
    assert tuple(len(fixture.steps) for fixture in FORCED_WINS) == (3, 5, 7)
