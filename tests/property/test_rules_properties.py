"""Reachable-state properties and the deterministic 50k-step PR fuzz tier."""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from checkers.rules.board import bit, coord
from checkers.rules.moves import apply_step, legal_steps, undo_step
from checkers.rules.notation import parse_state, serialize_state
from checkers.rules.oracle import oracle_legal_steps
from checkers.rules.state import State
from checkers.rules.zobrist import incremental_state_key, state_key

FUZZ_SEED = 20_260_727
PR_FUZZ_STEPS = 50_000
MAX_PLIES = 512
HYPOTHESIS_EXAMPLES = 200
MAX_TRAJECTORY_CHOICES = 80


def _piece_count(state: State) -> int:
    return state.occupied.bit_count()


def _assert_man_step_is_forward(state: State, origin: int, destination: int) -> None:
    actor = int(state.side_to_move)
    if not state.men[actor] & bit(origin):
        return
    origin_row, _ = coord(origin)
    destination_row, _ = coord(destination)
    expected_sign = 1 if actor == 0 else -1
    assert (destination_row - origin_row) * expected_sign > 0


def _assert_step_invariants(state: State, choice: int) -> State:
    fast = legal_steps(state)
    oracle = oracle_legal_steps(state)
    assert fast == oracle
    assert len(fast) == len(set(fast))
    assert fast
    if any(step.is_capture for step in fast):
        assert all(step.is_capture for step in fast)

    for legal in fast:
        _assert_man_step_is_forward(state, legal.origin, legal.destination)

    step = fast[choice % len(fast)]
    actor = int(state.side_to_move)
    opponent = int(state.side_to_move.opponent)
    was_man = bool(state.men[actor] & bit(step.origin))
    was_king = bool(state.kings[actor] & bit(step.origin))
    before_key = state_key(state)
    transition = apply_step(state, step)
    after = transition.after

    assert undo_step(transition) == state
    after_key = incremental_state_key(before_key, state, after)
    assert after_key == state_key(after)
    assert incremental_state_key(after_key, after, undo_step(transition)) == before_key
    assert after.ply == state.ply + 1
    assert parse_state(serialize_state(after)) == after

    removed = 0
    if transition.move_completed and step.is_capture:
        assert step.captured is not None
        removed = (state.captured_pending | bit(step.captured)).bit_count()
    assert _piece_count(state) - _piece_count(after) == removed

    if was_king:
        assert after.kings[actor] & bit(step.destination)
    if was_man:
        destination_row, _ = coord(step.destination)
        promotes = destination_row == (7 if actor == 0 else 0)
        destination_board = after.kings[actor] if promotes else after.men[actor]
        assert destination_board & bit(step.destination)

    if transition.move_completed:
        assert after.side_to_move is state.side_to_move.opponent
        assert not after.capture_in_progress
        assert after.captured_pending == 0
        expected_counter = 0 if was_man or step.is_capture else state.no_progress[actor] + 1
        assert after.no_progress[actor] == expected_counter
        assert after.no_progress[opponent] == state.no_progress[opponent]
    else:
        assert after.side_to_move is state.side_to_move
        assert after.capture_in_progress
        assert after.moving_square == step.destination
        assert after.no_progress == state.no_progress
        assert step.captured is not None
        assert after.captured_pending & bit(step.captured)
        assert after.occupied & after.captured_pending == after.captured_pending

    return after


def test_pr_ci_50k_reachable_step_invariants_and_dual_generator_agreement() -> None:
    rng = random.Random(FUZZ_SEED)
    state = State.initial()
    for _ in range(PR_FUZZ_STEPS):
        if state.ply >= MAX_PLIES or not legal_steps(state):
            state = State.initial()
        state = _assert_step_invariants(state, rng.randrange(2**32))


@settings(max_examples=HYPOTHESIS_EXAMPLES, deadline=None, derandomize=True)
@given(st.lists(st.integers(min_value=0, max_value=2**32 - 1), max_size=MAX_TRAJECTORY_CHOICES))
def test_hypothesis_reachable_state_serialization_and_undo(choices: list[int]) -> None:
    state = State.initial()
    for choice in choices:
        if not legal_steps(state):
            break
        state = _assert_step_invariants(state, choice)
