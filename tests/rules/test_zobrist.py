"""Full-state and official-position Zobrist key contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.state import PlayerId, State
from checkers.rules.zobrist import incremental_state_key, position_key, state_key

TERMINAL_KEY_VARIANTS = 3
INITIAL_STATE_KEY = 0xE5E9_55C8_5267_6C2B
INITIAL_POSITION_KEY = 0x8E2A_8A0D_45AE_B292


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _step(origin: int, destination: int, captured: int | None = None) -> Step:
    return Step(
        origin=origin - 1,
        destination=destination - 1,
        captured=None if captured is None else captured - 1,
    )


def test_state_key_separates_midsequence_state_from_same_placement_boundary() -> None:
    initial = State(
        men=(0, _mask(6, 15)),
        kings=(_mask(1), 0),
        side_to_move=PlayerId.RED,
    )
    mid_sequence = apply_step(initial, _step(1, 10, 6)).after
    same_placement = State(
        men=mid_sequence.men,
        kings=mid_sequence.kings,
        side_to_move=mid_sequence.side_to_move,
        no_progress=mid_sequence.no_progress,
        ply=mid_sequence.ply,
    )

    assert state_key(mid_sequence) != state_key(same_placement)
    with pytest.raises(ValueError, match="move boundary"):
        position_key(mid_sequence)
    assert isinstance(position_key(same_placement), int)


def test_state_key_includes_terminal_counters_omitted_by_goal_field_list() -> None:
    below = State(
        men=(0, 0),
        kings=(_mask(14), _mask(24)),
        side_to_move=PlayerId.RED,
        no_progress=(40, 39),
        ply=511,
    )
    no_progress_terminal = State(
        men=below.men,
        kings=below.kings,
        side_to_move=below.side_to_move,
        no_progress=(40, 40),
        ply=below.ply,
    )
    ply_terminal = State(
        men=below.men,
        kings=below.kings,
        side_to_move=below.side_to_move,
        no_progress=below.no_progress,
        ply=512,
    )

    assert (
        len({state_key(below), state_key(no_progress_terminal), state_key(ply_terminal)})
        == TERMINAL_KEY_VARIANTS
    )
    assert position_key(below) == position_key(no_progress_terminal) == position_key(ply_terminal)


def test_state_key_includes_sequence_origin_propagated_to_future_states() -> None:
    first_origin = State(
        men=(_mask(18), _mask(14, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        capture_in_progress=True,
        moving_square=17,
        sequence_origin=8,
        captured_pending=_mask(14),
    )
    second_origin = State(
        men=first_origin.men,
        kings=first_origin.kings,
        side_to_move=first_origin.side_to_move,
        capture_in_progress=True,
        moving_square=first_origin.moving_square,
        sequence_origin=9,
        captured_pending=first_origin.captured_pending,
    )

    assert state_key(first_origin) != state_key(second_origin)


def test_position_key_contains_only_placement_and_side() -> None:
    red = State(
        men=(_mask(9), _mask(24)),
        kings=(_mask(1), 0),
        side_to_move=PlayerId.RED,
    )
    red_with_counters = State(
        men=red.men,
        kings=red.kings,
        side_to_move=red.side_to_move,
        no_progress=(39, 17),
        ply=511,
    )
    white = State(
        men=red.men,
        kings=red.kings,
        side_to_move=PlayerId.WHITE,
    )

    assert position_key(red) == position_key(red_with_counters)
    assert position_key(red) != position_key(white)
    assert state_key(red) != state_key(red_with_counters)


def test_incremental_key_matches_recomputation_and_reverses_over_reachable_steps() -> None:
    state = State.initial()
    key = state_key(state)
    for index in range(300):
        steps = legal_steps(state)
        if not steps:
            state = State.initial()
            key = state_key(state)
            steps = legal_steps(state)
        transition = apply_step(state, steps[index % len(steps)])

        updated = incremental_state_key(key, state, transition.after)

        assert updated == state_key(transition.after)
        assert incremental_state_key(updated, transition.after, state) == key
        state = transition.after
        key = updated


def test_frozen_zobrist_schema_has_stable_known_keys() -> None:
    assert state_key(State.initial()) == INITIAL_STATE_KEY
    assert position_key(State.initial()) == INITIAL_POSITION_KEY


@pytest.mark.parametrize("key_function", [state_key, position_key])
def test_public_key_functions_require_a_state(key_function: Callable[[State], int]) -> None:
    with pytest.raises(TypeError, match="State"):
        key_function(cast(State, "state"))


@pytest.mark.parametrize("previous_key", [True, -1, 1 << 64])
def test_incremental_key_rejects_invalid_previous_key(previous_key: int) -> None:
    state = State.initial()
    with pytest.raises((TypeError, ValueError), match="previous_key"):
        incremental_state_key(previous_key, state, state)
