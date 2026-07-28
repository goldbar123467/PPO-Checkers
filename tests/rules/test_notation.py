"""ACF move notation and full-state serialization tests for R7."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from checkers.rules.notation import (
    MovePath,
    format_move,
    parse_move,
    parse_state,
    serialize_state,
)
from checkers.rules.state import PlayerId, State


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


class TestMoveExamples:
    @pytest.mark.parametrize(
        ("text", "squares", "is_capture"),
        [
            ("11-15", (10, 14), False),
            ("22x18", (21, 17), True),
            ("22x18x9", (21, 17, 8), True),
        ],
    )
    def test_r7_1_acf_simple_jump_and_multijump_examples(
        self,
        text: str,
        squares: tuple[int, ...],
        *,
        is_capture: bool,
    ) -> None:
        move = MovePath(squares=squares, is_capture=is_capture)
        assert format_move(move) == text
        assert parse_move(text) == move


def test_r7_2_move_notation_round_trip() -> None:
    moves = (
        MovePath((0, 4), False),
        MovePath((31, 22), True),
        MovePath((0, 9, 18, 25), True),
    )
    for move in moves:
        assert parse_move(format_move(move)) == move


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (lambda: MovePath(cast(tuple[int, ...], [0, 4]), False), TypeError, "tuple"),
        (lambda: MovePath((0, 4), cast(bool, 1)), TypeError, "bool"),
        (lambda: MovePath((), False), ValueError, "at least two"),
        (lambda: MovePath((0, 4, 9), False), ValueError, "exactly two"),
        (lambda: MovePath((cast(int, True), 4), False), TypeError, "integer"),
    ],
)
def test_r7_2_move_path_rejects_invalid_runtime_values(
    factory: Callable[[], MovePath],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        factory()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "11",
        "0-4",
        "1-33",
        "01-05",
        "11x15-18",
        "11-15-18",
        "11X15",
        " 11-15",
    ],
)
def test_r7_2_parser_rejects_noncanonical_or_mixed_notation(text: str) -> None:
    with pytest.raises(ValueError, match="notation"):
        parse_move(text)


def test_r7_2_public_move_functions_reject_wrong_object_types() -> None:
    with pytest.raises(TypeError, match="MovePath"):
        format_move(cast(MovePath, "11-15"))
    with pytest.raises(TypeError, match="text"):
        parse_move(cast(str, 11))


def test_r7_3_initial_state_has_frozen_canonical_serialization() -> None:
    expected = (
        "CHK1;RM=00000fff;RK=00000000;WM=fff00000;WK=00000000;STM=R;CIP=0;"
        "MS=-;SO=-;CP=00000000;NP=0,0;PLY=0"
    )
    assert serialize_state(State.initial()) == expected
    assert parse_state(expected) == State.initial()


def test_r7_3_full_midsequence_state_round_trips_exactly() -> None:
    state = State(
        men=(_mask(18), _mask(14, 22)),
        kings=(_mask(3), _mask(30)),
        side_to_move=PlayerId.RED,
        capture_in_progress=True,
        moving_square=17,
        sequence_origin=8,
        captured_pending=_mask(14),
        no_progress=(17, 39),
        ply=511,
    )

    encoded = serialize_state(state)

    assert "CIP=1;MS=18;SO=9;CP=00002000;NP=17,39;PLY=511" in encoded
    assert parse_state(encoded) == state
    assert serialize_state(parse_state(encoded)) == encoded


def test_r7_3_white_boundary_state_round_trips_exactly() -> None:
    state = State(men=(0, _mask(24)), kings=(0, 0), side_to_move=PlayerId.WHITE)
    encoded = serialize_state(state)
    assert "STM=W" in encoded
    assert parse_state(encoded) == state


@pytest.mark.parametrize(
    "text",
    [
        "",
        "CHK2;RM=00000000;RK=00000000;WM=00000000;WK=00000000;STM=R;CIP=0;MS=-;SO=-;CP=00000000;NP=0,0;PLY=0",
        "CHK1;RM=000000000;RK=00000000;WM=00000000;WK=00000000;STM=R;CIP=0;MS=-;SO=-;CP=00000000;NP=0,0;PLY=0",
        "CHK1;RM=00000000;RK=00000000;WM=00000000;WK=00000000;STM=B;CIP=0;MS=-;SO=-;CP=00000000;NP=0,0;PLY=0",
    ],
)
def test_r7_3_state_parser_rejects_noncanonical_schema(text: str) -> None:
    with pytest.raises(ValueError, match="state serialization"):
        parse_state(text)


def test_r7_3_state_parser_wraps_semantic_invariant_failures() -> None:
    invalid_capture_state = (
        "CHK1;RM=00000000;RK=00000000;WM=00000000;WK=00000000;STM=R;CIP=1;"
        "MS=-;SO=-;CP=00000000;NP=0,0;PLY=0"
    )
    with pytest.raises(ValueError, match="state serialization"):
        parse_state(invalid_capture_state)


def test_r7_3_public_state_functions_reject_wrong_object_types() -> None:
    with pytest.raises(TypeError, match="State"):
        serialize_state(cast(State, "state"))
    with pytest.raises(TypeError, match="text"):
        parse_state(cast(str, State.initial()))
