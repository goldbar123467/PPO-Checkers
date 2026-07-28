"""Board geometry tests for WCDF R1.1-R1.3 and the frozen ACF orientation."""

from __future__ import annotations

from typing import cast

import pytest

from checkers.rules.board import (
    acf_number,
    bit,
    coord,
    double_corner,
    is_playable_coord,
    rotate_square,
)
from checkers.rules.state import PlayerId

EXPECTED_ACF_ROWS = (
    (4, 3, 2, 1),
    (8, 7, 6, 5),
    (12, 11, 10, 9),
    (16, 15, 14, 13),
    (20, 19, 18, 17),
    (24, 23, 22, 21),
    (28, 27, 26, 25),
    (32, 31, 30, 29),
)
EXPECTED_PLAYABLE_SQUARES = 32


def test_r1_1_board_has_exactly_32_playable_dark_squares() -> None:
    seen: set[tuple[int, int]] = set()
    for square in range(32):
        row, column = coord(square)
        assert is_playable_coord(row, column)
        assert acf_number(row, column) == square + 1
        seen.add((row, column))
        assert bit(square) == 1 << square
    assert len(seen) == EXPECTED_PLAYABLE_SQUARES


def test_r1_1_light_and_out_of_bounds_coordinates_are_rejected() -> None:
    for row in range(8):
        for column in range(8):
            if (row, column) not in {coord(square) for square in range(32)}:
                assert not is_playable_coord(row, column)
                with pytest.raises(ValueError, match="playable"):
                    acf_number(row, column)
    for row, column in ((-1, 0), (0, -1), (8, 0), (0, 8)):
        assert not is_playable_coord(row, column)


def test_r1_2_acf_mapping_matches_frozen_unmirrored_diagram() -> None:
    for row, acf_row in enumerate(EXPECTED_ACF_ROWS):
        playable_columns = [column for column in range(8) if is_playable_coord(row, column)]
        assert [acf_number(row, column) for column in playable_columns] == list(acf_row)


def test_r1_2_rotation_maps_each_square_to_33_minus_its_acf_number() -> None:
    for square in range(32):
        rotated = rotate_square(square)
        assert rotated == 31 - square
        assert rotate_square(rotated) == square


def test_r1_3_double_corner_is_on_each_players_right() -> None:
    assert tuple(square + 1 for square in double_corner(PlayerId.RED)) == (1, 5)
    assert tuple(square + 1 for square in double_corner(PlayerId.WHITE)) == (28, 32)


@pytest.mark.parametrize("square", [-1, 32])
def test_board_square_indices_are_range_checked(square: int) -> None:
    with pytest.raises(ValueError, match="square"):
        coord(square)
    with pytest.raises(ValueError, match="square"):
        bit(square)
    with pytest.raises(ValueError, match="square"):
        rotate_square(square)


@pytest.mark.parametrize("square", [True, 1.5])
def test_board_square_indices_reject_non_integer_values(square: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        coord(cast(int, square))


def test_double_corner_requires_explicit_player_identity() -> None:
    with pytest.raises(TypeError, match="PlayerId"):
        double_corner(cast(PlayerId, 0))
