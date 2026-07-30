"""Frozen 64-bit Zobrist keys for complete states and official positions."""

from __future__ import annotations

from collections.abc import Iterator

from checkers.rules.state import PlayerId, State

UINT64_MASK = (1 << 64) - 1
ZOBRIST_SCHEMA_VERSION = 1
ZOBRIST_MASTER_SEED = 0xC4E3_2026_0728_0001
SPLITMIX_INCREMENT = 0x9E37_79B9_7F4A_7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58_476D_1CE4_E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D0_49BB_1331_11EB

PIECE_FEATURE_BASE = 0
SIDE_TO_MOVE_FEATURE = 128
CAPTURE_IN_PROGRESS_FEATURE = 129
MOVING_SQUARE_FEATURE_BASE = 130
SEQUENCE_ORIGIN_FEATURE_BASE = 162
CAPTURED_PENDING_FEATURE_BASE = 194
NO_PROGRESS_RED_FEATURE = 226
NO_PROGRESS_WHITE_FEATURE = 227
PLY_FEATURE = 228


def _splitmix64(value: int) -> int:
    mixed = (value + SPLITMIX_INCREMENT) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * SPLITMIX_MULTIPLIER_1) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * SPLITMIX_MULTIPLIER_2) & UINT64_MASK
    return (mixed ^ (mixed >> 31)) & UINT64_MASK


def _feature_seed(feature: int) -> int:
    return _splitmix64(ZOBRIST_MASTER_SEED ^ (feature * SPLITMIX_INCREMENT))


def _scalar_seed(feature: int, value: int) -> int:
    return _splitmix64(_feature_seed(feature) ^ value)


def _iter_squares(mask: int) -> Iterator[int]:
    remaining = mask
    while remaining:
        least_significant = remaining & -remaining
        yield least_significant.bit_length() - 1
        remaining ^= least_significant


def _piece_feature(board_index: int, square: int) -> int:
    return PIECE_FEATURE_BASE + board_index * 32 + square


def _placement_key(state: State) -> int:
    key = 0
    boards = (state.men[0], state.men[1], state.kings[0], state.kings[1])
    for board_index, board in enumerate(boards):
        for square in _iter_squares(board):
            key ^= _feature_seed(_piece_feature(board_index, square))
    if state.side_to_move is PlayerId.WHITE:
        key ^= _feature_seed(SIDE_TO_MOVE_FEATURE)
    return key


def _validate_state(state: State) -> None:
    if not isinstance(state, State):
        raise TypeError("state must be a State")


def state_key(state: State) -> int:
    """Recompute the frozen 64-bit key for the complete transition state.

    The key includes `sequence_origin`, both terminal counters, and `ply`. Those fields alter
    future complete states or terminal transitions and therefore belong to a Markov-state cache
    key.

    Args:
        state: Complete immutable state.

    Returns:
        Unsigned 64-bit deterministic Zobrist key.

    Raises:
        TypeError: If state is not a State.
    """

    _validate_state(state)
    key = _placement_key(state)
    if state.capture_in_progress:
        key ^= _feature_seed(CAPTURE_IN_PROGRESS_FEATURE)
    if state.moving_square is not None:
        key ^= _feature_seed(MOVING_SQUARE_FEATURE_BASE + state.moving_square)
    if state.sequence_origin is not None:
        key ^= _feature_seed(SEQUENCE_ORIGIN_FEATURE_BASE + state.sequence_origin)
    for square in _iter_squares(state.captured_pending):
        key ^= _feature_seed(CAPTURED_PENDING_FEATURE_BASE + square)
    key ^= _scalar_seed(NO_PROGRESS_RED_FEATURE, state.no_progress[0])
    key ^= _scalar_seed(NO_PROGRESS_WHITE_FEATURE, state.no_progress[1])
    key ^= _scalar_seed(PLY_FEATURE, state.ply)
    return key & UINT64_MASK


def position_key(state: State) -> int:
    """Return placement-plus-side key for official boundary positions.

    Args:
        state: Complete immutable state at a completed-move boundary.

    Returns:
        Unsigned 64-bit deterministic position key.

    Raises:
        TypeError: If state is not a State.
        ValueError: If a capture sequence is in progress.
    """

    _validate_state(state)
    if state.capture_in_progress:
        raise ValueError("position_key is defined only at a completed-move boundary")
    return _placement_key(state) & UINT64_MASK


def _xor_changed_mask(key: int, before: int, after: int, feature_base: int) -> int:
    for square in _iter_squares(before ^ after):
        key ^= _feature_seed(feature_base + square)
    return key


def _xor_changed_optional(
    key: int,
    before: int | None,
    after: int | None,
    feature_base: int,
) -> int:
    if before == after:
        return key
    if before is not None:
        key ^= _feature_seed(feature_base + before)
    if after is not None:
        key ^= _feature_seed(feature_base + after)
    return key


def incremental_state_key(previous_key: int, before: State, after: State) -> int:
    """Update a complete-state key by XORing only fields that changed.

    Args:
        previous_key: `state_key(before)` or a key produced by this function.
        before: State represented by previous_key.
        after: New state after one transition or undo.

    Returns:
        Incrementally updated unsigned 64-bit key.

    Raises:
        TypeError: If keys or states have the wrong runtime types.
        ValueError: If previous_key is outside the uint64 range.
    """

    if isinstance(previous_key, bool) or not isinstance(previous_key, int):
        raise TypeError("previous_key must be an integer")
    if not 0 <= previous_key <= UINT64_MASK:
        raise ValueError("previous_key must fit uint64")
    _validate_state(before)
    _validate_state(after)

    key = previous_key
    before_boards = (before.men[0], before.men[1], before.kings[0], before.kings[1])
    after_boards = (after.men[0], after.men[1], after.kings[0], after.kings[1])
    for board_index, (before_board, after_board) in enumerate(
        zip(before_boards, after_boards, strict=True)
    ):
        key = _xor_changed_mask(
            key,
            before_board,
            after_board,
            PIECE_FEATURE_BASE + board_index * 32,
        )

    if before.side_to_move is not after.side_to_move:
        key ^= _feature_seed(SIDE_TO_MOVE_FEATURE)
    if before.capture_in_progress != after.capture_in_progress:
        key ^= _feature_seed(CAPTURE_IN_PROGRESS_FEATURE)
    key = _xor_changed_optional(
        key,
        before.moving_square,
        after.moving_square,
        MOVING_SQUARE_FEATURE_BASE,
    )
    key = _xor_changed_optional(
        key,
        before.sequence_origin,
        after.sequence_origin,
        SEQUENCE_ORIGIN_FEATURE_BASE,
    )
    key = _xor_changed_mask(
        key,
        before.captured_pending,
        after.captured_pending,
        CAPTURED_PENDING_FEATURE_BASE,
    )

    scalar_fields = (
        (NO_PROGRESS_RED_FEATURE, before.no_progress[0], after.no_progress[0]),
        (NO_PROGRESS_WHITE_FEATURE, before.no_progress[1], after.no_progress[1]),
        (PLY_FEATURE, before.ply, after.ply),
    )
    for feature, before_value, after_value in scalar_fields:
        if before_value != after_value:
            key ^= _scalar_seed(feature, before_value)
            key ^= _scalar_seed(feature, after_value)
    return key & UINT64_MASK
