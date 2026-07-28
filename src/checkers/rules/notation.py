"""Strict ACF move notation and versioned complete-state serialization."""

from __future__ import annotations

import re
from dataclasses import dataclass

from checkers.rules.board import bit
from checkers.rules.state import PlayerId, State

ACF_SQUARE_PATTERN = r"(?:[1-9]|[12][0-9]|3[0-2])"
NONNEGATIVE_PATTERN = r"(?:0|[1-9][0-9]*)"
MINIMUM_PATH_LENGTH = 2
MOVE_PATTERN = re.compile(
    rf"(?:{ACF_SQUARE_PATTERN}-{ACF_SQUARE_PATTERN}|"
    rf"{ACF_SQUARE_PATTERN}(?:x{ACF_SQUARE_PATTERN})+)"
)
STATE_PATTERN = re.compile(
    rf"CHK1;RM=(?P<red_men>[0-9a-f]{{8}});RK=(?P<red_kings>[0-9a-f]{{8}});"
    rf"WM=(?P<white_men>[0-9a-f]{{8}});WK=(?P<white_kings>[0-9a-f]{{8}});"
    rf"STM=(?P<side>[RW]);CIP=(?P<capture>[01]);"
    rf"MS=(?P<moving>-|{ACF_SQUARE_PATTERN});SO=(?P<origin>-|{ACF_SQUARE_PATTERN});"
    rf"CP=(?P<pending>[0-9a-f]{{8}});"
    rf"NP=(?P<red_counter>{NONNEGATIVE_PATTERN}),(?P<white_counter>{NONNEGATIVE_PATTERN});"
    rf"PLY=(?P<ply>{NONNEGATIVE_PATTERN})"
)


@dataclass(frozen=True, slots=True, init=False)
class MovePath:
    """One complete checkers move in internal zero-based ACF squares.

    Args:
        squares: Ordered origin, landing, and optional continuation landing squares.
        is_capture: True for an `x`-separated capture path; false for a simple move.

    Raises:
        TypeError: If fields do not have their exact declared runtime types.
        ValueError: If the path length or any square is invalid.
    """

    squares: tuple[int, ...]
    is_capture: bool

    def __init__(self, squares: tuple[int, ...], is_capture: bool) -> None:
        object.__setattr__(self, "squares", squares)
        object.__setattr__(self, "is_capture", is_capture)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.squares, tuple):
            raise TypeError("move squares must be a tuple")
        if not isinstance(self.is_capture, bool):
            raise TypeError("is_capture must be bool")
        if len(self.squares) < MINIMUM_PATH_LENGTH:
            raise ValueError("move path needs at least two squares")
        if not self.is_capture and len(self.squares) != MINIMUM_PATH_LENGTH:
            raise ValueError("simple move path must contain exactly two squares")
        for square in self.squares:
            bit(square)


def format_move(move: MovePath) -> str:
    """Format one complete move in canonical ACF notation.

    Args:
        move: Valid internal move path.

    Returns:
        Canonical `11-15`, `22x18`, or `22x18x9` style text.

    Raises:
        TypeError: If `move` is not a `MovePath`.
    """

    if not isinstance(move, MovePath):
        raise TypeError("move must be a MovePath")
    separator = "x" if move.is_capture else "-"
    return separator.join(str(square + 1) for square in move.squares)


def parse_move(text: str) -> MovePath:
    """Parse strict canonical ACF move notation.

    Args:
        text: ACF text using one consistent `-` or `x` separator.

    Returns:
        Parsed zero-based move path.

    Raises:
        TypeError: If `text` is not a string.
        ValueError: If `text` is not canonical ACF notation.
    """

    if not isinstance(text, str):
        raise TypeError("move notation must be text")
    if MOVE_PATTERN.fullmatch(text) is None:
        raise ValueError("invalid ACF move notation")
    is_capture = "x" in text
    separator = "x" if is_capture else "-"
    squares = tuple(int(token) - 1 for token in text.split(separator))
    return MovePath(squares=squares, is_capture=is_capture)


def _format_optional_square(square: int | None) -> str:
    return "-" if square is None else str(square + 1)


def serialize_state(state: State) -> str:
    """Serialize every field of a state into canonical versioned text.

    Args:
        state: Complete state, including any active capture sequence and counters.

    Returns:
        A deterministic `CHK1` full-state string with lowercase uint32 hex fields.

    Raises:
        TypeError: If `state` is not a `State`.
    """

    if not isinstance(state, State):
        raise TypeError("state must be a State")
    side = "R" if state.side_to_move is PlayerId.RED else "W"
    capture = int(state.capture_in_progress)
    return (
        f"CHK1;RM={state.men[0]:08x};RK={state.kings[0]:08x};"
        f"WM={state.men[1]:08x};WK={state.kings[1]:08x};"
        f"STM={side};CIP={capture};MS={_format_optional_square(state.moving_square)};"
        f"SO={_format_optional_square(state.sequence_origin)};CP={state.captured_pending:08x};"
        f"NP={state.no_progress[0]},{state.no_progress[1]};PLY={state.ply}"
    )


def _parse_optional_square(text: str) -> int | None:
    return None if text == "-" else int(text) - 1


def parse_state(text: str) -> State:
    """Parse a strict `CHK1` serialization into a validated complete state.

    Args:
        text: Canonical string produced by `serialize_state`.

    Returns:
        A validated immutable state exactly representing every serialized field.

    Raises:
        TypeError: If `text` is not a string.
        ValueError: If syntax is noncanonical or the decoded state violates invariants.
    """

    if not isinstance(text, str):
        raise TypeError("state serialization must be text")
    match = STATE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("invalid state serialization syntax")
    groups = match.groupdict()
    side = PlayerId.RED if groups["side"] == "R" else PlayerId.WHITE
    try:
        return State(
            men=(int(groups["red_men"], 16), int(groups["white_men"], 16)),
            kings=(int(groups["red_kings"], 16), int(groups["white_kings"], 16)),
            side_to_move=side,
            capture_in_progress=groups["capture"] == "1",
            moving_square=_parse_optional_square(groups["moving"]),
            sequence_origin=_parse_optional_square(groups["origin"]),
            captured_pending=int(groups["pending"], 16),
            no_progress=(int(groups["red_counter"]), int(groups["white_counter"])),
            ply=int(groups["ply"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid state serialization: {error}") from error
