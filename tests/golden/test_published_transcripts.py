"""End-to-end replay of 20 externally published American Checkers games."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pytest

from checkers.rules.moves import Step, apply_step, legal_steps
from checkers.rules.notation import MovePath, parse_move, serialize_state
from checkers.rules.state import PlayerId, State

EXPECTED_ARCHIVE_SHA256 = "d1c2eb648e46827cf7eb2441f4ab22964329aa013896a0681932d837bd6de662"
EXPECTED_GAME_COUNT = 20
COMPRESSED_CAPTURE_PATH_LENGTH = 2
FIXTURE_PATH = Path(__file__).parent / "data" / "published_games.json"


class PublishedGame(TypedDict):
    source_index: int
    event: str
    result: str
    moves: list[str]
    source_record_sha256: str


class SourceMetadata(TypedDict):
    page_url: str
    archive_url: str
    archive_sha256: str
    archive_bytes: int
    archive_member: str
    retrieved_date: str
    license: str
    selection: str


class PublishedFixture(TypedDict):
    schema_version: int
    source: SourceMetadata
    games: list[PublishedGame]


@dataclass(frozen=True, slots=True)
class _CompletedMove:
    path: tuple[int, ...]
    steps: tuple[Step, ...]
    after: State


def _load_fixture() -> PublishedFixture:
    return cast(PublishedFixture, json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


PUBLISHED_FIXTURE = _load_fixture()
PUBLISHED_GAMES = PUBLISHED_FIXTURE["games"]


def _complete_moves(state: State) -> tuple[_CompletedMove, ...]:
    completed: list[_CompletedMove] = []

    def continue_capture(
        current: State,
        path: tuple[int, ...],
        steps: tuple[Step, ...],
    ) -> None:
        for step in legal_steps(current):
            transition = apply_step(current, step)
            next_path = (*path, step.destination)
            next_steps = (*steps, step)
            if transition.move_completed:
                completed.append(_CompletedMove(next_path, next_steps, transition.after))
            else:
                continue_capture(transition.after, next_path, next_steps)

    for step in legal_steps(state):
        transition = apply_step(state, step)
        path = (step.origin, step.destination)
        steps = (step,)
        if transition.move_completed:
            completed.append(_CompletedMove(path, steps, transition.after))
        else:
            continue_capture(transition.after, path, steps)
    return tuple(completed)


def _matches_recorded_path(candidate: _CompletedMove, recorded: MovePath) -> bool:
    candidate_is_capture = candidate.steps[0].is_capture
    if candidate_is_capture != recorded.is_capture:
        return False
    if len(recorded.squares) > COMPRESSED_CAPTURE_PATH_LENGTH:
        return candidate.path == recorded.squares
    return candidate.path[0] == recorded.squares[0] and candidate.path[-1] == recorded.squares[-1]


def _replay(game: PublishedGame) -> State:
    state = State.initial()
    for move_index, text in enumerate(game["moves"]):
        recorded = parse_move(text)
        candidates = [
            candidate
            for candidate in _complete_moves(state)
            if _matches_recorded_path(candidate, recorded)
        ]
        assert len(candidates) == 1, (
            f"{game['event']} move {move_index + 1} {text}: "
            f"expected one legal interpretation, found {len(candidates)}; "
            f"state={serialize_state(state)}"
        )
        state = candidates[0].after
    return state


def test_published_fixture_has_hash_pinned_provenance_and_exact_selection() -> None:
    source = PUBLISHED_FIXTURE["source"]
    assert PUBLISHED_FIXTURE["schema_version"] == 1
    assert source["archive_sha256"] == EXPECTED_ARCHIVE_SHA256
    assert source["archive_url"].startswith("https://www.bobnewell.net/checkers/pdn/")
    assert len(PUBLISHED_GAMES) == EXPECTED_GAME_COUNT
    assert len({game["event"] for game in PUBLISHED_GAMES}) == EXPECTED_GAME_COUNT


@pytest.mark.parametrize("game", PUBLISHED_GAMES, ids=[game["event"] for game in PUBLISHED_GAMES])
def test_published_transcript_replays_every_move_and_preserves_result_tag(
    game: PublishedGame,
) -> None:
    canonical_record = json.dumps(
        {"event": game["event"], "moves": game["moves"], "result": game["result"]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert hashlib.sha256(canonical_record).hexdigest() == game["source_record_sha256"]
    assert game["result"] in {"1-0", "0-1", "1/2-1/2"}

    final_state = _replay(game)

    if not legal_steps(final_state):
        mechanically_derived = "0-1" if final_state.side_to_move is PlayerId.RED else "1-0"
        assert game["result"] == mechanically_derived
