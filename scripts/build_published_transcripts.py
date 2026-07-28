#!/usr/bin/env python3
"""Build the pinned 20-game legality fixture from a published PDN archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zipfile import ZipFile

EXPECTED_ARCHIVE_SHA256 = "d1c2eb648e46827cf7eb2441f4ab22964329aa013896a0681932d837bd6de662"
EXPECTED_MEMBER = "Tricks traps and shots.pdn"
GAME_COUNT = 20
TAG_PATTERN = re.compile(r'\[([A-Za-z0-9_]+)\s+"([^"]*)"\]')
MOVE_PATTERN = re.compile(
    r"(?<![0-9])(?:[1-9]|[12][0-9]|3[0-2])"
    r"(?:[-x](?:[1-9]|[12][0-9]|3[0-2]))+(?![0-9])"
)
RESULT_PATTERN = re.compile(r"(?<!\S)(?:1/2-1/2|1-0|0-1|\*)(?!\S)")
GAME_NUMBER_PATTERN = re.compile(r"^TTS Game ([1-9]|1[0-9]|20)(?:\s|$)")


@dataclass(frozen=True, slots=True)
class _Args:
    archive: Path
    output: Path
    check: bool


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    namespace = parser.parse_args()
    return _Args(
        archive=cast(Path, namespace.archive),
        output=cast(Path, namespace.output),
        check=cast(bool, namespace.check),
    )


def _strip_nested(text: str, opening: str, closing: str) -> str:
    output: list[str] = []
    depth = 0
    for character in text:
        if character == opening:
            depth += 1
        elif character == closing and depth:
            depth -= 1
        elif depth == 0:
            output.append(character)
    if depth:
        raise ValueError(f"unclosed {opening}{closing} annotation")
    return "".join(output)


def _canonical_record(game: dict[str, object]) -> bytes:
    selected = {key: game[key] for key in ("event", "moves", "result")}
    return json.dumps(
        selected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _extract_games(pdn_text: str) -> list[dict[str, object]]:
    games: list[dict[str, object]] = []
    for chunk in re.split(r'(?=\[Event\s+")', pdn_text):
        if not chunk.strip():
            continue
        tags = dict(TAG_PATTERN.findall(chunk))
        event = tags.get("Event", "")
        game_number_match = GAME_NUMBER_PATTERN.match(event)
        if game_number_match is None or tags.get("Setup") == "1" or "FEN" in tags:
            continue
        body = TAG_PATTERN.sub(" ", chunk)
        body = _strip_nested(body, "{", "}")
        body = _strip_nested(body, "(", ")")
        body = RESULT_PATTERN.sub(" ", body)
        game: dict[str, object] = {
            "source_index": int(game_number_match.group(1)),
            "event": event,
            "result": tags["Result"],
            "moves": MOVE_PATTERN.findall(body),
        }
        game["source_record_sha256"] = hashlib.sha256(_canonical_record(game)).hexdigest()
        games.append(game)
    games.sort(key=lambda game: cast(int, game["source_index"]))
    selected = games[:GAME_COUNT]
    if [game["source_index"] for game in selected] != list(range(1, GAME_COUNT + 1)):
        raise ValueError("published archive does not contain the expected first 20 full games")
    return selected


def _build_payload(archive: Path) -> bytes:
    archive_bytes = archive.read_bytes()
    actual_hash = hashlib.sha256(archive_bytes).hexdigest()
    if actual_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {actual_hash}")
    with ZipFile(archive) as bundle:
        if bundle.namelist() != [EXPECTED_MEMBER]:
            raise ValueError(f"unexpected archive members: {bundle.namelist()}")
        pdn_text = bundle.read(EXPECTED_MEMBER).decode("utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": {
            "page_url": "https://www.bobnewell.net/checkers/pdn/pdndownloads.html",
            "archive_url": "https://www.bobnewell.net/checkers/pdn/tts.zip",
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_bytes": len(archive_bytes),
            "archive_member": EXPECTED_MEMBER,
            "retrieved_date": "2026-07-27 America/New_York",
            "license": (
                "Not stated on the source page; fixture contains only 20 factual move records "
                "and is excluded from training data."
            ),
            "selection": "First 20 non-setup entries named TTS Game 1 through TTS Game 20.",
        },
        "games": _extract_games(pdn_text),
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def main() -> int:
    """Generate or verify the deterministic published-game fixture.

    Returns:
        Zero after writing a new fixture or confirming an exact existing fixture.

    Raises:
        FileExistsError: If generation would overwrite an existing fixture.
        ValueError: If archive integrity, structure, selection, or check output differs.
    """

    args = _parse_args()
    generated = _build_payload(args.archive)
    if args.check:
        if args.output.read_bytes() != generated:
            raise ValueError("published transcript fixture differs from pinned archive extraction")
        print(f"fixture_check=PASS sha256={hashlib.sha256(generated).hexdigest()}")
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated)
    print(f"fixture_write=PASS sha256={hashlib.sha256(generated).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
