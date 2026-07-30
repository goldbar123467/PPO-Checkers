"""Exhaustive oracle evidence and the deduplicated three-move evaluation ballots."""

from __future__ import annotations

from pathlib import Path

from checkers.eval.ballots import (
    BALLOT_COUNT,
    BALLOT_MOVES,
    SEQUENCE_COUNT,
    load_ballot_set,
    load_sequence_manifest,
    replay_opening,
)
from checkers.rules.moves import legal_steps
from checkers.rules.oracle import oracle_legal_steps
from checkers.rules.zobrist import position_key

BALLOT_PATH = Path("data/ballots_v1.json")
SEQUENCE_PATH = Path("data/ballot_sequences_v1.json")
EXPECTED_SEQUENCE_COUNT = 302
EXPECTED_BALLOT_COUNT = 216
EXPECTED_COMPLETED_MOVES = 3
EXPECTED_FIRST_MOVE_COUNT = 7
EXPECTED_EXAMPLE_COUNT = 2


def test_sequence_evidence_is_exhaustive_unique_and_passes_first_move_gate() -> None:
    manifest = load_sequence_manifest(SEQUENCE_PATH)

    assert manifest.count == SEQUENCE_COUNT == EXPECTED_SEQUENCE_COUNT
    assert manifest.completed_moves == BALLOT_MOVES == EXPECTED_COMPLETED_MOVES
    assert manifest.distinct_first_moves == EXPECTED_FIRST_MOVE_COUNT
    assert manifest.distinct_positions == BALLOT_COUNT == EXPECTED_BALLOT_COUNT
    assert len({sequence.sequence_hash for sequence in manifest.sequences}) == SEQUENCE_COUNT
    assert len({sequence.actions for sequence in manifest.sequences}) == SEQUENCE_COUNT
    assert len(manifest.transposition_examples) == EXPECTED_EXAMPLE_COUNT


def test_eval_ballots_are_position_key_deduplicated_legal_and_exactly_replayable() -> None:
    ballot_set = load_ballot_set(BALLOT_PATH)

    assert ballot_set.count == BALLOT_COUNT == EXPECTED_BALLOT_COUNT
    assert ballot_set.completed_moves == BALLOT_MOVES == EXPECTED_COMPLETED_MOVES
    assert ballot_set.source_sequence_count == SEQUENCE_COUNT == EXPECTED_SEQUENCE_COUNT
    assert ballot_set.distinct_first_moves == EXPECTED_FIRST_MOVE_COUNT
    assert ballot_set.deduplicate_on == "position_key"
    assert len({ballot.position_key for ballot in ballot_set.ballots}) == BALLOT_COUNT

    for ballot in ballot_set.ballots:
        first = replay_opening(ballot.actions)
        second = replay_opening(ballot.actions)
        assert first == second
        assert first.state == ballot.state
        assert first.completed_moves == BALLOT_MOVES
        assert not first.state.capture_in_progress
        assert position_key(first.state) == ballot.position_key
        assert legal_steps(first.state) == oracle_legal_steps(first.state)
        assert legal_steps(first.state)
