"""Externally published completed-move perft checks for the fast rules engine."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from checkers.rules.moves import apply_step, legal_steps
from checkers.rules.state import State

FIXTURE_PATH = Path(__file__).parent / "data" / "external_perft.json"


@cache
def _completed_move_perft(state: State, depth: int) -> int:
    if depth == 0:
        return 1
    leaves = 0
    for step in legal_steps(state):
        transition = apply_step(state, step)
        next_depth = depth - int(transition.move_completed)
        leaves += _completed_move_perft(transition.after, next_depth)
    return leaves


def test_external_bik_perft_matches_through_completed_move_depth_seven() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    gate_depth = fixture["gate_depth"]
    expected = {int(depth): count for depth, count in fixture["leaf_nodes"].items()}

    assert fixture["classification"] == "EXTERNAL CORRECTNESS EVIDENCE"
    assert fixture["source"]["doi"] == "10.3233/ICG-2012-35403"
    assert {
        depth: _completed_move_perft(State.initial(), depth) for depth in range(gate_depth + 1)
    } == {depth: expected[depth] for depth in range(gate_depth + 1)}
