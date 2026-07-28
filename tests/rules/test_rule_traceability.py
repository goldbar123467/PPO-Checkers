"""Trace every GOAL.md rule to primary authority or an explicit project classification."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES_DOC = ROOT / "docs" / "RULES.md"
WCDF_URL = "https://wcdf.net/rules/rules_of_checkers_english.pdf"
WCDF_SHA256 = "aa1d1235632046c05db7621437f16c33bc7b86b472ccaa039a7a41b897b180b7"

RULE_IDS = {
    "R1.1",
    "R1.2",
    "R1.3",
    "R1.4",
    "R1.5",
    "R2.1",
    "R2.2",
    "R3.1",
    "R3.2",
    "R3.3",
    "R4.1",
    "R4.2",
    "R4.3.1",
    "R4.3.2",
    "R4.4",
    "R4.5",
    "R4.6",
    "R5.1",
    "R5.2",
    "R5.3",
    "R6.1",
    "R6.2",
    "R6.3",
    "R6.4",
    "R6.5",
    "R6.6",
    "R6.7",
    "R7.1",
    "R7.2",
    "R7.3",
}
ENGINE_VARIANTS = {"R6.3", "R6.4", "R6.5", "R6.6"}
EXPECTED_MAN_ADVANCES = 144
EXPECTED_RESETTING_MOVES = 167
EXPECTED_COMPLETED_MOVES = 13_440
EXPECTED_ENVIRONMENT_STEPS = 13_462
PLY_CAP = 512
ROW_PATTERN = re.compile(
    r"^\|\s*(R\d+\.\d+(?:\.\d+)?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$",
    re.MULTILINE,
)


def _traceability_rows(document: str) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for rule_id, authority, tests in ROW_PATTERN.findall(document):
        assert rule_id not in rows, f"duplicate traceability row: {rule_id}"
        rows[rule_id] = (authority.strip(), tests.strip())
    return rows


def test_phase_1_every_rule_id_has_exactly_one_authority_row() -> None:
    document = RULES_DOC.read_text(encoding="utf-8")
    rows = _traceability_rows(document)
    assert rows.keys() == RULE_IDS
    allowed_authority = ("WCDF", "ENGINE VARIANT", "PROJECT CONTRACT", "DERIVED")
    assert all(authority.startswith(allowed_authority) for authority, _tests in rows.values())
    assert all(tests != "TBD" for _authority, tests in rows.values())


def test_phase_1_all_rules_departures_are_explicit_engine_variants() -> None:
    document = RULES_DOC.read_text(encoding="utf-8")
    rows = _traceability_rows(document)
    actual_variants = {
        rule_id
        for rule_id, (authority, _tests) in rows.items()
        if authority.startswith("ENGINE VARIANT")
    }
    assert actual_variants == ENGINE_VARIANTS


def test_phase_1_primary_source_provenance_is_pinned() -> None:
    document = RULES_DOC.read_text(encoding="utf-8")
    assert WCDF_URL in document
    assert WCDF_SHA256 in document
    assert "177,885 bytes" in document
    assert "Last-Modified: Mon, 19 Aug 2013 22:58:02 GMT" in document


def test_r6_7_termination_bound_is_derived_and_above_ply_cap() -> None:
    men_per_starting_row = 4
    forward_row_distances = (7, 6, 5)
    colours = 2
    max_man_advances = colours * men_per_starting_row * sum(forward_row_distances)
    max_capture_removals = 23
    max_resetting_moves = max_man_advances + max_capture_removals
    moves_per_counter_window = 2 * 40
    max_completed_moves = (max_resetting_moves + 1) * moves_per_counter_window
    max_extra_jump_steps = max_capture_removals - 1
    max_environment_steps = max_completed_moves + max_extra_jump_steps

    assert max_man_advances == EXPECTED_MAN_ADVANCES
    assert max_resetting_moves == EXPECTED_RESETTING_MOVES
    assert max_completed_moves == EXPECTED_COMPLETED_MOVES
    assert max_environment_steps == EXPECTED_ENVIRONMENT_STEPS
    assert max_environment_steps > PLY_CAP

    document = RULES_DOC.read_text(encoding="utf-8")
    for value in (
        EXPECTED_MAN_ADVANCES,
        max_capture_removals,
        EXPECTED_RESETTING_MOVES,
        EXPECTED_COMPLETED_MOVES,
        EXPECTED_ENVIRONMENT_STEPS,
        PLY_CAP,
    ):
        assert f"{value:,}" in document


def test_r1_2_board_orientation_and_player_convention_are_frozen() -> None:
    document = RULES_DOC.read_text(encoding="utf-8")
    assert "FROZEN ACF 1–32 orientation" in document
    assert "PlayerId.RED" in document
    assert "Red moves toward increasing square numbers" in document
    for square in range(1, 33):
        assert re.search(rf"(?<!\d){square}(?!\d)", document)
