"""End-to-end proof that the five mandatory rule mutations are killed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_five_mandatory_rule_mutations_are_killed(tmp_path: Path) -> None:
    report_path = tmp_path / "mutation-challenges.json"
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_rule_mutation_challenges.py"),
            "--output",
            str(report_path),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["baseline"]["returncode"] == 0
    assert report["all_killed"] is True
    assert {result["name"] for result in report["challenges"]} == {
        "allow_backward_man_jumps",
        "make_captures_optional",
        "remove_captured_pieces_immediately",
        "permit_continuation_after_promotion",
        "add_majority_capture_rule",
    }
    assert all(result["returncode"] == 1 for result in report["challenges"])
