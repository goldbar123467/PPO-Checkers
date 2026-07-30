"""Deterministic tactical-fixture generator and source-hash regression checks."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from checkers.eval.suites import load_dev_tactical_suite, replay_tactical_case
from scripts.generate_dev_tactics import GenerationConfig, generate_cases

EXPECTED_CASES_SHA256 = "cf0bf4040185dfb229099f9780f988b9650425833c36390f2427c181729ffd01"
EXPECTED_FILE_SHA256 = "12a343d3ae9d186c0ad91bc0cf852b38d7d83725bc81be029b827c3bb0f1899f"
TACTICAL_PATH = Path("src/checkers/eval/data/dev_tactics_v1.json")
GENERATOR_PATH = Path("scripts/generate_dev_tactics.py")
CONTRACT_PATH = Path("docs/experiment-contract.md")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rules_sha256() -> str:
    paths = tuple(Path("src/checkers/rules").glob("*.py")) + (Path("src/checkers/env/masking.py"),)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_one_game_scan_reproduces_first_reachable_exact_case() -> None:
    result = generate_cases(
        GenerationConfig(
            max_games=1,
            target_cases=1,
            minimum_depth1_misses=0,
        )
    )

    assert len(result.cases) == 1
    assert result.games_scanned == 1
    assert result.cases[0].case_id == "dev-tactic-001"
    assert replay_tactical_case(result.cases[0]) == result.cases[0].state


def test_packaged_tactical_data_and_case_digests_are_pinned() -> None:
    suite = load_dev_tactical_suite()

    assert suite.manifest.cases_sha256 == EXPECTED_CASES_SHA256
    assert _file_sha256(TACTICAL_PATH) == EXPECTED_FILE_SHA256


def test_manifest_source_hashes_match_current_generator_rules_and_contract() -> None:
    manifest = load_dev_tactical_suite().manifest

    assert manifest.generator_source_sha256 == _file_sha256(GENERATOR_PATH)
    assert manifest.rules_source_sha256 == _rules_sha256()
    assert manifest.goal_sha256 == _file_sha256(CONTRACT_PATH)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: GenerationConfig(seed=-1), "seed"),
        (lambda: GenerationConfig(seed=cast(int, True)), "seed"),
        (lambda: GenerationConfig(horizon=0), "positive"),
        (lambda: GenerationConfig(max_games=0), "positive"),
        (lambda: GenerationConfig(target_cases=0), "positive"),
        (lambda: GenerationConfig(target_cases=1, minimum_depth1_misses=2), "within"),
    ],
)
def test_generation_config_rejects_invalid_budgets(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_bounded_scan_reports_unsatisfied_criteria() -> None:
    with pytest.raises(RuntimeError, match="exhausted"):
        generate_cases(
            GenerationConfig(
                max_games=1,
                target_cases=2,
                minimum_depth1_misses=0,
            )
        )
