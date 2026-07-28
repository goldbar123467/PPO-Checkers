"""Phase 0 scaffold contract derived from GOAL.md §§0.3, 12.9-12.11, and 14."""

from __future__ import annotations

import hashlib
import importlib
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOAL_SHA256 = "dab54331c088a201c1e43e0743866e1780aa84e3b0868b0b7cce34271c17660f"


def test_phase_0_goal_is_canonical_and_unchanged() -> None:
    goal = ROOT / "GOAL.md"
    assert goal.is_file()
    assert hashlib.sha256(goal.read_bytes()).hexdigest() == GOAL_SHA256


def test_phase_0_checkers_package_imports() -> None:
    package = importlib.import_module("checkers")
    assert package.__version__ == "0.1.0"


def test_phase_0_control_files_and_directories_exist() -> None:
    required_files = {
        ".github/workflows/offline-ci.yml",
        ".pre-commit-config.yaml",
        "BLOCKERS.md",
        "DECISIONS.md",
        "PROGRESS.md",
        "STATE.json",
        "logs/SUMMARY.md",
    }
    required_directories = {
        "docs",
        "logs/gates",
        "logs/iterations",
        "logs/test-output",
        "src/checkers/agents",
        "src/checkers/env",
        "src/checkers/eval",
        "src/checkers/rl",
        "src/checkers/rules",
        "tests/env",
        "tests/eval",
        "tests/golden",
        "tests/integration",
        "tests/metamorphic",
        "tests/property",
        "tests/rl",
        "tests/rules",
    }
    assert not {path for path in required_files if not (ROOT / path).is_file()}
    assert not {path for path in required_directories if not (ROOT / path).is_dir()}


def test_phase_0_goal_make_targets_are_wired() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        "check",
        "eval",
        "format",
        "fuzz",
        "fuzz-ci",
        "lint",
        "mutate",
        "perft",
        "smoke",
        "test",
        "train",
        "types",
    }
    missing = {
        target for target in targets if not re.search(rf"(?m)^{re.escape(target)}:", makefile)
    }
    assert not missing


def test_phase_0_quality_tools_are_exactly_pinned_and_strict() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = config["dependency-groups"]["dev"]
    required_tools = {"hypothesis", "mypy", "mutmut", "pre-commit", "pytest", "pytest-cov", "ruff"}
    pins = {dependency.split("==", maxsplit=1)[0]: dependency for dependency in dev_dependencies}
    assert required_tools <= pins.keys()
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^ ;]+", pin) for pin in pins.values())
    all_direct_dependencies = [
        *config["build-system"]["requires"],
        *config["project"]["dependencies"],
        *dev_dependencies,
        *(
            dependency
            for group in config["project"]["optional-dependencies"].values()
            for dependency in group
        ),
    ]
    exact_pin = re.compile(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^ ;]+")
    assert all(exact_pin.fullmatch(dependency) for dependency in all_direct_dependencies)
    assert config["tool"]["mypy"]["strict"] is True
    expected_lint = {
        "ANN",
        "ARG",
        "B",
        "E",
        "ERA",
        "F",
        "I",
        "N",
        "PL",
        "PTH",
        "RET",
        "SIM",
        "UP",
        "W",
    }
    assert expected_lint <= set(config["tool"]["ruff"]["lint"]["select"])


def test_phase_0_offline_ci_blocks_egress_after_install() -> None:
    workflow = (ROOT / ".github/workflows/offline-ci.yml").read_text(encoding="utf-8")
    install_at = workflow.index("uv sync")
    block_at = workflow.index("iptables")
    check_at = workflow.index("make check")
    assert install_at < block_at < check_at
    assert "WANDB_MODE: offline" in workflow
