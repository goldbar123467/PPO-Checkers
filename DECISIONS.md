# Decisions

## ADR-001 — Integrate checkers without deleting the existing ML lab

- Status: Accepted
- Authority: Tier C project integration decision; constrained by the user request and `AGENTS.md`.
- Evidence stage: Setup validation.
- Context: `/home/thecl/ml-lab` already contains the `ml_lab` package, experiments, and scripts.
- Decision: Add `src/checkers`, checkers tests, and goal-mandated project files at the repository
  root. Preserve unrelated source and artifacts; make shared quality gates cover both packages.
- Consequence: The repository is a superset of the §14 reference tree rather than a destructive
  replacement. Pytest retains the legacy tests; the new strict lint, typing, and coverage gates are
  scoped to the checkers project so unrelated legacy typing debt is not misreported as checkers
  correctness. The pre-existing `make mypy` target remains available for `src/ml_lab`.

## ADR-002 — Canonical specification path and integrity

- Status: Accepted
- Authority: `GOAL.md` file contract and §14.
- Evidence stage: Setup validation.
- Decision: Keep the moved document at repository-root `GOAL.md`, byte-for-byte unchanged.
- Evidence: SHA-256 `dab54331c088a201c1e43e0743866e1780aa84e3b0868b0b7cce34271c17660f`.

## ADR-003 — Phase 0 quality-tool configuration

- Status: Accepted
- Authority: Tier B engineering practice; `GOAL.md` §§12.9–12.11 and official tool documentation.
- Evidence stage: Setup validation.
- Decision: Pin every direct dependency, use Ruff as the sole formatter/linter, enable the exact
  mandated Ruff families, set `mypy` strict mode, collect branch coverage through pytest-cov, and
  use local pre-commit hooks backed by the locked environment.
- Primary sources:
  - Ruff configuration: <https://docs.astral.sh/ruff/configuration/>
  - Ruff formatter: <https://docs.astral.sh/ruff/formatter/>
  - mypy configuration and strict mode: <https://mypy.readthedocs.io/en/stable/config_file.html>
  - pytest-cov configuration: <https://pytest-cov.readthedocs.io/en/stable/config.html>
  - pre-commit configuration: <https://pre-commit.com/>
  - GitHub workflow syntax: <https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax>
- Evidence: `logs/gates/phase-0.txt` and successful `pre-commit validate-config`.

## ADR-004 — Torch lock resolution changed the local CUDA wheel

- Status: Accepted after runtime validation
- Authority: Local hardware evidence; `AGENTS.md` hardware discipline.
- Evidence stage: Setup validation.
- Context: `uv sync --locked --all-groups` replaced `torch 2.13.0+cu132` with the locked
  `torch 2.13.0+cu130` distribution while adding Phase 0 tooling.
- Decision: Retain the locked CUDA 13.0 build because the hardware doctor proved CUDA execution,
  finite backward gradients, BF16, compiled `sm_120`, and bitsandbytes NF4 forward/backward on the
  RTX 5070. Revisit only if a later experiment produces a concrete incompatibility.
- Evidence: `logs/gates/phase-0-gpu-doctor.txt` (exit 0, `overall_passed: true`).

## ADR-005 — Source-correct delayed removal and R6.6 classification

- Status: Accepted provisionally; BLOCK-001 remains open for the read-only spec erratum.
- Authority: Tier A WCDF clauses 1.18–1.20 and 1.32.
- Evidence stage: Rules verification.
- Decision: Read WCDF 1.19, the specific multi-jump clause, as controlling 1.18's single-jump
  removal language. During a sequence, captured pieces remain occupying their squares and cannot
  be jumped twice. Separately, label R6.6 ENGINE VARIANT because WCDF 1.32 permits agreed draws.
- Rationale: This preserves the more specific capture-sequence text and never attributes a clear
  project departure to WCDF.
- Evidence: `docs/RULES.md`, `tests/rules/test_rule_traceability.py`, and BLOCK-001.

## ADR-006 — Correct ACF geometry before move generation

- Status: Accepted correction to a pre-implementation near-miss.
- Authority: Tier A WCDF clauses 1.4–1.5 and the official board diagram.
- Evidence stage: Correctness baseline.
- Context: The first Phase 1 ASCII diagram increased ACF numbers from Red's left to right. That
  mirrored the official orientation and placed the double corner on the wrong side.
- Decision: Freeze rows as `(4,3,2,1)`, `(8,7,6,5)`, …, `(32,31,30,29)` when viewed from Red's
  side. Internal zero-based square `s` still denotes ACF square `s + 1`; only coordinate mapping is
  affected.
- Evidence: `docs/RULES.md`, `tests/rules/test_board.py`, and
  `logs/test-output/000003-check-4.txt`. The defect was found before move-generation code existed,
  so no trained artifacts or transcripts were invalidated.
