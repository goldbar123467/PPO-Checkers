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

## ADR-007 — Delayed removal is state-significant but not move-set divergent

- Status: Accepted implementation; BLOCK-002 remains open against the contradictory gate text.
- Authority: Tier A WCDF 1.18–1.21 plus a derived coordinate-parity proof.
- Evidence stage: Correctness baseline.
- Decision: Retain captured pieces in opponent bitboards and mark them in `captured_pending` until
  the complete sequence ends. Exclude marked pieces as future jump midpoints. Do not fabricate a
  legal-set divergence fixture: short jumps keep the mover in one coordinate-parity class and all
  captured midpoints in the other, so a captured square can never be a later landing.
- Consequence: Mid-sequence state, serialization, observation, and mutation tests can distinguish
  correct delayed removal from immediate removal. Legal continuation sets cannot distinguish them
  when WCDF's no-repeat rule is honored.
- Evidence: BLOCK-002, `docs/RULES.md`, and the two R4.5 tests in
  `tests/rules/test_captures.py`.

## ADR-008 — Use only the valid composed board symmetry

- Status: Accepted implementation; BLOCK-003 remains open against the gate wording.
- Authority: Tier A WCDF board/movement clauses plus an exhaustive D4 geometry audit.
- Evidence stage: Metamorphic verification.
- Decision: Treat 180° board rotation composed with `PlayerId`/colour swap as the single
  nontrivial rank-preserving symmetry. Require it to be an involution and to commute with legal
  generation and state transition. Do not assert mirror-only, colour-only, or rotation-only
  invariance.
- Evidence: `tests/metamorphic/test_rules_symmetry.py` and BLOCK-003.

## ADR-009 — Separate transcript legality from publisher-adjudicated results

- Status: Accepted evidence classification; BLOCK-004 remains open against Gate 2 wording.
- Authority: Tier A move legality from WCDF; external published game records for move/result data.
- Evidence stage: External-anchor validation.
- Decision: Treat legal replay of every recorded move as rules correctness evidence. Preserve the
  publisher's result tag and verify it against the pinned extraction, but call it board-derived
  only when the final state itself has no legal move. Resignation, agreement, and adjudication are
  not inferred from a nonterminal board.
- Evidence: `tests/golden/test_published_transcripts.py`, fixture SHA-256
  `1ab2a5b530d2ff44d5595e0d7674ec521db12e99fed21cbe13c64f8790974311`, and BLOCK-004.
