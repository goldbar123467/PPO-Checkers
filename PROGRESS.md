# Progress

## Current Phase

Phase 2 — State model, board, and move generation (IN PROGRESS)

Phase 1 is BLOCKED by P1 BLOCK-001, so §0.1 directs work to the next non-blocked phase.

Work label: correctness baseline.

Falsifiable objective: a typed immutable state and two independently structured legal-step
generators agree on R1–R5/R7, including exact delayed-removal state. BLOCK-002 records the proof
that the specification's requested legal-continuation divergence cannot exist under short-jump
geometry.

## In Flight

1. Configure and run mutation analysis over `rules/`; add targeted killing tests as needed.
2. Run the final Phase 2 technical gate and mark its formal status with all P1 blockers.
3. Advance to unaffected Phase 3 terminal conditions and hashing per `GOAL.md` §0.1.

## Gate Evidence

- Gate 0: `make check`, exit 0; 19 passed, 0 failed; scaffold coverage 100%;
  `logs/gates/phase-0.txt`.
- Sensitivity: failing-test and Ruff F401 probes both exited nonzero; evidence under
  `logs/test-output/000001-injected-*-red.txt`.
- GPU doctor after lock change: exit 0; `logs/gates/phase-0-gpu-doctor.txt`.
- Phase 1 technical gate: `make check`, exit 0; 24 passed; `logs/gates/phase-1.txt`.
  Phase remains BLOCKED, not GREEN, pending BLOCK-001 resolution.
- Phase 2 board/state increment: `make check`, exit 0; 57 passed; `src/checkers/rules/board.py`
  and `state.py` each at 100% statement/branch coverage; evidence
  `logs/test-output/000003-check-4.txt`.
- Phase 2 moves/oracle increment: `make check`, exit 0; 86 passed; all four rules modules at 100%
  statement/branch coverage; depth-5 BFS compared 3,811 unique frontier states and discovered
  12,916 states with zero disagreements. Evidence `logs/test-output/000005-check-2.txt`.
- Phase 2 notation/fuzz increment: `make check`, exit 0; 118 passed; rules coverage 100%;
  deterministic 50,000-step invariant/oracle fuzz and 200 Hypothesis trajectories pass. The valid
  composed rotation/colour-swap relation commutes with transitions through BFS depth 4. Evidence
  `logs/test-output/000006-check-3.txt`.
- Differential runner validation: `make check`, exit 0; 125 passed; rules coverage 100%; a
  1,000-position CLI smoke report loaded successfully and reproduced digest
  `c74d17b…c0b35`. Evidence `logs/test-output/000008-check.txt`.
- Phase 2 large differential: 5,000,000 playout positions and 40,801 BFS positions compared at
  depth 7; 123,632 unique BFS states discovered; zero disagreements; exit 0. Report SHA-256
  `af0c85dd…9e7b5dbe`; evidence `reports/phase2_differential_5m_seed20260727.json` and
  `logs/gates/phase-2-differential.txt`.
- Published transcripts: 20 hash-pinned external scores, 515 completed moves / 561 steps, all legal
  with unique resolution; 21 tests pass. Evidence `logs/test-output/000010-transcripts.txt`.
- Mutation harness: Mutmut 3.6 collected 127 isolated tests, passed its clean/forced-failure
  controls, and killed all 15 `coord()` mutants. The repository gate remains at 100% rules
  statement/branch coverage with 148 plus eight tests. Evidence
  `logs/test-output/000011-mutmut-probe-8.txt` and `000011-check.txt`.

## Last Five Iterations

- 000007: 50k deterministic fuzz, Hypothesis trajectories, and valid combined symmetry pass;
  BLOCK-003 records why separate mirror/colour/rotation claims are false.
- 000008: added a deterministic, metadata-bearing 5M differential CLI; its 1,000-position smoke
  and 125-test repository gate pass at 100% rules coverage.
- 000009: full 5M plus BFS-depth-7 differential completed in 396.35 seconds with zero disagreements;
  independent report validation passed after one expected-SHA typo was caught.
- 000010: 20 published PDN games replayed all 515 moves legally; BLOCK-004 separates preserved
  publisher results from outcomes that cannot be inferred from nonterminal boards.
- 000011: made the pinned Mutmut 3.6 harness isolation-safe, preserved each configuration failure,
  and killed a 15-mutant production probe before authorizing the full 968-mutant run.

## Open Risks

- BLOCK-001 prevents Phase 1 from being marked GREEN until a human accepts the R6.6 erratum.
- BLOCK-002 prevents the impossible R4.5 landing-divergence fixture; exact delayed state and a
  coordinate-parity proof exist instead.
- BLOCK-003 prevents separate mirror/colour/rotation metamorphisms; the only valid nontrivial
  composition is fully transition-tested.
- BLOCK-004 prevents claiming publisher resignation/adjudication results were board-derived.
- Phase 2 mutation ≥85% and its final consolidated technical gate remain.

## Next Step

Run all 968 generated rules mutants, add targeted killing tests for actionable survivors, and
export a structured mutation-score report.
