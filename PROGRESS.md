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

1. Add the 5M-position differential runner and phase-gate report.
2. Replay at least 20 published WCDF transcripts with source/result provenance.
3. Run mutation analysis over `rules/` and add targeted mutation-killing tests as needed.

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

## Last Five Iterations

- 000004: implemented mandatory captures, continuation, delayed removal, promotion, counters, and
  O(1) immutable undo; filed BLOCK-002 after exhaustive parity proof.
- 000005: independent object-grid oracle agrees with the bitboard path on hand fixtures and 3,811
  BFS states through depth 5; repository gate is 86 tests at 100% rules coverage.
- 000006: strict ACF move grammar and CHK1 complete-state format pass canonical, invalid, and exact
  mid-sequence round-trip tests; repository gate remains at 100% rules coverage.
- 000007: 50k deterministic fuzz, Hypothesis trajectories, and valid combined symmetry pass;
  BLOCK-003 records why separate mirror/colour/rotation claims are false.
- 000008: added a deterministic, metadata-bearing 5M differential CLI; its 1,000-position smoke
  and 125-test repository gate pass at 100% rules coverage.

## Open Risks

- BLOCK-001 prevents Phase 1 from being marked GREEN until a human accepts the R6.6 erratum.
- BLOCK-002 prevents the impossible R4.5 landing-divergence fixture; exact delayed state and a
  coordinate-parity proof exist instead.
- BLOCK-003 prevents separate mirror/colour/rotation metamorphisms; the only valid nontrivial
  composition is fully transition-tested.
- Phase 2's 5M differential, published transcripts, mutation, and coverage gates are not yet run.

## Next Step

Implement and run the saved 5M-position fast/oracle differential gate.
