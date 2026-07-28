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

1. Implement R7 notation and complete-state serialization with exact mid-sequence round trips.
2. Add reachable-state property/metamorphic suites and the 5M-position differential runner.
3. Replay at least 20 published WCDF transcripts and run mutation analysis over `rules/`.

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

## Last Five Iterations

- 000001: Gate 0 GREEN; injected lint/test failures detected; CUDA/BF16/NF4 doctor passed.
- 000002: WCDF source hash-pinned; 30-rule matrix and R6.7 proof pass; R6.6 label defect filed.
- 000003: corrected mirrored ACF diagram before move code; immutable uint32 state and board mapping
  pass 33 focused tests and the 57-test repository gate at 100% checkers coverage.
- 000004: implemented mandatory captures, continuation, delayed removal, promotion, counters, and
  O(1) immutable undo; filed BLOCK-002 after exhaustive parity proof.
- 000005: independent object-grid oracle agrees with the bitboard path on hand fixtures and 3,811
  BFS states through depth 5; repository gate is 86 tests at 100% rules coverage.

## Open Risks

- BLOCK-001 prevents Phase 1 from being marked GREEN until a human accepts the R6.6 erratum.
- BLOCK-002 prevents the impossible R4.5 landing-divergence fixture; exact delayed state and a
  coordinate-parity proof exist instead.
- Phase 2's 5M differential, published transcripts, mutation, and coverage gates are not yet run.

## Next Step

Write failing R7 notation and full-state serialization round-trip tests.
