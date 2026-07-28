# Progress

## Current Phase

Phase 2 — State model, board, and move generation (IN PROGRESS)

Phase 1 is BLOCKED by P1 BLOCK-001, so §0.1 directs work to the next non-blocked phase.

Work label: correctness baseline.

Falsifiable objective: a typed immutable state and two independently structured legal-step
generators agree on R1–R5/R7, including a hand-constructed R4.5 position where delayed and immediate
removal produce different continuations.

## In Flight

1. Write failing R3–R5 tests, then implement the fast legal-step/apply/undo path with delayed removal.
2. Write an independently structured oracle plus R4.5 divergence and differential tests.
3. Implement R7 notation and complete-state serialization with exact mid-sequence round trips.

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

## Last Five Iterations

- 000001: Gate 0 GREEN; injected lint/test failures detected; CUDA/BF16/NF4 doctor passed.
- 000002: WCDF source hash-pinned; 30-rule matrix and R6.7 proof pass; R6.6 label defect filed.
- 000003: corrected mirrored ACF diagram before move code; immutable uint32 state and board mapping
  pass 33 focused tests and the 57-test repository gate at 100% checkers coverage.

## Open Risks

- BLOCK-001 prevents Phase 1 from being marked GREEN until a human accepts the R6.6 erratum.
- Delayed-removal semantics are high risk and require a real divergence fixture, not only prose.
- Phase 2's 5M differential, published transcripts, mutation, and coverage gates are not yet run.

## Next Step

Write failing fast-path legal-step tests for R2–R5, including delayed-removal continuation state.
