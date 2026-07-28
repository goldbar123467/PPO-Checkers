# Build Summary

## Active Work

- Phase 2 — State model, board, and move generation: IN PROGRESS.
- Phase 1 remains formally BLOCKED by the read-only R6.6 classification defect in BLOCK-001;
  unaffected work proceeds under `GOAL.md` §0.1.

## Completed Gates

- Gate 0 GREEN: scaffold, pinned environment, injected-red sensitivity checks, offline CI, and GPU
  doctor. See `logs/gates/phase-0.txt` and `logs/gates/phase-0-gpu-doctor.txt`.
- Gate 1 technical checks pass, but formal status is BLOCKED by BLOCK-001. See
  `logs/gates/phase-1.txt`.

## Known Failures and Root Causes

- Phase 1 initially drew the ACF diagram mirrored. Primary-source reinspection caught the issue
  before move generation; ADR-006 records the corrected frozen mapping.
- The first Phase 2 full check exposed a harness interaction: fuzz-only pytest reran global
  coverage over six scaffold tests and reported new rules modules at 0%. Fuzz/perft targets now
  use `--no-cov`; the mandatory full suite remains the single coverage authority.
- BLOCK-002: the requested R4.5 landing-block divergence is impossible under American Checkers'
  ±2 short-jump geometry. Exact delayed occupancy is implemented and tested; the coordinate-parity
  proof replaces any fabricated fixture.
- BLOCK-003: separate mirror, colour-only, and rotation-only transforms are not game symmetries.
  The valid 180° rotation plus player swap is tested as an involutive transition symmetry.
- BLOCK-004: published scores generally end by resignation/adjudication with legal moves remaining,
  so result tags are preserved but not falsely called board-derived.

## Open Risks

- The independent grid oracle agrees across 5,000,000 playout positions plus BFS depth 7; 20
  external scores replay legally. Phase 2 mutation analysis remains.
- Gate 2 still requires 5M differential positions, breadth-first depth evidence, published
  transcript replay, mutation ≥85%, and rules coverage ≥98%.
- Clean-clone egress-blocked verification remains a final acceptance item.
