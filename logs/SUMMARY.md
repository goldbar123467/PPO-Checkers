# Build Summary

## Active Work

- Phase 6 — offline RL core: IN PROGRESS; only T7 and the consolidated gate remain.
- Phase 1 remains formally BLOCKED by the read-only R6.6 classification defect in BLOCK-001;
  unaffected work proceeds under `GOAL.md` §0.1.
- Phase 2's feasible technical gate is complete but formally BLOCKED by BLOCK-002/003/004.
- Phase 3's feasible technical gate is complete but formally BLOCKED by BLOCK-001/005.
- Phase 4's feasible technical gate is complete but formally BLOCKED by inherited
  BLOCK-001/002/003/005. The Phase 6 N7 test resolved BLOCK-006.

## Completed Gates

- Gate 0 GREEN: scaffold, pinned environment, injected-red sensitivity checks, offline CI, and GPU
  doctor. See `logs/gates/phase-0.txt` and `logs/gates/phase-0-gpu-doctor.txt`.
- Gate 1 technical checks pass, but formal status is BLOCKED by BLOCK-001. See
  `logs/gates/phase-1.txt`.
- Gate 2 technical checks pass: 5M differential, published transcripts/perft, exact semantic
  mutation challenges, 95.76% killed-only mutation score, and 100% rules coverage. See
  `logs/gates/phase-2.txt` and `reports/phase2_mutation_analysis.md`.
- Gate 3 technical checks pass: R6 boundaries, loss-first precedence, boundary-only repetition,
  complete-vs-position key separation, and incremental hash equivalence through 50k reachable
  steps. See `logs/gates/phase-3.txt`.
- Gate 4 technical checks pass: canonical 128-action/eight-plane encoding, strict Gymnasium API,
  exact scalar/vector mid-sequence restore, illegal-action atomicity, and 100% current environment
  coverage. The immutable 5M fuzz report records all three failure counters at zero. See
  `logs/gates/phase-4.txt`, `logs/gates/phase-4-fuzz-5m.txt`, and
  `reports/phase4_environment_fuzz_5m_seed20260728.json`.
- Gate 5 GREEN: all 4,704 powered games, six exact checkpoint resumes, source-audited statistical
  caveats, and 608 tests plus eight property tests pass. See `reports/phase5_baseline_analysis.md`
  and `logs/gates/phase-5.txt`.
- Phase 6 focused work through D3 is green: masked sampling, signed GAE, chronological buffer,
  exact GroupNorm network, PPO-Clip, and ten-update CPU/GPU determinism all have independent
  oracles and 100% focused statement/branch coverage. T7 and the consolidated gate remain.

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
- BLOCK-005: §5.3 omits counters/ply from a key that claims to capture transition semantics. The
  implementation includes them and proves the 39/40 and 511/512 separation.
- BLOCK-006 was a phase-order defect; the implemented Phase 6 network now passes N7 and the blocker
  is resolved without backdating the Phase 4 evidence.

## Open Risks

- Total acceptance volume exceeded 400 at Gate 5; a final consolidated count remains required.
- Training, self-play, W&B, and full-budget ablations remain unrun and must not be described as
  completed. The powered fixed-agent baseline arena is complete.
- Clean-clone egress-blocked verification remains a final acceptance item.
