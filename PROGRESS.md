# Progress

## Current Phase

Phase 6 — RL core, verified offline (IN PROGRESS)

Phases 1–4 are formally BLOCKED by specification defects recorded as BLOCK-001 through BLOCK-006.
Their technically feasible work and gates are complete, so §0.1 directs work to the next
source-correct portion of Phase 6. Phase 5 is GREEN.

Work label: offline RL-core verification.

Engineering objective: implement masked categorical sampling, two-player perspective-aware GAE,
the rollout buffer, GroupNorm residual network, and PPO update so every T1–T8 numerical oracle and
D2/D3 determinism requirement passes offline without exceeding the 12 GB local VRAM envelope.

## In Flight

1. Run the consolidated Gate 6 static, coverage, CPU, and RTX checks.
2. Publish the source-bounded Phase 6 analysis and advance state only if the full gate is green.

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
- Full mutation gate: 927 of 968 mutants killed (95.76% killed-only), 39 survivors disclosed, two
  infinite-loop timeouts, and zero untested/skipped mutants. All five exact §12.4 C semantic
  challenges fail their permanent tests after a green unmodified baseline. Evidence
  `reports/phase2_mutation_analysis.md`, `reports/phase2_mutation_stats.json`, and
  `reports/phase2_rule_mutation_challenges.json`.
- External perft: Aart Bik's published American-checkers counts match exactly through completed-
  move depth 7 (179,740 leaves). Evidence `tests/golden/data/external_perft.json` and
  `logs/test-output/000016-external-perft.txt`.
- Phase 2 consolidated technical gate: `make check`, exit 0; 158 tests plus eight property tests,
  strict typing/lint/format, and 100% statement/branch rules coverage. Evidence
  `logs/gates/phase-2.txt`. Formal status is BLOCKED only by BLOCK-002/003/004.
- Phase 3 consolidated technical gate: `make check`, exit 0; 188 tests plus eight property tests;
  R6 boundary fixtures, narrow/full key separation, and 50k incremental/recomputed/undo checks;
  all eight rules modules at 100% statement/branch coverage. Evidence `logs/gates/phase-3.txt`.
  Formal status is BLOCKED only by BLOCK-001/005.
- Phase 4 scalar/vector environment gate: `make check`, exit 0; 322 tests plus eight property
  tests; 276 tests under `rules/` and `env/`; every current checkers module at 100%
  statement/branch coverage. Evidence `logs/gates/phase-4.txt`.
- Phase 4 large fuzz: exactly 5,000,000 transitions and 120,154 terminal games; 1,015,426 capture
  steps, 121,260 continuations, 224,842 promotions, and 500 snapshot reloads; zero invariant
  violations, mask disagreements, and empty nonterminal masks. Report SHA-256
  `1472e4ea…0f479ed`; evidence `reports/phase4_environment_fuzz_5m_seed20260728.json` and
  `logs/gates/phase-4-fuzz-5m.txt`.
- Phase 5 deterministic layers: random/greedy/minimax agents, alternating-colour arena, powered
  score intervals, Elo transforms, full payoff/population diagnostics, and exact tactical suite
  are committed through `757f826`; the tactical artifact reproduces byte-for-byte with 50/50
  depth-3 versus 15/50 depth-1 solves.
- Phase 5 runner preflight: frozen six-edge YAML; lossless action/seed/outcome checkpoints;
  deterministic gzip archive; exact identity checks; disjoint 2,352-stream blocks per matchup;
  no sealed-suite query; explicit unavailable/proxy labels. Focused new modules have 100%
  statement/branch coverage. Full `make check` exit 0: 606 tests plus eight property tests, strict
  static gates, all 3,162 Checkers statements and 1,120 branches covered. Evidence
  `logs/test-output/000060-phase5-baseline-runner-final-check.txt`.
- First powered run completed all 4,704 games and persisted valid raw actions, but report audit
  rejected its conclusion: depth-2 solved 10/50 versus depth-1's 16/50 while the aggregate flag
  said no non-monotonicity. The audit also found suite-order dependence from a shared tie-break RNG.
  The failed report/run are preserved in `logs/test-output/000061-*`; they are not Gate 5 evidence.
- Tactical-report correction: each case now receives a fresh same-seed policy, matching the suite
  generator and making results order-independent (depths 1/2/3 solve 15/9/50). The 1→2 regression
  is explicitly diagnosed as a non-quiescent material-evaluator horizon effect, while the
  predeclared 1→3 gate remains green. `make check` exit 0: 608 tests plus eight property tests and
  100% statement/branch Checkers coverage. Evidence
  `logs/test-output/000063-phase5-tactical-report-fix-check.txt`.
- Corrected powered rerun at `e386b31` reproduced every one of the 4,704 prior game records exactly;
  all six checkpoint resumes regenerated byte-identical raw/report artifacts. Source audit then
  required stronger machine-readable caveats: pseudorandom seed uniqueness does not prove
  independence; the NIST continuity correction is not used; only the standard opening is sampled;
  the 100-Elo residual threshold is project-defined; and the dev suite was construction-selected.
  The caveat regression went RED then `make check` passed 608+8 tests at 100% coverage. Evidence
  `logs/test-output/000064-*` and `000065-phase5-statistical-caveats-check.txt`.
- Phase 5 final powered run at source `6deefb9` completed all six 784-game pairs (4,704 games),
  then a real resume loaded all six checkpoints and regenerated byte-identical raw/report
  artifacts. Raw SHA-256 is `c5ca9d1d…d19d00b4`; report SHA-256 is
  `2a866255…94504c4`. Minimax(2) scored 0.9968 [0.9898, 0.9990] versus minimax(1),
  and tactical depths 1/2/3 solved 15/9/50, explicitly retaining the depth-2 regression.
  The consolidated gate replayed all records and passed formatting, Ruff, strict mypy, 608 tests,
  eight property tests, and 100% Checkers statement/branch coverage. Evidence:
  `reports/phase5_baseline_analysis.md`, `reports/phase5_baseline_report_v1.json`,
  `reports/phase5_baseline_games_v1.json.gz`, and `logs/gates/phase-5.txt`.
- Phase 6 foundation: froze the complete T1–T8/D2–D3 oracle matrix, then implemented the sole
  dtype-aware masked categorical and the time-major two-player GAE recursion. The RED import test
  failed as intended; the first quality pass exposed real formatting/typing/coverage defects; the
  corrected gate passed 41 tests with both modules at 100% statement/branch coverage. The explicit
  four-step hand oracle includes both perspective signs, truncation/terminal tests pass, illegal
  logits receive exact-zero gradient, and the sigma-degenerate path is bitwise equal to a separate
  conventional GAE loop. Evidence: `docs/PHASE6_TEST_MATRIX.md` and
  `logs/test-output/000066-*` through `000069-*`.
- Phase 6 chronological buffer: T8 computes signed GAE over complete time-major vector lanes before
  exposing trainable-only policy/value views. Stored masks and observations are cloned/detached;
  stable environment IDs are enforced; a rollout ending mid-capture bootstraps with positive
  perspective sign; finalization is one-use. The initial GREEN attempt caught and corrected a hand-
  oracle sign and a mislabeled trainable fixture. The final gate passed 50 tests with all 211
  statements and 96 branches covered. Evidence `logs/test-output/000070-*` through `000074-*`.
- Phase 6 network: the exact six-block shared GroupNorm trunk and separate policy/value heads have
  470,410 parameters (1.794 MiB FP32). Structural tests prohibit BatchNorm and pin every layer,
  group count, output shape/range, and orthogonal gain. T1 policy accuracy is 1.0 on 64 fixed
  states; T2 value MSE is `4.69e-14`; T5 reaches every parameter and gives every major module a
  nonzero aggregate gradient. The deferred N7 state pair now produces different logits, resolving
  BLOCK-006. Focused strict gates pass 13 tests at 100% statement/branch coverage. Evidence:
  `logs/test-output/000075-*` through `000081-*`.
- Phase 6 PPO core: the literal T3 oracle matches signed GAE, ratios, clipped surrogate, plain-MSE
  value loss, masked entropy, k3 KL, clip fraction, and total loss within `1e-12`. T4 fixed-batch
  probabilities move monotonically in the advantage direction until both ratio bounds clip and
  plateau. The update uses stored masks, trainable-only rows, Adam-compatible zeroing, and observed
  global norm clipping. The RED/fixture-correction trail is retained; the final strict gate passes
  31 tests with all 164 statements and 56 branches covered. Evidence `logs/test-output/000082-*`
  through `000087-*`.
- Phase 6 determinism: one SplitMix-derived root seeds Python, NumPy, Torch, CUDA, and four stable
  environment streams. A ten-update fixture rebuilt twice yields bitwise-identical CPU actions and
  loss diagnostics; on the same RTX 5070, actions are identical and diagnostics meet
  `atol=1e-5, rtol=1e-4`. Native CUDA BF16 masking remains finite/legal with exact-zero illegal
  gradients. The hardware/software/BLAS stack and deterministic flags are pinned in
  `logs/gates/phase-6-determinism.txt`; 13 tests and the 68-statement/16-branch module are at 100%
  focused coverage in `logs/test-output/000091-*`.
- Phase 6 T7: real rules/environment trajectories of exactly 3/5/7 forced steps produce literal
  actor-frame targets. A distinct R6.2 ending and terminal four-jump sequence are covered. Removing
  recursive `sigma` makes all three fail; restored code passes all four focused tests. Evidence
  `logs/test-output/000092-*` through `000097-*`.

## Last Five Iterations

- 000020: implemented full-chronology lockstep rollout storage, proved trainable filtering occurs
  only after GAE, and completed T8 at 100% focused statement/branch coverage.
- 000021: implemented and structurally verified the exact GroupNorm network, passed T1/T2/T5 and
  N7, and resolved the phase-order-only BLOCK-006 with the real logit regression.
- 000022: implemented stored-mask PPO-Clip, matched the literal T3 oracle to 1e-12, passed T4
  directional clipping, and covered the module completely.
- 000023: unified deterministic seeding and proved ten-update D2 CPU bitwise reproduction plus D3
  same-RTX tolerance and native BF16 masking against a fully recorded reference stack.
- 000024: froze three uniquely forced 3/5/7-step engine trajectories, proved exact actor-frame T7
  targets, and killed a deliberately injected missing-recursive-sign defect.

## Open Risks

- BLOCK-001 prevents Phase 1 from being marked GREEN until a human accepts the R6.6 erratum.
- BLOCK-002 prevents the impossible R4.5 landing-divergence fixture; exact delayed state and a
  coordinate-parity proof exist instead.
- BLOCK-003 prevents separate mirror/colour/rotation metamorphisms; the only valid nontrivial
  composition is fully transition-tested.
- BLOCK-004 prevents claiming publisher resignation/adjudication results were board-derived.
- BLOCK-005 prevents the incomplete §5.3 state-key field list; source-correct code includes
  sequence origin, both counters, and ply.
- BLOCK-006 is resolved: the earlier phase-order defect remains documented and the Phase 6 N7
  different-logits test now passes against the exact implemented network.
- Exploitability-proxy evidence correctly remains unavailable until a trained Phase 7 best response
  exists; Phase 5 reports `NOT_EVALUATED` rather than substituting a heuristic.
- Test volume is 608 with none skipped/xfailed, exceeding the final ≥400 requirement; the existing
  276 substantive rules/environment tests exceed that category's ≥250 requirement.

## Next Step

Run the consolidated Gate 6 check over the complete repository, including strict static gates,
branch coverage, property tests, and the real CPU/RTX determinism paths.
