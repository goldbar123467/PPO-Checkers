# Progress

## Current Phase

Phase 5 — Baselines and arena (IN PROGRESS)

Phases 1–4 are formally BLOCKED by specification defects recorded as BLOCK-001 through BLOCK-006.
Their technically feasible work and gates are complete, so §0.1 directs work to the next
source-correct portion of Phase 5.

Work label: deterministic, powered, replay-complete baseline evaluation.

Falsifiable objective: over every pair in the frozen four-agent population, run 784 colour-
balanced games (the predeclared two-sided 0.05-effect, alpha=.05, power=.80 budget), retain every
seed/action/outcome, and pass Gate 5 only if minimax(2) scores at least 0.40 versus minimax(1) and
the exact dev-tactical depth gate passes.

## In Flight

1. Commit the green, resumable baseline runner from an exact clean source tree.
2. Execute or resume all six powered comparisons (4,704 games total) and load-validate artifacts.
3. Diagnose results, source the report, run the consolidated Phase 5 gate, and update phase state.

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

## Last Five Iterations

- 000011: made the pinned Mutmut 3.6 harness isolation-safe, preserved each configuration failure,
  and killed a 15-mutant production probe before authorizing the full 968-mutant run.
- 000012: strengthened survivor assertions, killed 927/968 generated mutants conservatively,
  killed all five mandated semantic mutations in isolation, matched published perft, and completed
  the Phase 2 technical gate.
- 000013: filed BLOCK-005, implemented source-correct R6 outcomes and complete/narrow Zobrist keys,
  proved incremental equivalence through 50k reachable steps, and completed the Phase 3 technical
  gate.
- 000014: implemented canonical scalar/vector environments and exact mid-sequence resume, passed
  322 tests at 100% current checkers coverage, completed the 5M environment gate with all failure
  counters zero, and filed BLOCK-006 for the premature N7-logit requirement.
- 000015: completed and fully branch-tested the powered Phase 5 evaluation stack and crash-safe
  runner; corrected cross-match seed overlap before execution; `make check` passed 606+8 tests.

## Open Risks

- BLOCK-001 prevents Phase 1 from being marked GREEN until a human accepts the R6.6 erratum.
- BLOCK-002 prevents the impossible R4.5 landing-divergence fixture; exact delayed state and a
  coordinate-parity proof exist instead.
- BLOCK-003 prevents separate mirror/colour/rotation metamorphisms; the only valid nontrivial
  composition is fully transition-tested.
- BLOCK-004 prevents claiming publisher resignation/adjudication results were board-derived.
- BLOCK-005 prevents the incomplete §5.3 state-key field list; source-correct code includes
  sequence origin, both counters, and ply.
- BLOCK-006 prevents a Phase 4 claim about logits before the Phase 6 network exists; the distinct-
  observation regression passes now and the actual logit test remains scheduled for Phase 6.
- Exploitability-proxy evidence correctly remains unavailable until a trained Phase 7 best response
  exists; Phase 5 reports `NOT_EVALUATED` rather than substituting a heuristic.
- Test volume is 606 with none skipped/xfailed, exceeding the final ≥400 requirement; the existing
  276 substantive rules/environment tests exceed that category's ≥250 requirement.

## Next Step

Commit the green runner, then execute
`.venv-train/bin/python scripts/evaluate_baselines.py --progress-log logs/gates/phase-5-baseline-run.txt`
from that clean commit. Preserve and load-validate all checkpoints, raw games, and the final report.
