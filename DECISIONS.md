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

## ADR-010 — Make the rules package compatible with Mutmut 3.6 isolation

- Status: Accepted.
- Authority: Tier B test-tool engineering; pinned Mutmut 3.6.0 source and documentation.
- Evidence stage: Mutation-harness validation.
- Context: Mutmut 3 uses generated trampolines and executes pytest from an isolated `mutants/`
  tree. Import-time calls into functions that Mutmut instruments can resolve to generated
  filenames such as `<string>` or `<frozen importlib._bootstrap>`, and nested non-source fixtures
  are not copied unless their parent directory is declared.
- Decision: Freeze the already-tested 32-square geometry as an immutable constant; give the three
  small frozen value dataclasses explicit typed constructors; copy the published fixture and docs
  directories into the isolated tree; and use Mutmut's current configuration-driven CLI. Do not
  set `max_stack_depth`: the installed 3.6.0 implementation shows that any non-default value walks
  and resolves the Python stack on every instrumented call, while the default records all touched
  functions without that filter.
- Sources:
  - Mutmut project documentation: <https://mutmut.readthedocs.io/en/latest/>
  - Pinned package: `mutmut==3.6.0`, wheel SHA-256
    `a9f5b8dcf6cbf9496769d7cf8bdbba37a0ec709ad98f88d103238b62f10bdf37` in `uv.lock`.
- Evidence: `logs/test-output/000011-mutmut-probe-8.txt` kills all 15 selected `coord()`
  mutants; `logs/test-output/000011-check.txt` proves 148 repository tests plus eight property
  tests pass with 100% statement/branch coverage over `src/checkers/rules`.

## ADR-011 — Score mutation testing conservatively and challenge exact semantic defects

- Status: Accepted.
- Authority: `GOAL.md` §12.4 C and Tier B mutation-testing practice.
- Evidence stage: Correctness baseline.
- Decision: Use all 968 generated rules mutants as the denominator and count only explicit kills
  toward the reported score. Do not remove equivalent mutants or count timeouts to reach the gate.
  In addition, apply the five exact semantic defects named by the goal in isolated temporary trees,
  because syntactic mutation generation does not faithfully construct all five.
- Result: 927/968 killed-only (95.76%), 39 survivors disclosed, two infinite-loop timeouts, and all
  five exact semantic challenges killed after an unmodified baseline passes.
- Evidence: `reports/phase2_mutation_analysis.md`, `reports/phase2_mutation_stats.json`, and
  `reports/phase2_rule_mutation_challenges.json`.

## ADR-012 — Count external perft in completed checkers moves

- Status: Accepted.
- Authority: Aart J. C. Bik, *Computing Deep Perft and Divide Numbers for Checkers*, ICGA Journal
  35(4), 206–213 (2012), DOI `10.3233/ICG-2012-35403`.
- Evidence stage: External-anchor validation.
- Decision: Compare start-position perft by decrementing depth only on `move_completed=True`.
  Capture continuations are separate environment steps in this project but one move in the
  published perft definition.
- Result: Published leaf counts match exactly through depth 7 (179,740 leaves); deeper published
  values remain pinned but are not redundantly run in every repository gate.
- Evidence: `tests/golden/data/external_perft.json`,
  `tests/golden/test_external_perft.py`, and `logs/test-output/000016-external-perft.txt`.

## ADR-013 — Use a complete transition key and a narrow repetition key

- Status: Accepted implementation; BLOCK-005 remains open against the contradictory §5.3 list.
- Authority: Project Markov-state contract plus Albert L. Zobrist, *A New Hashing Method With
  Application for Game Playing*, University of Wisconsin Technical Report 88 (1970),
  <https://minds.wisconsin.edu/handle/1793/57624>.
- Evidence stage: Terminal/hash baseline.
- Decision: `position_key` hashes placement and side only and raises during a capture sequence.
  `state_key` hashes every field that changes a future complete state or terminal transition,
  including `sequence_origin`, both no-progress counters, and ply. Freeze the 64-bit expansion
  scheme and verify XOR incremental updates against full recomputation and reverse updates.
- Rationale: Counter 39/40 and ply 511/512 pairs are deterministic semantic collisions under the
  goal's explicit incomplete field list. A transposition or dedup key cannot merge them safely.
- Evidence: `tests/rules/test_zobrist.py`, the 50,000-step reachable-state property in
  `tests/property/test_rules_properties.py`, and `logs/gates/phase-3.txt`.

## ADR-014 — Resolve coincident terminal boundaries loss-first

- Status: Accepted.
- Authority: Tier A WCDF 1.30 for no-piece/no-move losses; project-defined R6.3–R6.5 draws.
- Evidence stage: Terminal/hash baseline.
- Decision: Evaluate no-piece and no-legal-move loss before automatic no-progress, repetition, or
  ply-cap draws. Treat `ply >= max_plies` as the draw boundary because Gate 3 explicitly requires
  511 versus 512 fixtures despite R6.5's prose saying "exceeding."
- Consequence: A training-only draw rule cannot rescue a player who already lost under WCDF. The
  boundary behavior is deterministic and permanently tested.
- Evidence: `tests/rules/test_terminal.py` and `logs/gates/phase-3.txt`.

## ADR-015 — Use canonical step-wise Gymnasium actions

- Status: Accepted.
- Authority: `GOAL.md` §§5.2 and 6; official [Gymnasium `Env`
  API](https://gymnasium.farama.org/api/env/).
- Evidence stage: Environment baseline.
- Decision: Expose `Discrete(128)` as canonical origin × direction and make one API step equal one
  simple move or one jump. Rotate White states/actions 180° so one shared policy always views
  itself as the actor. Preserve actor identity explicitly in info because canonical tensors do not
  encode stable colour identity.
- Consequence: Multi-jumps span several timesteps and retain the same actor; GAE must later apply
  the explicit perspective sign. The smaller fixed action space is retained because every short
  step maps bijectively and the full environment contract now tests the sequencing cost.
- Evidence: `tests/env/test_actions.py`, `tests/env/test_encoding.py`, and
  `logs/gates/phase-4.txt`.

## ADR-016 — Persist environment history needed beyond the Markov rules state

- Status: Accepted.
- Authority: `GOAL.md` §§5.1–5.2, 7.5, and R7.3.
- Evidence stage: Environment baseline.
- Decision: Serialize current and reset `State`, rule configuration, sorted repetition counts, and
  the partial ACF path. Validate the entire `CHECKERS_ENV_1` record before mutating a live wrapper.
- Rationale: The rules state is sufficient for future legal transitions, but repetition depends on
  prior boundary visits and the eventual full move notation depends on earlier jump landings.
- Evidence: `tests/env/test_serialization.py`; exact `9x18x25` completion after mid-sequence
  restore.

## ADR-017 — Make vector batches transactional and masks authoritative

- Status: Accepted.
- Authority: `GOAL.md` §§6.1, 6.5, and 7.5; Huang and Ontañón,
  [*A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms*](https://arxiv.org/abs/2006.14171).
- Evidence stage: Environment baseline.
- Decision: A nonterminal lane exposes exactly its generated legal IDs; terminal lanes expose an
  all-false mask. Illegal actions raise before mutation. The synchronous vector wrapper
  prevalidates every lane before advancing any lane by one environment step.
- Consequence: Gymnasium's generic checker cannot uniformly sample `Discrete(128)` and assume every
  ID is legal; passive API checks use a known legal ID, while explicit tests prove all invalid IDs
  raise. One bad vector lane never leaves a partially advanced batch.
- Evidence: `tests/env/test_environment.py`, `tests/env/test_vec_env.py`, and 5M gate metrics.

## ADR-018 — Classify the 5M environment gate as randomized evidence, not proof

- Status: Accepted.
- Authority: `GOAL.md` §§12.3–12.5 and the research-behavior contract.
- Evidence stage: Phase gate.
- Decision: Run a fixed-seed mixture of full games and adversarial boundary fixtures; check masks,
  transitions, counters, notation, rewards, observations, hashes, and periodic snapshot reloads on
  every applicable state. Halt on the first failure and write a new immutable report only after a
  complete run.
- Result: 5,000,000 steps completed with zero invariant violations, zero mask disagreements, and
  zero empty nonterminal masks. Report SHA-256
  `1472e4ea1da80f591ee248748d066fdb05bea72cc78f3a0f5f9aecebb0f479ed`.
- Limitation: Agreement with recomputation and randomized coverage is strong regression evidence,
  not exhaustive proof or independent external rules correctness.

## ADR-019 — Power and preserve the complete fixed-baseline population

- Status: Accepted for the Phase 5 baseline experiment.
- Authority: `GOAL.md` §§11.1, 11.3–11.4; NIST/SEMATECH sample-size derivation;
  Wilson (1927), DOI `10.1080/01621459.1927.10502953`; Bradley & Terry (1952), DOI
  `10.1093/biomet/39.3-4.324`; official FIDE rating regulations.
- Evidence stage: Baseline, before any learned policy exists.
- Decision: Freeze random, greedy, minimax(1), and minimax(2); evaluate all six unordered pairs at
  784 alternating-colour games. This is the even ceiling above the NIST normal-approximation result
  of 783 games for a two-sided score change of 0.05 from 0.50, alpha .05, and power .80.
- Statistics: retain W/D/L and Wilson-style fractional-score intervals; label draw coverage and
  league-Elo delta-method CIs approximate; report residuals and directed 3-cycles rather than
  treating scalar Elo as ground truth.
- Reproducibility: assign each comparison a disjoint contiguous block of `3 × 784` SplitMix64
  inputs, checkpoint every complete match atomically, and retain every game seed/action/outcome in
  a deterministic gzip archive bound to Git/config/goal hashes.
- Limitations: do not query the sealed suite; report external anchor `NOT_AVAILABLE` and the
  Phase-7-trained best-response proxy `NOT_EVALUATED` until those inputs genuinely exist.
- Evidence: `configs/checkers-baselines-v1.yaml`, `tests/eval/test_baseline_*.py`, and
  `logs/test-output/000060-phase5-baseline-runner-final-check.txt`.

## ADR-020 — Isolate tactical cases and report every depth regression

- Status: Accepted after rejecting the first Phase 5 report.
- Authority: `GOAL.md` §11.3 and the controlled-experiment requirement.
- Evidence stage: Baseline report audit.
- Finding: the first report reused one stateful seeded tie-break RNG across all cases. Its depth-1
  count (16/50) therefore differed from the generator's fresh-policy-per-case contract (15/50) and
  depended on case order. It also showed depth-2 at 10/50 but set aggregate non-monotonicity false
  because that flag inspected arena scores only.
- Decision: instantiate a fresh policy with the declared seed for each tactical case; permanently
  test order independence and manifest agreement; inspect every adjacent tactical depth as well as
  common arena anchors when deriving the non-monotonicity flag.
- Diagnosis rule: a depth regression under the non-quiescent material evaluator is recorded as a
  horizon-effect/evaluator finding, not automatically an engine defect and not silently omitted.
  The predeclared depth-1 versus depth-3 tactical decision remains unchanged.
- Evidence: `logs/test-output/000061-phase5-report-semantic-red.json`,
  `000062-phase5-tactical-report-red.txt`, and `000063-phase5-tactical-report-fix-check.txt`.

## ADR-021 — Scope statistical claims to the sourced approximation

- Status: Accepted after primary-source audit.
- Authority: NIST/SEMATECH §7.2.4.2; Wilson (1927); Bradley & Terry (1952); official FIDE rating
  regulations. Evidence stage: final baseline-report audit.
- Decision: retain 784 games because it attains .80074 power under the explicitly declared
  uncorrected two-sided normal approximation (raw ceiling 783), but state that NIST's separately
  recommended continuity correction was not applied. Do not silently imply exact power.
- Seed scope: describe SplitMix64-derived streams as distinct pseudorandom seeds. Statistical
  independence and representativeness of that seed schedule remain model assumptions, not facts
  established by injectivity.
- Evaluation scope: state that every game starts at the standard opening; label the 100-Elo
  transitivity residual threshold project-defined; disclose that the dev tactical suite was
  selected for depth-3 success and is not an unbiased or sealed sample.
- Evidence: primary URLs embedded in `reports/phase5_baseline_report_v1.json`; permanent caveat
  assertions in `tests/eval/test_baseline_eval.py`; `logs/test-output/000065-*.txt`.

## ADR-022 — Accept the replay-complete Phase 5 baseline

- Status: Accepted; Gate 5 GREEN.
- Evidence stage: Final powered baseline at source revision
  `6deefb959cc995517b5bbe3c452610e99058adc8`.
- Decision: Accept the frozen six-pair, 784-game-per-pair population as the Phase 5 engineering
  baseline. Minimax(2) scored 0.9968 [0.9898, 0.9990] against minimax(1), above the 0.40
  catastrophic-inversion floor. Tactical depths 1/2/3 solved 15/9/50; depth 3 is a strict
  superset of depth 1, while the depth-2 regression remains explicit.
- Population scope: zero strict three-cycles were observed. League Elo is conditional on the
  checked approximate-transitivity model; the 92.33-Elo maximum residual is below, but close to,
  the project-defined 100-Elo diagnostic. The trained-best-response exploitability proxy remains
  `NOT_EVALUATED`, the external anchor `NOT_AVAILABLE`, and sealed evaluation `NOT_EVALUATED`.
- Reproducibility: the raw archive SHA-256 is
  `c5ca9d1d446a4462932b80bcc8570b5a0a778c38261f3faa46f08751d19d00b4`; the machine-report
  SHA-256 is `2a866255e9ed86b771d130bb8e9a728ce0289d94c7f9a6fb4a1bb543594504c4`.
  A second invocation resumed all six identity-bound checkpoints and regenerated both artifacts
  byte-for-byte. The consolidated gate independently parsed all 4,704 replay records and passed
  608 tests plus eight property tests at 100% Checkers statement/branch coverage.
- Evidence: `reports/phase5_baseline_analysis.md`, `reports/phase5_baseline_report_v1.json`,
  `reports/phase5_baseline_games_v1.json.gz`, `logs/gates/phase-5-baseline-run.txt`,
  `logs/gates/phase-5-baseline-run-resume.txt`, and `logs/gates/phase-5.txt`.

## ADR-023 — Make GAE time-major and masking singular

- Status: Accepted foundation for Phase 6.
- Authority: `GOAL.md` §§6.5 and 7.3–7.5; original GAE paper for the single-agent recursion;
  official PyTorch categorical-distribution contract. The perspective-sign extension is a project
  derivation validated by tests, not by citation.
- Decision: expose one `MaskedCategorical` wrapper everywhere and reject an all-false row before
  construction. Replace illegal logits with the input dtype's finite minimum before the underlying
  categorical normalization. Expose a time-major signed-GAE function whose trailing dimensions
  are independent environment lanes and whose final bootstrap has the shape of one time row.
- Rationale: one distribution prevents rollout/update mask drift. Time-major GAE makes vector-lane
  adjacency explicit, handles a slice ending mid-capture with a per-lane bootstrap, and reduces T8
  to a testable reshape/grouping responsibility in the buffer rather than hidden global state.
- Evidence: 41 focused tests, including float32/BF16 and `k=1` mask cases, exact illegal-gradient
  zeros, a four-transition hand oracle with both signs, terminal/truncation boundaries, colour-swap
  negation, batched lanes, and bitwise single-agent equivalence. Both modules are at 100%
  statement/branch coverage in `logs/test-output/000069-phase6-mask-gae-quality-final.txt`.

## ADR-024 — Store rollouts as complete lockstep chronology

- Status: Accepted; T8 GREEN.
- Authority: `GOAL.md` §§7.4–7.5 and the project-derived signed recursion of §7.3.
- Decision: append exactly one complete row containing every stable environment lane; clone and
  detach every field, including the legal mask; compute GAE on the resulting `(time, env)` tensors;
  flatten only afterward; and derive policy/value optimization views from the stored `trainable`
  flags. Default value loss is trainable-only, while the declared opponent-state ablation requires
  an explicit `include_nontrainable=True` view.
- Invariants: environment IDs equal stable lane order on every row, selected actions are legal under
  the stored masks, actor/sign/reward domains are exact, capacity cannot overflow, and a rollout is
  finalized only once. These constraints make filtered-pre-GAE data and accidental replay fail
  loudly.
- Evidence: the two-lane hand oracle retains a non-trainable middle step that changes a later
  trainable advantage; policy source indices are exactly `[0,3,4]`. A separate mid-capture boundary
  uses a positive sign and a 0.75 bootstrap to produce value target 0.75. Fifty tests cover all 211
  statements and 96 branches in `logs/test-output/000074-phase6-buffer-quality-final.txt`.

## ADR-025 — Use the exact GroupNorm network and scope invariance numerically

- Status: Accepted; T1, T2, network portion of T5, and N2–N7 GREEN.
- Authority: `GOAL.md` §9; the project's GroupNorm choice is a correctness design for PPO ratios,
  while the supervised tests establish capacity rather than learning quality.
- Decision: implement the exact 8→64 stem, six two-convolution residual blocks, 2-channel policy
  head, and 1-channel bounded value head. Use GroupNorm(8,64) in the trunk, GroupNorm(1,2/1) in
  heads, orthogonal gains `sqrt(2)/0.01/1.0`, and zero biases. BatchNorm is structurally forbidden.
- Invariance scope: train/eval output for the identical tensor is bitwise equal. A sample evaluated
  in batch sizes 1 and 4 is checked at `atol=1e-5, rtol=1e-4` because convolution kernels may use
  different floating-point accumulation order even though GroupNorm has no batch statistics.
- Result: 470,410 parameters (1.794 MiB FP32); T1 accuracy 1.0; T2 MSE `4.69e-14`; every parameter
  graph-connected and every major module's aggregate gradient nonzero. The real N7 mid-sequence
  versus boundary pair yields different logits, resolving BLOCK-006.
- Evidence: `tests/rl/test_networks.py`, `logs/test-output/000080-phase6-networks-quality-final.txt`,
  and `000081-phase6-supervised-metrics.txt`.

## ADR-026 — Keep PPO loss literal and stored-mask only

- Status: Accepted; T3, T4, and T5 GREEN.
- Authority: original PPO paper for the clipped surrogate; `GOAL.md` §8 for the classified
  objective and defaults; official PyTorch gradient-clipping behavior.
- Decision: compute new log-probability and entropy only through `MaskedCategorical` with the mask
  persisted at collection. Normalize advantages per minibatch using population standard deviation
  plus `1e-8`; use unclipped MSE values; compute k3 KL and strict ratio-bound clip fraction; form
  `policy + vf_coef*value - ent_coef*entropy`; clip the global gradient norm before Adam steps.
- Oracle: for ratios `[1.1,0.9,1.3,0.7]`, the hand-frozen losses are policy `0.06898048`, value
  `0.35025398359296`, entropy `0.6677927263741105`, KL `0.02609025383118571`, clip fraction `0.5`,
  and total `0.23742954453273896`. Each matches within `1e-12`.
- Direction: repeated updates increase positive-advantage selected-action probability and decrease
  negative-advantage probability until ratios cross 1.2/0.8; clip fraction becomes one and both
  probabilities plateau.
- Evidence: 31 focused tests and 100% statement/branch coverage in
  `logs/test-output/000087-phase6-ppo-quality-final.txt`.

## ADR-027 — Scope deterministic reproduction to a fully recorded stack

- Status: Accepted; D1–D3 GREEN.
- Authority: `GOAL.md` §12.7 and official PyTorch deterministic-algorithm documentation.
- Decision: derive independent Python, NumPy, Torch, CUDA, and per-environment streams from one
  unsigned 64-bit root with SplitMix64; enable deterministic Torch algorithms and deterministic
  cuDNN in test/smoke mode; rebuild the complete network/optimizer/data fixture before each
  reproduction run. Require exact CPU tuple equality, but only identical actions and
  `atol=1e-5, rtol=1e-4` loss agreement on the same GPU/software stack.
- Claim boundary: no bitwise GPU, cross-machine, cross-driver, or cross-library claim is made.
  Distinct deterministic sub-seeds are not claimed to establish statistical independence.
- Result: two ten-update CPU traces are bitwise identical; two ten-update RTX 5070 traces have
  identical actions and satisfy the declared tolerance. Native CUDA BF16 masked sampling is legal
  and finite, with exact-zero illegal gradients.
- Evidence: `tests/rl/test_determinism.py`, `logs/gates/phase-6-determinism.txt`, and
  `logs/test-output/000091-phase6-determinism-quality-final.txt`.

## ADR-028 — Make forced-win targets literal actor-relative outcomes

- Status: Accepted; T7 GREEN.
- Authority: `GOAL.md` §§6.4, 7.1–7.3, and 12.6 T7; production WCDF rules engine for trajectory
  legality and terminal classification.
- Decision: use deterministic-search-discovered boundary states only as frozen test inputs. During
  every test, require that the production rules engine returns exactly the one expected ACF step;
  take actors, rewards, continuation signs, and terminal outcomes through `CheckersEnv`. With zero
  values and `gamma=lambda=1`, freeze the independently hand-derived target as the final winner's
  `+1/-1` outcome in each transition actor's frame.
- Coverage: exact path lengths are 3, 5, and 7. The length-3 path ends by R6.2 no legal move; the
  distinct length-5 path terminates on the fourth jump by the same actor. Thus one fixture cannot
  accidentally satisfy both special-case obligations.
- Sensitivity: temporarily omitting the recursive `sigma` produces all-positive targets and fails
  all three numerical fixtures. The tracked source is restored byte-for-byte afterward.
- Evidence: `tests/rl/test_forced_mate_targets.py` and `logs/test-output/000096-*`/`000097-*`.
