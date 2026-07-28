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
