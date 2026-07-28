# GOAL.md — Autonomous Build Plan: American Checkers Agent via PPO Self-Play
### Version 2.0 — supersedes v1.0 in full. Incorporates the findings of the v1.0 audit.

> **File contract:** This file is the single source of truth and is **READ-ONLY** to the agent.
> The agent may never edit, reformat, or "clean up" this file. All agent-authored state goes in
> `STATE.json`, `PROGRESS.md`, `DECISIONS.md`, `BLOCKERS.md`, `logs/`, and `docs/`.
>
> **Escape hatch (new in v2.0):** a read-only spec is a hazard if the spec is wrong. If the agent
> finds a defect *in this document* — a rule that contradicts the cited source, an equation that is
> wrong, a gate that is impossible — it **must not** silently comply, and must not edit this file.
> It writes the defect to `BLOCKERS.md` using the template in §0.4, halts the affected phase, and
> continues on unaffected phases. A human resolves it. **Correct-per-spec is not the goal;
> correct-per-checkers and correct-per-mathematics is the goal.**

---

## CHANGELOG v1.0 → v2.0 (what was wrong, and where it is fixed)

| # | v1.0 defect | Fix |
|---|---|---|
| 1 | Captured pieces removed immediately (contradicts WCDF: removal at end of sequence) | §4 R4.5, §5.1 |
| 2 | Multi-jump continuation state absent → non-Markov, observation aliasing | §5.1, §6.2 |
| 3 | Zobrist key omitted continuation state → semantic collisions by construction | §5.3 |
| 4 | Repetition counted on intermediate jump substates | §5.3, R6.4 |
| 5 | "repetition count" plane cannot make repetition Markov | R6.4 — removed from training env |
| 6 | 40-move rule presented as faithful WCDF | R6.3 — corrected + labelled ENGINE VARIANT |
| 7 | Threefold rule presented as faithful WCDF | R6.4 — labelled ENGINE VARIANT |
| 8 | GAE recursion lacked the perspective sign σ | §7.3 |
| 9 | Filtering opponent transitions breaks GAE adjacency | §7.4 — full chronological buffer |
| 10 | "both players' segments" / "episode segment" undefined | §7.1 — every term defined once |
| 11 | BatchNorm mandated, then admitted hazardous | §9 N3 — GroupNorm baseline |
| 12 | `-1e8` asserted safer than `-inf` | §6.5 — dtype-aware `finfo.min` |
| 13 | `illegal_action_attempts` measures nothing real | §13.2 — renamed mask diagnostics |
| 14 | Entropy ratio used mean-of-H over log-of-mean-k (Jensen error) | §13.2 |
| 15 | Exhaustive test aimed at a trivial geometric fact | §12.2 item 3 |
| 16 | `[HUANG37]` treated as binding law in full | §3.3, §8 — three authority tiers |
| 17 | Project guesses presented as source-derived | §8.3 — labelled HYPOTHESIS |
| 18 | γ=1.0 justified by citation | §8.3 — declared an objective choice, with its own argument |
| 19 | AlphaStar cited to mandate a 60/40 league | §10.2 — league is an experiment arm |
| 20 | "Strong amateur level" unoperationalized | §1.2, §11.1 |
| 21 | "Number of kings never decreases" — false invariant | §12.3 |
| 22 | Piece-count invariant unqualified | §12.3 |
| 23 | Fuzz scale contradictory (10^6 steps vs 10^6 games) | §12.5 — tiered |
| 24 | Differential agreement treated as correctness proof | §12.4 — external anchors + mutation testing |
| 25 | Self-generated perft blurred with ground truth | §12.4 E |
| 26 | "Every parameter gets nonzero gradient" overstrict | §12.6 T5 |
| 27 | "PPO overfits a batch to ~zero loss" conceptually muddled | §12.6 T1–T4 |
| 28 | Bitwise GPU determinism unachievable across stacks | §12.7 |
| 29 | Checkpoint contents incomplete for self-play resume | §12.8 |
| 30 | Baseline gate assumed deeper search is monotonically stronger | §11.3 |
| 31 | Single scalar Elo in a possibly non-transitive population | §11.4 |
| 32 | "Monotone-ish Elo" unfalsifiable | §11.5 |
| 33 | "Held-out" suite committed in the repo | §11.6 — sealed evaluation |
| 34 | minimax(4)/(6) built in Phase 4 = procedural holdout only | §11.6 |
| 35 | Acceptance made non-binding by an "or diagnose it" branch | §19 — three separate tiers |
| 36 | Commit-every-iteration forces permanently red commits | §0.2 |
| 37 | Append-only PROGRESS.md re-read each loop → context blowup | §0.3, §15 |
| 38 | "Paste actual command output" unbounded | §0.3, §15.2 |
| 39 | Frozen config vs mutable schedule state conflated | §8.4 |
| 40 | black + ruff double-mandated | §12.9 — ruff authoritative |
| 41 | Offline-reproduction wording ambiguous | §12.10 |
| 42 | Mandatory ablations could explode the compute budget | §8.5 — ablation ladder |

---

## 0. HARNESS LOOP PROTOCOL

### 0.1 Per-iteration prompt (re-inject verbatim each loop)

```
You are an autonomous engineering agent building a correct, well-tested American Checkers
reinforcement-learning system. Your spec is ./GOAL.md, which is READ-ONLY.

CONTEXT BUDGET: each iteration read GOAL.md, STATE.json, PROGRESS.md (capped at 400 lines),
BLOCKERS.md, and at most 2 files from logs/. Never read logs/ wholesale. For older context read
logs/SUMMARY.md.

LOOP PROCEDURE — in order, every iteration:

1.  READ   GOAL.md sections relevant to the current phase, STATE.json, PROGRESS.md, BLOCKERS.md.
2.  ORIENT Take the lowest-numbered phase in §14 that is not GREEN. Work only on that phase.
           If it is BLOCKED, take the next non-blocked phase and say so.
3.  PLAN   Write 1-3 atomic tasks (each <= ~300 lines of diff) into PROGRESS.md "## In Flight".
4.  RED    Write the failing test(s) first. Run them. Confirm they fail for the RIGHT reason.
           Full output -> logs/test-output/<iter>-red.txt. Only a 5-line summary in PROGRESS.md.
5.  GREEN  Write the minimum code that passes. No speculative generality.
6.  VERIFY Run `make check`. If red, fix now. Do not proceed. Do not commit.
7.  GATE   If §14's exit criteria for this phase are met, run the gate command, save full output
           to logs/gates/phase-N.txt, record summary + path + exit code in PROGRESS.md, set the
           phase GREEN in STATE.json.
8.  LOG    Append logs/iterations/<NNNNNN>.md. REWRITE (do not append) PROGRESS.md, keeping it
           under 400 lines: current phase, in-flight tasks, last 5 iteration one-liners, open
           risks, next step. Every 25 iterations regenerate logs/SUMMARY.md as a compaction.
9.  COMMIT Only from a green tree. Conventional commit. Never --no-verify. If the iteration ends
           mid red-green, do NOT commit: record your position in PROGRESS.md and resume next time.
10. STOP   If §19 Tier 1 and Tier 2 are green, print "ACCEPTANCE MET" and halt.
           If BLOCKERS.md has an unresolved P0, print "BLOCKED <id>" and halt.

HARD RULES: obey §2 (DO NOTs) and §8 (LAW) absolutely. If a GOAL.md rule appears to be WRONG (not
merely inconvenient), file it in BLOCKERS.md per §0.4 and halt that phase. Never relax a rule to
make progress. Correctness is the objective; speed is not.
```

### 0.2 Commit policy (revised)
Commit **only from a green tree.** An iteration ending inside a red-green cycle does not commit; it
records its position and resumes. Failing-test evidence lives in `logs/test-output/`, not as
permanent red commits. Every bug still gets a permanent regression test — committed together with
its fix, in one green commit.

### 0.3 State files and size limits

| File | Rule | Cap |
|---|---|---|
| `STATE.json` | machine-readable gates, iteration, git SHA | 5 KB |
| `PROGRESS.md` | **rewritten, not appended**, every iteration | **400 lines** |
| `BLOCKERS.md` | open spec defects / external blockers | 200 lines |
| `DECISIONS.md` | ADR log, ~15 lines per entry | 2000 lines |
| `logs/iterations/NNNNNN.md` | immutable per-iteration record | 100 lines each |
| `logs/SUMMARY.md` | regenerated compaction every 25 iterations | 300 lines |
| `logs/test-output/*.txt`, `logs/gates/*.txt` | raw output, **never read wholesale** | unbounded on disk |

**Never paste full test, coverage, or fuzz output into a file read every loop.** Record: command,
exit code, counts (`412 passed, 0 failed, cov 97.8%`), the log path, and — only when red — a
≤30-line failure excerpt.

### 0.4 Blocker template
```markdown
### BLOCK-007 [P0|P1] §<goal-section> — <one line>
Claim in GOAL.md: <quote>
Why it is wrong: <argument, with source or derivation>
Evidence: <test name / file path / citation>
Phases affected: <list>
Proposed correction: <concrete replacement text>
Status: OPEN
```

---

## 1. MISSION AND OBJECTIVE

### 1.1 What is being built
A from-scratch, exhaustively tested American Checkers (English Draughts) engine, a Gymnasium-style
self-play environment, and a PPO training system that learns to play it, fully instrumented in
Weights & Biases.

### 1.2 What success means (operationalized — no unanchored skill claims)
v1.0 said "strong amateur level." That is unmeasurable without an external rating anchor, so it is
**withdrawn**. Success is three separated tiers, defined in §19:

- **Tier 1 — Engineering acceptance (binding).** The engine is provably correct, the tests are
  rigorous and honest, training runs, reproduces, and is fully instrumented.
- **Tier 2 — Learning acceptance (binding).** The agent demonstrably learns: it beats the fixed dev
  baselines by stated margins with confidence intervals, on ≥3 seeds.
- **Tier 3 — Strength (non-binding, reported either way).** Performance against a sealed suite and,
  if available offline, an external reference engine.

**The primary objective is correctness.** A slow, provably-correct system that reaches only Tier 2
is a PASS. A fast system with a subtly wrong jump generator that looks strong is a total FAIL. Any
strength number reported without the correctness evidence behind it is meaningless.

---

## 2. STRICT DO NOTs

### 2.1 Do not cheat the tests
- **DO NOT** delete, skip, `xfail`, comment out, or weaken any test to make a suite pass. Tests are
  ratchets: they only get stricter.
- **DO NOT** lower a coverage threshold, mutation-score threshold, lint severity, mypy strictness,
  or numerical tolerance to turn a gate green.
- **DO NOT** modify a file or test marked `# FROZEN` except to add to it.
- **DO NOT** write a test that treats the implementation's own output as ground truth. Ground truth
  comes from §4, hand computation, an independent oracle, an external source, or a published
  transcript. Self-generated expectations may exist **only** as labelled *regression baselines*,
  never as *correctness evidence* (§12.4).
- **DO NOT** swallow an exception a test would otherwise catch.
- **DO NOT** commit with `--no-verify`.

### 2.2 Do not cheat the game
- **DO NOT** alter the rules of American Checkers to make learning easier.
- **DO NOT** implement International / Brazilian / Italian / Russian / Turkish draughts rules by
  accident. Every rule is checked against §4 and its ID appears in the test name.
- **DO NOT** import, vendor, copy, or transliterate an existing checkers engine, move generator,
  opening book, or endgame database. Reading published rules text is required; copying code is not.
- **DO NOT** place an opening book, tablebase, or hand-coded heuristic anywhere in the learning
  agent's decision path. Heuristics exist only as fixed evaluation opponents (§11.2).
- **DO NOT** describe any engine-specific draw rule as "the WCDF rule." R6.3, R6.4, and R6.5 are
  declared ENGINE VARIANTS and must be labelled as such in code, docs, and README.

### 2.3 Do not cheat the learning
- **DO NOT** use reward shaping. Rewards are terminal-only, defined in §7.1 against explicit player
  IDs. No material bonuses, no promotion bonuses, no per-step penalties.
- **DO NOT** tune against, select checkpoints against, or inspect the **sealed suite** (§11.6).
- **DO NOT** report a learning claim from a single seed. ≥3 seeds, mean ± CI, always.
- **DO NOT** `nan_to_num`, clamp, or otherwise hide a NaN/Inf in a loss. Raise, dump the batch to a
  W&B artifact, diagnose.
- **DO NOT** let a sampled action be illegal. `mask/sample_legality_violations` must be exactly 0.
- **DO NOT** change a hyperparameter mid-run and report one curve. New config → new run.

### 2.4 Do not cheat the engineering
- **DO NOT** leave `TODO`, `FIXME`, stub `pass`, or `NotImplementedError` inside a GREEN phase.
- **DO NOT** use `# type: ignore` or `Any` without a reason comment and a `DECISIONS.md` entry.
- **DO NOT** introduce global mutable state, hidden singletons, or import-time side effects.
- **DO NOT** use `sleep`, retries, or ordering luck to pass a flaky test. Flaky = bug.
- **DO NOT** work in a phase you are not on.
- **DO NOT** claim a gate passed without the log path and exit code recorded.
- **DO NOT** paste raw multi-thousand-line output into any file read each loop (§0.3).
- **DO NOT** edit `GOAL.md`. File a blocker instead (§0.4).

### 2.5 Do not cheat the report
- **DO NOT** call untested code "working," "verified," or "correct."
- **DO NOT** summarize a failing run as "mostly passing."
- **DO NOT** delete or hide W&B runs, including failures. Tag them `failed`.
- **DO NOT** present differential-test agreement as a correctness proof (§12.4).

---

## 3. REFERENCES, WITH THEIR ACTUAL AUTHORITY

Each reference carries a tier. The agent cites the tag **and** the tier in `DECISIONS.md`.

- **Tier A — Definitional.** True by definition of the object being built.
- **Tier B — Empirically supported default.** Good evidence; deviation needs justification.
- **Tier C — Project hypothesis.** Our guess; must be labelled as ours.

### 3.1 Game rules — `[WCDF]` — Tier A
World Checkers Draughts Federation / American Checker Federation official rules for English
Draughts, including ACF 1–32 dark-square numbering.
**Mandatory verification:** before Phase 1 exits, the agent obtains the primary rules text and
produces `docs/RULES.md` giving, for every rule ID in §4: the rule text, the source clause it
derives from — or the label **ENGINE VARIANT** where §4 deliberately departs — and the covering
tests. If the primary text is unavailable offline, every rule is marked `UNVERIFIED`, and that fact
appears in the README and as a P1 in `BLOCKERS.md`.

### 3.2 PPO and GAE — `[SCHULMAN17]`, `[GAE]`, `[KL3]`
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* (arXiv:1707.06347). Defines the
  clipped surrogate. **Tier A.**
- Schulman et al. (2015), *GAE* (arXiv:1506.02438). Defines GAE(λ). **Tier A** for the recursion.
  The two-player perspective extension in §7.3 is **our derivation**; it earns Tier A status only
  by passing the proofs-in-tests of §12.6, not by citation.
- Schulman, *Approximating KL Divergence* — k3 estimator for the logged `approx_kl`. **Tier B**
  (a diagnostic choice).

### 3.3 PPO implementation details — `[HUANG37]` — **Tier B, not binding law**
Huang, Dossa, Raffin, Kanervisto, Wang (2022), *The 37 Implementation Details of Proximal Policy
Optimization*. It is an implementation-analysis article spanning core PPO2 details, environment-
specific details, LSTM details, continuous-control details, and *auxiliary/optional* techniques.
v1.0 wrongly called it binding. It is a **checklist to adjudicate, not a law to obey.** Notably, KL
early stopping is described there as auxiliary and off by default, so §8.2 lists it as a DEFAULT
with a diagnostic role rather than as LAW.
Deliverable `docs/PPO_CHECKLIST.md` adjudicates each applicable item as
`Implemented / Not applicable + reason / Deviated + reason + evidence stage`.

### 3.4 What matters empirically — `[ENGSTROM20]`, `[ANDRY21]` — Tier B
- Engstrom et al. (2020), *Implementation Matters in Deep Policy Gradients* → value clipping OFF by
  default.
- Andrychowicz et al. (2021), *What Matters in On-Policy RL?* → orthogonal init with small policy
  gain, Adam `eps` ≈ 1e-5, advantage normalization, gradient clipping.

### 3.5 Self-play — `[SILVER17]`, `[EXIT]`, `[LEAGUE]`
- Silver et al. (2017), AlphaZero — canonical orientation, shared trunk with policy and value
  heads, terminal-only reward. **Tier B.**
- Anthony, Tian, Barber (2017), Expert Iteration — relevant only to optional Phase 9.
- Vinyals et al. (2019), AlphaStar league. **Tier C as applied here.** That league used specialized
  exploiter agents and payoff-based matchmaking; it does not establish that any particular
  current/historical mixture is right for PPO checkers, and in some symmetric games current-policy
  self-play beats checkpoint mixtures at a fixed budget. Historical-opponent mixing is therefore an
  **experiment arm** (§10.2), not doctrine.

### 3.6 Engineering discipline — `[KARPATHY]`, `[ZINKEVICH]`, `[MLTS]`, `[SCULLEY]` — Tier B
Karpathy's *Recipe for Training Neural Networks* (the debugging ladder in §16 is mandatory);
Zinkevich's *Rules of Machine Learning* (infrastructure correctness before model cleverness);
Breck et al., *The ML Test Score* (self-scored in `docs/ML_TEST_SCORE.md`); Sculley et al.,
*Hidden Technical Debt in ML Systems* (one config object, no glue sprawl).

### 3.7 Theory — `[SB2]` — Tier A for MDP/semi-MDP formalism
Sutton & Barto, 2nd ed., Ch. 3, 9, 13. It does **not** mandate γ = 1.0; see §8.3.

### 3.8 Testing — `[PYTEST]`, `[HYPOTHESIS]`, `[MUTATION]`, `[BECK]` — Tier B
`pytest`, `pytest-cov`, `hypothesis`, `pytest-xdist`, and a mutation-testing tool (`mutmut` or
`cosmic-ray`), see §12.4. Beck, *TDD by Example*, for the red-green rhythm.

---

## 4. RULES SPECIFICATION — AMERICAN CHECKERS

Every rule ID must appear in ≥1 test name (e.g. `test_R4_5_marked_piece_still_blocks_landing`).
`docs/RULES.md` holds the traceability matrix and source clause for each ID.

### R1 — Board and setup
- **R1.1** 8×8 board; play on the 32 dark squares only.
- **R1.2** ACF 1–32 numbering. The exact orientation and mapping is fixed once, drawn as ASCII in
  `docs/RULES.md`, and thereafter FROZEN.
- **R1.3** Each player has a double corner on their right.
- **R1.4** Initial position: the first player occupies 1–12, the second 21–32, 13–20 empty; all 24
  pieces are men.
- **R1.5** The first player moves first. `[WCDF]` names the first player Red; computer literature
  usually says Black. Pick **one** convention, state it once in `docs/RULES.md`, and use it
  consistently in code, notation, logs, and W&B metric names. Player identity is an explicit
  `PlayerId` enum and is **never** inferred from the canonical observation (§7.1).

### R2 — Turn structure
- **R2.1** Players alternate turns.
- **R2.2** A *checkers move* (a full turn) is one simple move (R3) or one complete capture sequence
  (R4). The environment decomposes a capture sequence into multiple *environment steps* (§5.2). The
  distinction between a **checkers move** and an **environment step** is load-bearing and must be
  honored in code, counters, notation, and metrics.

### R3 — Simple moves
- **R3.1** A man moves one square diagonally forward to an adjacent empty dark square.
- **R3.2** A king moves one square diagonally, forward or backward, to an adjacent empty dark
  square.
- **R3.3** No flying kings; no moving onto an occupied square.

### R4 — Captures
- **R4.1** A jump moves a piece diagonally over an adjacent enemy piece to the immediately following
  square, which must be **empty** — where "empty" is evaluated under the occupancy rule R4.5.
- **R4.2** **Captures are mandatory.** If the side to move has any jump available, its legal move
  set contains only jumps. Evaluated **per player**, not per piece.
- **R4.3.1** A king may jump forward or backward.
- **R4.3.2** A man may jump **forward only**, never backward.
- **R4.4** **Continuation is mandatory.** After a jump, if the same piece has a further legal jump,
  it must make it; the checkers move does not end until that piece has no further legal jump,
  subject to R5.2.
- **R4.5 — DELAYED REMOVAL (corrected in v2.0).** Captured pieces **remain on their squares** for
  the duration of the capture sequence and are **marked as captured**. A marked piece:
  - may **not** be jumped again in the same sequence, and
  - **still occupies** its square, so it blocks landing squares and blocks any jump that would
    require that square to be empty.
  All marked pieces are removed simultaneously when the sequence ends.
  *v1.0 removed pieces immediately. Immediate removal can vacate a square that must remain blocked,
  admitting routes official play does not permit.* The state therefore carries a
  `captured_pending: uint32` bitmask (§5.1). R4.5 gets a dedicated test module of hand-constructed
  positions where the two semantics diverge; if no divergent position can be constructed, that must
  be **proved and recorded**, not assumed.
- **R4.6** **No majority-capture rule.** With several jumps or sequences available, the player
  chooses freely regardless of how many pieces each captures. Do not implement a maximum-capture
  rule; that belongs to International/Italian draughts.

### R5 — Promotion
- **R5.1** A man that **ends its checkers move** on the opponent's king row is promoted.
- **R5.2** Promotion ends the checkers move immediately. A man crowned by a jump may **not**
  continue jumping, even if a king would have a further jump. In the step-wise environment: after a
  jump landing on the king row, the mask contains no continuation actions, the sequence terminates,
  marked pieces are removed, and the side changes.
- **R5.3** A king is never demoted. (A king can of course be *captured* — see §12.3.)

### R6 — Terminal conditions
- **R6.1** A player with no pieces loses.
- **R6.2** A player with pieces but **no legal move** on their turn **loses.** Unlike chess,
  stalemate is a loss.
- **R6.3 — ENGINE VARIANT: automatic no-progress draw.**
  *Declared departure from `[WCDF]`.* The official rule is a **claim-and-demonstration** procedure
  in which a player shows that over their own previous 40 moves they have neither advanced an
  uncrowned man toward the king row nor had any piece removed. A self-play environment has no
  referee and no claimant, so this engine uses an automatic approximation:
  - Maintain **two per-player counters** `no_progress[p]`, incremented at the end of each
    **completed checkers move** by `p`, and reset to 0 at the end of that move if the move
    (a) captured at least one piece, **or** (b) moved an uncrowned man (all man moves advance
    toward the king row).
  - The game is a draw when **both** counters are ≥ 40 simultaneously.
  - Counters update **per completed checkers move only** — never per environment step, never per
    jump substep. A test asserts a multi-jump increments the counter exactly once.
  This per-player 40-move window replaces v1.0's global 80-ply counter, which was not logically
  equivalent to "each player's own previous 40 moves." It remains an approximation and must be
  labelled **ENGINE VARIANT** wherever described.
- **R6.4 — ENGINE VARIANT: repetition, and why it is OFF during training.**
  *Declared departure from `[WCDF]`.* The official rule is prospective and claim-based (a player
  demonstrates that their *next* move would produce the same position a third time). More
  importantly: **threefold repetition cannot be made Markov by any fixed-size observation.** Whether
  a move draws depends on the visit counts of *successor* positions, not only the current one — two
  games with identical boards and identical current-position counts can differ in whether an
  available move completes a repetition. v1.0's "repetition count / 2" plane does not fix this; it
  creates partial observability while appearing to solve it. Therefore:
  - **Training environment: repetition draws are DISABLED.** Termination is guaranteed by R6.3 and
    R6.5 alone (see R6.7).
  - **Arena/evaluation: an automatic threefold rule is available behind a flag**
    (`repetition_draws=True`), computed over the **official position key** (§5.3) and updated
    **only at completed-move boundaries**, never on intermediate capture substates.
  - Any evaluation using `repetition_draws=True` logs that fact, and the resulting train/eval rule
    mismatch is stated in the README and `docs/METRICS.md` as a known distribution shift.
- **R6.5 — Ply cap (a declared rule, not a safety net).** A game exceeding `max_plies` (default 512
  environment steps) is a **draw**. This is a genuine rule of the training MDP, part of the
  objective definition, visible in the observation (§6.2, plane 7). `env/ply_cap_draws` is a
  reported metric, not an alarm.
- **R6.6** No draw by agreement.
- **R6.7 — Termination proof obligation.** `docs/RULES.md` must contain a proof sketch that R6.3
  alone guarantees termination (man moves are monotone toward the king row and therefore a finite
  resource; captures are a finite resource; at most 40 moves per player elapse between resets),
  derive the worst-case bound, and state honestly that `max_plies = 512` is **below** that bound —
  so R6.5 is a genuine additional rule that shortens some games rather than a formality. Tests
  construct one game terminating by R6.3 and one by R6.5.

### R7 — Notation and serialization
- **R7.1** ACF notation: `11-15` simple, `22x18` jump, `22x18x9` multi-jump.
- **R7.2** Parse and format, with a round-trip property test.
- **R7.3** A FEN-like serializer for fixtures, artifacts, and transcript replay. It must serialize
  the **full environment state** (§5.1), **including mid-sequence states**, and round-trip exactly.

---

## 5. ENVIRONMENT STATE MODEL *(new section — v1.0's largest structural gap)*

### 5.1 The complete state
A state is **not** just placement plus side to move. A mid-capture-sequence state has different
legal actions than an otherwise-identical turn-boundary state, so all of the following are state:

```python
@dataclass(frozen=True, slots=True)
class State:
    men:          tuple[uint32, uint32]  # bitboards over 32 dark squares, indexed by PlayerId
    kings:        tuple[uint32, uint32]
    side_to_move: PlayerId
    # --- capture-sequence state (absent in v1.0; its absence broke the Markov property) ---
    capture_in_progress: bool            # a sequence has begun and not ended
    moving_square:    int | None         # square of the piece that must continue
    sequence_origin:  int | None         # where the sequence began (notation + undo)
    captured_pending: uint32             # enemy pieces marked captured, still occupying (R4.5)
    # --- counters (updated only at completed-move boundaries) ---
    no_progress: tuple[int, int]         # per player, R6.3
    ply: int                             # environment steps elapsed, R6.5
```
Invariants (tested): `capture_in_progress` ⇔ `moving_square is not None`; `captured_pending != 0`
implies `capture_in_progress`; `captured_pending` ⊆ the opponent's occupied squares; at every
completed-move boundary all sequence fields are cleared and `captured_pending == 0`.

### 5.2 Checkers move vs. environment step
- A **checkers move** is one legal turn under §4.
- An **environment step** is one `env.step()` call. A simple move is 1 step; a sequence of *k* jumps
  is *k* steps, all by the **same** player.
- R6.3 counters, notation, arena repetition, and "move completed" callbacks fire on **move**
  boundaries. `ply` (R6.5) counts **steps**. Every increment in the codebase must state its unit,
  and a test asserts each.

### 5.3 Two distinct hash keys *(v1.0 conflated them)*
1. **`state_key`** — Zobrist over everything in §5.1 that affects legal transitions: placement by
   type and colour, side to move, `capture_in_progress`, `moving_square`, `captured_pending`. Used
   for transposition, caching, dedup, and testing. Omitting the sequence fields, as v1.0 did,
   produces **semantic collisions by construction** — not merely probabilistic ones — because two
   states with identical placement can have different legal actions.
2. **`position_key`** — Zobrist over placement + side to move **only**, defined **only at completed
   move boundaries**, used solely for the optional arena repetition rule and opening-diversity
   statistics. Calling it on a mid-sequence state **raises**.

Both use FROZEN seeds committed to the repo. Incremental update must equal from-scratch
recomputation after every step and every undo (property test). A dedicated test asserts a
mid-sequence state and its "same placement, no sequence" counterpart have **different** `state_key`,
and that `position_key()` raises on the former.

---

## 6. ENVIRONMENT CONTRACT

### 6.1 API
```python
obs, info = env.reset(seed=None)            # info["legal_mask"]: np.ndarray[bool, (128,)]
obs, reward, terminated, truncated, info = env.step(action: int)
env.legal_mask() -> np.ndarray              # bool[128]
env.state_key() -> int                      # §5.3 (1)
env.position_key() -> int                   # §5.3 (2); raises mid-sequence
env.render("ansi") -> str                   # ASCII, ACF numbering, pending-capture marks
info = {"legal_mask", "actor": PlayerId, "move_completed": bool,
        "checkers_move_san": str | None, "outcome": Outcome | None}
```
- An illegal action raises `IllegalActionError`: never a no-op, never a resample, never a
  substitution.
- `terminated` is True only for R6.1–R6.5. `truncated` is reserved for rollout-boundary cutoffs
  imposed by the trainer and is **never** set by a game rule. R6.5 is a rule → `terminated`.

### 6.2 Observation — 8 planes, `(8, 8, 8)` float32, canonical to the side to move

| Plane | Contents |
|---|---|
| 0 | side-to-move's men |
| 1 | side-to-move's kings |
| 2 | opponent's men (**including** captured-pending pieces — they still occupy) |
| 3 | opponent's kings (including captured-pending) |
| 4 | opponent pieces marked **captured-pending** (subset of 2–3; may not be jumped again) |
| 5 | the **forced continuation piece** (one-hot on `moving_square`; zeros if no sequence) |
| 6 | `no_progress[side_to_move] / 40`, broadcast constant |
| 7 | `ply / max_plies`, broadcast constant |

The board is rotated 180° so the side to move always moves "up," hence no side plane
(`[SILVER17]`). A nonzero plane 5 encodes `capture_in_progress`, so no separate flag plane is
needed; this redundancy decision goes in `DECISIONS.md`. **Planes 4 and 5 are precisely the fix for
the v1.0 aliasing defect:** without them, an ordinary turn and a forced continuation with identical
placement produce identical observations while having different legal actions. Light squares are
always zero in planes 0–4 (tested). Plane 6 carries the *acting* player's counter; because the
opponent's counter also determines when R6.3 fires, a mandatory Stage-B ablation (§8.5) adds a 9th
plane with `no_progress[opponent] / 40`, and if it matters it becomes the default.

### 6.3 Action space
`Discrete(128)` = 32 dark squares × 4 diagonal directions in the **canonical (rotated) frame**;
`action = square_index * 4 + direction_index`. One action = one environment step = one simple move
or one jump of a sequence.

**Design rationale, with its cost stated (`DECISIONS.md`):** a `(from_square, direction)` pair
determines a single step uniquely, because jumps are short-range and the captured square is fixed by
geometry. This keeps the action space small and fixed. The cost is that a turn spans multiple
timesteps, which forces the perspective machinery of §7.3 and the extra state of §5.1. The
alternative — one action per complete legal move sequence — removes intermediate states but needs a
variable, position-dependent index map. Step-wise is chosen; full-sequence is the recorded fallback
if §7.3's tests cannot be made to pass.

### 6.4 Rewards
See §7.1. The environment emits reward only on the transition into a terminal state, from the
perspective of the **actor of that transition** (`info["actor"]`).

### 6.5 Masking *(dtype-aware — v1.0's `-1e8` claim was wrong)*
- **A1** `legal_mask()` returns bool[128] with `mask.any() == True` at every non-terminal state (a
  player with no legal move makes the state terminal by R6.2).
- **A2** Before constructing the distribution: `assert mask.any(dim=-1).all()`. An all-masked row is
  the genuinely catastrophic case and must raise, not be papered over.
- **A3** Masked logits are set to `torch.finfo(logits.dtype).min` — **not** a hardcoded `-1e8`
  (which overflows to `-inf` in fp16) and not an unconsidered `-inf`. A single tested
  `MaskedCategorical` is used everywhere. Its `sample`, `log_prob`, and `entropy` are unit-tested
  under **float32, bfloat16, and any autocast mode actually used**, including the `k = 1` case.
- **A4** Masking is in the gradient path: illegal logits are masked *before* the softmax and receive
  exactly zero gradient. A post-hoc filter on sampled actions is forbidden.
- **A5** The update uses the mask **stored in the buffer for that state**, never one recomputed
  from a mutated environment.
- **A6** Entropy and log-probs come from the masked distribution.
- Reference: Huang & Ontañón, *A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms* — Tier B.

---

## 7. RL FORMALISM *(normative — implement literally)*

### 7.1 Definitions — each term defined exactly once
v1.0's undefined phrase "episode segment" is **deleted**. The only terms are:

- **`actor(t)`** — the `PlayerId` choosing the action at step *t*. Stored explicitly per
  transition; **never** inferred from the canonical observation.
- **`z ∈ {+1, 0, −1}`** — the **game outcome label**, fixed at game end, expressed for a reference
  player.
- **`r_t`** — the **stored transition reward**. `0` for every non-terminal transition. For the
  transition that terminates the game, `r_t` is the outcome from the perspective of `actor(t)`:
  `+1` if `actor(t)` won, `−1` if it lost, `0` for a draw. (R6.2 means a player can lose on the
  *opponent's* move; this convention handles that correctly because it is always relative to the
  actor of the terminating transition.)
- **`V(s)`** — the value head: expected outcome **from the perspective of the player to move at
  `s`**, range `[−1, +1]`.
- **`σ_t ∈ {+1, −1}`** — the **perspective sign**: `+1` if `actor(t+1) == actor(t)` (a multi-jump
  continuation), `−1` otherwise.
- **`d_t ∈ {0, 1}`** — 1 if `s_{t+1}` is terminal by R6.1–R6.5.
- **truncation** — a rollout-boundary cutoff imposed by the trainer. Not a game rule, never sets
  `d_t`, always bootstrapped from.

### 7.2 Value targets
Two consistent formulations exist; **this project uses the GAE/TD formulation of §7.3**, not
retrospective outcome labelling. The retrospective label `z_t` (outcome from the perspective of the
player to move at `s_t`) is still computed and stored, for: (a) the value-calibration metric,
(b) the supervised value-head test (§12.6 T2), and (c) an optional MC-vs-GAE ablation. It is not the
default training target.

### 7.3 The two-player GAE recursion *(v1.0's central omission)*

```
δ_t = r_t + γ · (1 − d_t) · σ_t · V(s_{t+1}) − V(s_t)

A_t = δ_t + γ · λ · (1 − d_t) · σ_t · A_{t+1}
```

- At a terminal step (`d_t = 1`) both bootstrap terms vanish: `δ_t = r_t − V(s_t)`.
- At a rollout truncation boundary `d_t = 0`, `V(s_{t+1})` is the stored bootstrap value with `σ_t`
  applied, and `A_{t+1} = 0` at the very end of the buffer.
- **`σ_t` appears in both equations.** v1.0 negated only the bootstrapped value and not the
  recursively propagated advantage — that is the bug this fixes. `A_{t+1}` is expressed in
  `actor(t+1)`'s frame and must be transformed into `actor(t)`'s frame.
- `returns_t = A_t + V(s_t)`. Never mix an MC return target with a GAE advantage.

**This is the highest-risk mathematics in the project.** It is verified by: hand-computed
trajectories (T3), a σ-degenerate equivalence test against a single-agent reference (T6), forced-mate
value targets (T7), and a colour-swap symmetry test that negates all advantages.

### 7.4 Rollout buffer — full chronological (Design A)
Filtering out frozen-opponent transitions and then running GAE over the remainder is **wrong**: the
retained records are no longer adjacent one-step transitions. Therefore:

- The buffer stores **every** transition in chronological order with fields:
  `obs, legal_mask, action, behaviour_logprob, value, reward, done, actor, sigma, trainable,
  policy_id, env_id, move_completed`.
- **Returns and advantages are computed over the full chronology**, so bootstrapping always spans
  genuine adjacent steps.
- **The policy loss touches only `trainable=True` transitions.** Opponent-sampled actions were not
  drawn from `π_θ_old`; their importance ratios are meaningless.
- **The value loss is by default also restricted to `trainable=True` states**, with an ablation
  including opponent states recorded in `DECISIONS.md`.
- `V(s)` at opponent states is still computed and used for bootstrapping — required and correct.
- A test asserts batch composition: no non-trainable index reaches the policy loss, and GAE inputs
  are time-contiguous per `env_id`.

### 7.5 Vectorization
`num_envs` games run in lockstep by **environment step**, not by checkers move, so a rollout slice
can cut through the middle of a capture sequence. That is legal and handled by §7.3's truncation
bootstrapping — but it requires the environment to be **serializable mid-sequence** (§5.1, R7.3),
and a test must checkpoint and restore mid-sequence.

---

## 8. PPO RULES, CLASSIFIED BY AUTHORITY

v1.0 called all of this "law." It is now tiered. **LAW** may not change. **DEFAULT** may change with
evidence via §8.5. **HYPOTHESIS** is our guess and may be re-tuned cheaply.

### 8.1 LAW (algorithm-defining)
- **L1** PPO-Clip objective per `[SCHULMAN17]`: `L = −E[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]`,
  `r_t = π_θ(a_t|s_t)/π_θ_old(a_t|s_t)`.
- **L2** Advantages use GAE(λ) with the **two-player recursion of §7.3**, with correct terminal vs.
  truncation handling.
- **L3** `returns = advantages + values`.
- **L4** Masking per §6.5: in the gradient path, using stored masks.
- **L5** On-policy only: each rollout is used for exactly `update_epochs` passes, then discarded.
  No replay buffer.
- **L6** The policy loss touches only `trainable` transitions (§7.4).
- **L7** Total loss = `policy_loss + vf_coef·value_loss − ent_coef·entropy`.
- **L8** Global gradient-norm clipping before the optimizer step.
- **L9** No reward shaping (§2.3).

### 8.2 DEFAULT (empirically supported; deviation needs §8.5 evidence)
| Setting | Default | Basis |
|---|---|---|
| advantage normalization | per minibatch, `eps=1e-8` | `[ANDRY21]`, `[HUANG37]` — Tier B |
| value loss | plain MSE, **clipping OFF** | `[ENGSTROM20]` — Tier B |
| weight init | orthogonal; trunk `√2`, policy head `0.01`, value head `1.0`; biases 0 | `[ANDRY21]` — Tier B |
| optimizer | Adam, `eps = 1e-5` | `[ANDRY21]` — Tier B |
| LR schedule | linear anneal to 0 | `[HUANG37]` — Tier B |
| `clip_coef` | 0.2 | `[SCHULMAN17]` — Tier B |
| `gae_lambda` | 0.95 | `[GAE]` — Tier B |
| `vf_coef` | 0.5 | `[HUANG37]` — Tier B |
| `max_grad_norm` | 0.5 | `[ANDRY21]` — Tier B |
| KL early stop | ON, `target_kl = 0.02`, k3 estimator | **auxiliary** in `[HUANG37]`; kept as a safety diagnostic, logged as `train/kl_early_stops` |

### 8.3 HYPOTHESIS (ours — labelled as ours everywhere)
| Setting | Value | Our reasoning |
|---|---|---|
| `gamma` | **1.0** | An **objective definition**, not a citation. README must state the consequences: no preference for faster wins or slower losses, higher variance, every terminal outcome propagating undiscounted. The strongest argument for it here is structural: since a capture sequence is decomposed into a variable number of environment steps, any γ<1 systematically penalizes long capture sequences — an artifact of our action encoding, not of checkers. γ<1 is a legitimate ablation but must correct for step-count bias. |
| `total_timesteps` | 50M (configurable) | compute budget |
| `num_envs` / `num_steps` | 64 / 128 → batch 8192 | throughput guess |
| `num_minibatches` / `update_epochs` | 8 / 4 | common PPO2-family values |
| `learning_rate` | 3e-4 | common value |
| `ent_coef` | 0.01 → 0.001 linear over the first 50% | exploration then commitment |
| league mixture | §10.2 — an **experiment**, not a default |
| `max_plies` | 512 | bounds games; a declared rule (R6.5) |
| seeds | `{0,1,2}` minimum for any claim | §2.3 |

### 8.4 Config vs. trainer state *(v1.0 conflated them)*
Three objects, never merged:
1. **`RunConfig`** — `@dataclass(frozen=True)`, fully typed, `validate()` raises on nonsense
   (`batch_size % num_minibatches != 0`, `num_envs < 1`, `clip_coef <= 0`, `max_plies < 1`, …).
   Logged verbatim to W&B. **Never mutated.**
2. **`TrainerState`** — mutable: `global_step`, `update_idx`, RNG states, league pool, schedule
   phase, AMP scaler state.
3. **Derived schedule values** — pure functions `current_lr(cfg, state)`, `current_ent_coef(cfg,
   state)`. The current LR is never written back into the config; it is read from the optimizer and
   logged.

### 8.5 Ablation budget ladder *(prevents the explosion v1.0 implied)*
Not every question earns a 3-seed, 50M-step run. `DECISIONS.md` records which stage each conclusion
rests on.

| Stage | Budget | Purpose | Conclusive? |
|---|---|---|---|
| **A — Micro** | 1 seed, 2M steps | screen out obviously bad options | No — screening only |
| **B — Mid** | 3 seeds, 10M steps | measure effect size + CI | Yes, for DEFAULT/HYPOTHESIS changes |
| **C — Full** | 3 seeds, full budget | the headline baseline and ≤3 promoted variants | Required for the reported result |

**Mandatory Stage-B ablations (and only these):** shared vs. separate trunk; value loss on opponent
states; γ = 0.99 with step-count-bias correction; the league arms of §10.2; the 9th observation
plane of §6.2. Everything else is optional and must justify its compute in `DECISIONS.md` *before*
being run.

---

## 9. NETWORK

- **N1** Input `(8, 8, 8)` per §6.2.
- **N2** Trunk: `conv3x3(8→64) → GroupNorm(8, 64) → ReLU`, then 6 residual blocks
  `[conv3x3 → GN → ReLU → conv3x3 → GN → (+skip) → ReLU]`.
- **N3 — No BatchNorm (corrected in v2.0).** v1.0 mandated BatchNorm, admitted it was hazardous, and
  offered "use the same mode in rollout and update and prove equivalence" as an escape. That escape
  does not work: in training mode a sample's normalized activations depend on the other samples in
  its batch, and rollout batches (`num_envs` correlated states) and update minibatches (shuffled,
  size 1024) have structurally different composition — so identical parameters can produce different
  logits for the same state, silently corrupting the importance ratio `r_t`. In eval mode, running
  statistics lag a shifting self-play distribution. **GroupNorm is the baseline.** Removing
  BatchNorm is a correctness decision and needs no ablation. Any proposal to reintroduce it must
  first ship a passing test that rollout-time and update-time logits are bitwise equal for the same
  parameters and states.
- **N4** Policy head: `conv1x1(64→2) → GN → ReLU → flatten → Linear(→128)`.
- **N5** Value head: `conv1x1(64→1) → GN → ReLU → flatten → Linear(→64) → ReLU → Linear(→1) → tanh`.
- **N6** Shared trunk is permitted (`[SILVER17]`); shared vs. separate is a mandatory Stage-B
  ablation.
- **N7 — Aliasing regression test.** Two states differing only in `captured_pending` or
  `moving_square` must produce different logits. This is a direct regression test for the v1.0
  observation-aliasing defect.

---

## 10. SELF-PLAY REGIME

- **S1** One shared policy plays both colours; starting-colour assignment is balanced and logged
  (`env/first_player_frac ≈ 0.5`).
- **S2** Training samples from the masked categorical (temperature 1.0). Evaluation reports **both**
  greedy-argmax and sampled play.
- **S3** Only the current policy's transitions produce gradients (§7.4) — tested.
- **S4** Draws are expected between equal agents in checkers. A high draw rate is not itself a
  failure; a 100% draw rate with a flat value function is. Diagnostics: `env/draw_rate`,
  `env/mean_game_len_moves`, `env/captures_per_game`, `env/promotion_rate`, `env/no_progress_draws`,
  `env/ply_cap_draws`. **Do not add reward terms to move these numbers** (§2.3) — they are
  instruments, not targets.

### 10.2 Opponent selection is an EXPERIMENT *(corrected in v2.0)*
v1.0 cited AlphaStar to mandate a 60/40 current/historical mixture with a FIFO pool of 20. That
league was far more elaborate (specialized exploiters, payoff-based matchmaking) and does not
establish that any particular mixture suits PPO checkers; in some symmetric games, current-policy
self-play beats checkpoint mixtures at a fixed budget. So it is measured, not assumed.

**Stage-B arms, 3 seeds each:**
- **A0** current-policy self-play only (the honest simplest baseline)
- **A1** 80% current / 20% uniform from a historical pool
- **A2** 60% current / 40% uniform from a historical pool
- **A3** payoff-weighted historical sampling (prefer snapshots that beat the current policy)

Pool: snapshot every `snapshot_every` updates, capacity 20, FIFO, with the initial random policy
pinned. Selection uses the **population metrics of §11.4**, not scalar Elo alone. The winning arm
becomes the Stage-C default, and the result — including "A0 won" — is reported honestly.

---

## 11. EVALUATION SCIENCE

### 11.1 No unanchored strength claims
Never write "amateur," "intermediate," or "expert." Report performance **relative to named,
versioned opponents**, always with game counts and 95% CIs. Score = `(wins + 0.5·draws)/games` over
colour-balanced matches. If an external reference engine is available offline and its licence permits
use as an opponent (no code copied — §2.2), add it as an anchor and name its version and settings;
otherwise state plainly that no external anchor exists and all numbers are internal.

### 11.2 Opponent suites
- **Dev suite (may guide development and model selection):** `random_agent`, `greedy_agent`
  (max immediate material, seeded tie-break), `minimax(2)` with a simple material+king evaluator,
  and the self-play population metrics.
- **Sealed suite (§11.6):** deeper minimax, a tactical position set, and any external anchor.

### 11.3 Baseline ordering gate *(corrected — search is not monotone)*
v1.0 required `minimax(4) > minimax(2) > greedy > random`, each significant over 400 games. Deeper
search with a crude non-quiescent evaluator is **not** guaranteed stronger — horizon effects and
search pathology are real, and 400 games may be underpowered when draws are common. The corrected
gate requires:
- point estimates with 95% CIs over a game count justified by an explicit **power calculation** for
  the smallest effect worth detecting (compute it; do not guess 400);
- **no catastrophic inversion** (no deeper agent scores below 0.40 against a shallower one);
- a tactical check that deeper search solves a superset, or substantially more, of what shallower
  search solves;
- **non-monotonicity is reported and diagnosed as a property of the evaluator, not automatically
  treated as an engine bug.** If depth 6 is not stronger than depth 4, that is a finding about the
  heuristic; the suite is adjusted (e.g. add quiescence) and the change recorded.

### 11.4 Population metrics *(a single Elo is not enough)*
Self-play populations can be non-transitive (A beats B beats C beats A) — exactly the cycling this
design worries about — so scalar Elo cannot be the sole selection metric. Log and report:
- the full **payoff matrix** against the last *K* snapshots (win/draw/loss, not just score);
- **league Elo with a CI**, explicitly labelled valid only under an approximate-transitivity
  assumption that is itself checked;
- a **non-transitivity indicator** (count and magnitude of 3-cycles in the payoff matrix);
- an **exploitability proxy**: train a short-budget best-response agent against a frozen checkpoint
  and report the best response's score;
- scores against the **fixed dev anchors**, which are stationary by construction and therefore the
  most trustworthy trend line.

### 11.5 "Improvement" defined quantitatively *(v1.0's "monotone-ish" was unfalsifiable)*
The training-progress criterion is met if **all** of:
- the Theil–Sen slope of `eval/vs_minimax2` over the final 50% of training is positive with a
  bootstrap CI excluding 0;
- final rolling-mean league Elo exceeds the 10%-of-training rolling mean by ≥150 Elo with
  non-overlapping CIs;
- no sustained regression >100 Elo persisting over >5 consecutive evaluations;
- end-of-training scores against every fixed dev anchor are within 0.02 of their best observed
  value.

### 11.6 Sealed evaluation *(v1.0's "held out" was not held out)*
Anything committed to the repo is visible to the agent, so `tests/golden/` cannot hold a held-out
set, and implementing `minimax(4)`/`minimax(6)` in an early phase makes them callable from day one.
Corrected scheme:
- **Dev tactical suite** (~50 positions) lives in the repo and may be used freely.
- **Sealed suite** — a separate position set plus the deep-search anchors — lives **outside the
  working tree** at `$SEALED_EVAL_DIR`, supplied by the harness operator. Not in git, not in CI, not
  in any log the agent reads.
- `scripts/final_eval.py` takes `$SEALED_EVAL_DIR` and a checkpoint and writes
  `reports/final_eval_<sha>.json`, including a hash of the sealed suite so the result is auditable.
- The agent may run `final_eval.py` **at most once per candidate checkpoint, at most 3 candidates
  total**, recording each invocation in `PROGRESS.md`. Repeated querying is selection on the sealed
  set and is a §2.3 violation.
- If no `$SEALED_EVAL_DIR` is provided, Tier 3 is reported as **NOT EVALUATED**. That is an
  acceptable outcome. Fabricating a sealed result is not.

---

## 12. TESTING AND CODE RIGOUR

### 12.1 Volume and gates
- **≥400 tests**, ≥250 of them on `rules/` and `env/`.
- Coverage: `rules/` ≥98%, `env/` ≥98%, `rl/` ≥90%; overall ≥92% line and ≥85% branch.
- **Mutation score ≥85% on `src/checkers/rules/`** (§12.4 C) — coverage alone does not show the
  tests would catch a rules bug.
- `mypy --strict` clean; `ruff check` clean; `ruff format --check` clean.
- Every public function documents Args/Returns/Raises.

### 12.2 Unit and golden tests
1. Every rule ID R1.1–R7.3 in ≥1 test name; traceability matrix complete.
2. Golden fixtures in `tests/golden/`: opening; forced multi-jump; **R4.5 divergence positions**
   (where delayed vs. immediate removal differ); promotion-ends-turn-mid-sequence; king mobility;
   no-legal-move loss; double-corner endgames; R6.3 boundary; R6.5 boundary. Each is hand-verified
   against §4 with its reasoning recorded.
3. **Action-encoding tests (retargeted).** v1.0 demanded an exhaustive reachable-state search to
   prove `(from, direction)` uniqueness — a trivial geometric fact not worth that budget. Test
   instead what can actually break:
   - encode/decode bijection over all 128 IDs;
   - a legal step's action ID uniquely determines destination **and** captured square;
   - every generated legal step maps to exactly one ID, and no two distinct legal steps collide;
   - the canonical 180° rotation maps direction indices correctly (round-trip through both frames);
   - a forced continuation never exposes an action belonging to a different piece;
   - a promotion-ending jump yields a mask with zero continuation actions;
   - `legal_mask.sum() == len(legal_steps())` at every state.

### 12.3 Property-based tests *(`[HYPOTHESIS]`; invariants corrected)*
Generated over **reachable** states and scoped to *"one legal step applied within one active game"* —
v1.0's unqualified phrasing wrongly constrained `reset`, fixture construction, undo, and
deserialization too:
- total piece count is non-increasing across a step, and decreases exactly by the number removed at
  the end of a sequence;
- **corrected:** *a king is never demoted*, and *rank only ever changes man → king*. v1.0's "the
  number of kings never decreases" is **false** — kings get captured — and would have rejected legal
  games;
- no piece ever occupies a light square;
- `apply(step); undo(step)` restores an identical `State` **and** an identical `state_key`;
- if any jump exists for the side to move, every legal step is a jump (R4.2);
- a man's legal steps never go backward (R3.1, R4.3.2);
- a piece in `captured_pending` is never jumped again and still blocks (R4.5);
- a jump landing on the king row yields zero continuation actions (R5.2);
- `no_progress` increments exactly once per completed checkers move, never per substep (R6.3);
- `position_key()` raises mid-sequence; `state_key()` differs between aliased states (§5.3);
- every game terminates within `max_plies`.

### 12.4 Correctness evidence vs. regression evidence *(v1.0 blurred these)*
Two independent generators agreeing five million times proves **agreement, not correctness** — they
can share a misreading of the rules. Correctness evidence must come from outside the codebase:
- **A — External anchors (correctness evidence):** the `[WCDF]` text clause by clause; ≥20
  **published game transcripts** replayed end to end with every move legal and the recorded result
  reproduced; externally published perft counts **only if** a trustworthy source is cited in
  `docs/RULES.md`.
- **B — Independent oracle (strong corroboration):** a second, deliberately different generator
  (naive object/list vs. bitboard). CI cross-checks them over BFS-reachable states to depth D plus
  millions of playout states. Any disagreement halts the run, is written to
  `tests/golden/disagreements/`, is adjudicated **by hand against §4** — never by editing the oracle
  to agree with the fast path — and becomes a permanent test.
- **C — Mutation testing (proves the tests bite):** inject rules mutations (allow backward man
  jumps; make captures optional; remove pieces immediately rather than at sequence end; permit
  continuation after promotion; add a majority-capture rule) and require the suite to kill each one.
  Mutation score ≥85% on `rules/`.
- **D — Metamorphic tests:** colour swap, 180° rotation, and mirror symmetry produce correspondingly
  transformed legal move sets and identical game-theoretic outcomes.
- **E — Internally generated perft (REGRESSION EVIDENCE ONLY):** committed as a golden file labelled
  `REGRESSION BASELINE — NOT GROUND TRUTH` inside the file and wherever reported. It may never be
  cited in the acceptance report as correctness evidence.

### 12.5 Fuzz tiers *(v1.0 contradicted itself: 10^6 steps vs 10^6 games)*
One million games at ~100 plies is ~10^8 steps with dual generators and per-step invariants — not a
CI job.

| Tier | Scale | When |
|---|---|---|
| PR CI | 50,000 steps, all invariants | every `make check` |
| Phase gate | 5,000,000 steps + adversarial generation (near-boundary R6.3/R6.5, deep sequences, promotion during a sequence) | phase exit |
| Nightly | 20,000,000 steps, dual-generator differential on | scheduled |
| Release soak | 1,000,000 games, invariants sampled at 1% of steps | once before Tier-1 acceptance, if budget allows; otherwise reported NOT RUN |

### 12.6 RL numerics tests T1–T8 *(replaces v1.0's muddled "overfit a batch")*
v1.0 asked PPO to drive a fixed batch's losses to zero and the policy to determinism. That is
confused: the clipped surrogate's optimum is not zero, stored old log-probs go stale across epochs,
the entropy bonus opposes determinism, advantage normalization changes the objective, and clipping
deliberately halts improvement. Replaced by:
- **T1 — Supervised policy memorization.** With cross-entropy against target actions (not PPO), the
  policy head fits 64 fixed states to >99% accuracy. Proves capacity and gradient flow.
- **T2 — Supervised value memorization.** With MSE against fixed scalar targets, the value head fits
  64 states to MSE < 1e-3.
- **T3 — One PPO update by hand.** A 4-transition toy rollout with hand-computed `σ_t`, `δ_t`, `A_t`,
  `returns`, ratios, clipped objective, and total loss, asserted to 1e-6. Must include one multi-jump
  continuation (σ = +1) and one side change (σ = −1).
- **T4 — Directional PPO behaviour.** Over repeated epochs on a fixed batch, probabilities of
  positive-advantage actions increase and negative-advantage actions decrease, monotonically until
  the clip threshold; `clipfrac` activates at the expected ratio bounds.
- **T5 — Gradient reachability** *(relaxed from v1.0's "every parameter nonzero")*. Per module, the
  aggregate gradient norm is nonzero on a purpose-built batch; no parameter is permanently
  disconnected across the whole suite; masked-out logits receive **exactly zero** gradient.
  Individual parameters legitimately receiving zero gradient on a given batch (inactive ReLUs,
  symmetric inputs) do not fail the test.
- **T6 — σ-degenerate equivalence.** With all `σ_t = +1` and a single actor, §7.3 reproduces a
  reference single-agent GAE implementation bitwise.
- **T7 — Forced-mate value targets.** On scripted forced wins of length 3, 5, and 7 — including one
  ending in a multi-jump and one ending by R6.2 — computed value targets equal hand-derived values
  exactly.
- **T8 — Buffer composition.** No non-trainable transition reaches the policy loss; GAE inputs are
  time-contiguous per `env_id`; a rollout boundary falling mid-capture-sequence bootstraps with the
  correct σ.

### 12.7 Determinism, scoped realistically *(v1.0's bitwise-GPU demand was unachievable)*
- **D1** One `seed_everything(seed)` covering `random`, `numpy`, `torch`, CUDA, and per-worker env
  streams with deterministic distinct sub-seeds.
- **D2 — CPU:** two runs with identical seed, config, and software stack produce **bitwise-identical
  losses** for the first 10 updates. Required test.
- **D3 — GPU, same machine and stack:** identical action sequences and losses within
  `atol=1e-5, rtol=1e-4` over 10 updates.
- **D4 — Across machines, GPU architectures, or library versions:** **no bitwise claim is made.**
  Runs record torch/CUDA/cuDNN/BLAS versions, GPU model, and determinism flags so deviations are
  explainable. Reproduction instructions name the reference stack.
- **D5** `torch.use_deterministic_algorithms(True)` in test and smoke modes. If disabled for
  throughput in a production run, `deterministic: false` is logged in the W&B config and stated in
  the README.

### 12.8 Checkpoint and resume *(contents expanded for self-play)*
Checkpoints are written **only at a completed update boundary**. Contents: model state; optimizer
state; `RunConfig`; `TrainerState`; git SHA + dirty flag; `global_step` and `update_idx`; all RNG
states (python, numpy, torch CPU/CUDA, per-env, opponent sampling); **the league pool with snapshot
IDs and their parameters or artifact references**; **fully serialized vector-env states including
mid-capture-sequence fields, per-env `no_progress`, and `ply`**; the arena repetition history if
enabled; LR and entropy schedule phase; minibatch permutation state if relevant; AMP `GradScaler`
state; W&B run id and logging step counters.
**Test:** a run interrupted at update *k* and resumed reproduces the uninterrupted run's losses
within §12.7 tolerance for updates *k+1…k+10*, including a case where an environment is mid-sequence
at the checkpoint.

### 12.9 Tooling — single formatter *(v1.0 double-mandated)*
**`ruff` is authoritative** for both linting (`E,F,W,I,N,UP,B,SIM,ANN,RET,ARG,PTH,ERA,PL`) and
formatting (`ruff format`). Black is **not** used; the overlap created churn for no benefit. All
tool versions are pinned exactly in `pyproject.toml`.

### 12.10 Network policy *(disambiguated)*
- **Dependency installation may use the network.**
- **Tests, fuzz, smoke runs, and training must run with the network disabled:** no runtime
  downloads, no pretrained weights, no online W&B requirement (`WANDB_MODE=offline` must work), no
  network-backed rules verification at test time. A CI job runs the suite with egress blocked to
  prove it.
- The `[WCDF]` verification of §3.1 is a **build-time** step whose output is committed as text in
  `docs/RULES.md`; it is never fetched at test time.

### 12.11 Make targets
```
make format   # ruff format
make lint     # ruff check
make types    # mypy --strict src tests
make test     # pytest + coverage gates
make mutate   # mutation testing on rules/   (slow; phase gate + nightly)
make fuzz-ci  # 50k-step invariant fuzz
make fuzz     # 5M-step phase-gate fuzz
make perft    # dual-generator perft comparison + external anchors if present
make check    # format-check + lint + types + test + fuzz-ci      <-- the gate
make smoke    # ~5-minute end-to-end train, WANDB_MODE=offline, network blocked
make train    # full run
make eval     # dev-suite arena + population metrics report
```

---

## 13. WEIGHTS & BIASES INSTRUMENTATION

### 13.1 Setup
Project `checkers-ppo`; run name `{phase}-{git_sha7}-seed{seed}-{stamp}`; tags `phase-N`, `seed-K`,
`arm-{A0..A3}`, `stage-{A,B,C}`, `failed`. `wandb.init(config=asdict(run_config))` plus git SHA,
dirty flag, library versions, hostname, device, determinism flags. `WANDB_MODE=offline` must work;
`wandb sync` is documented. **No API key in the repo** — a test asserts no key-shaped string is
committed.

### 13.2 Metrics *(names and definitions corrected)*
**Optimization:** `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/approx_kl` (k3),
`train/clipfrac`, `train/explained_variance`, `train/grad_norm`, `train/lr`, `train/ent_coef`,
`train/kl_early_stops`, `train/trainable_frac`, `charts/SPS`.

**Masking (renamed).** v1.0's `illegal_action_attempts` measured nothing real: under a masked
distribution, illegal actions have probability zero, so a nonzero count would indicate masking
failure, env/mask disagreement, an index-mapping error, or numerical corruption — not an "attempt."
Log instead: `mask/sample_legality_violations` (**must be exactly 0**), `mask/oracle_disagreements`
(**must be 0**), `mask/empty_mask_count` (**must be 0**), `mask/mean_legal_actions`,
`mask/continuation_state_frac`.

**Policy health:** `policy/normalized_entropy`, computed **per state and then averaged**:
`mean_i[ H(π_i) / log k_i ]` over states with `k_i > 1`. v1.0 used `mean_i[H_i] / log(mean_i[k_i])`,
which is not the mean normalized entropy — by Jensen, the mean of a ratio is not the ratio of means
— and would misdiagnose collapse. Also `policy/max_prob_mean`, `policy/frac_states_k_eq_1`.

**Value health:** `value/mean`, `value/std`, `value/target_mean`, `value/explained_variance`,
`value/calibration_mae` (binned predicted value vs. realized outcome `z`).

**Game:** `env/mean_game_len_moves`, `env/mean_game_len_steps`, `env/draw_rate`,
`env/first_player_win_rate`, `env/captures_per_game`, `env/mean_sequence_len`, `env/promotion_rate`,
`env/no_progress_draws`, `env/ply_cap_draws`, `env/first_player_frac`.

**Evaluation:** `eval/vs_random`, `eval/vs_greedy`, `eval/vs_minimax2` (each with `_ci_low`,
`_ci_high`, `_games`), `eval/league_elo` + CI, `eval/payoff_matrix` (table),
`eval/three_cycle_count`, `eval/exploitability_proxy`, `eval/dev_tactical_acc`,
`eval/greedy_vs_sampled_delta`.

**Artifacts:** versioned checkpoints; the resolved config; the git diff if dirty;
`docs/PPO_CHECKLIST.md`; a `wandb.Table` of rendered games every `eval_every` including **at least
one loss and one draw**, not only wins; the payoff matrix; the Elo curve.

### 13.3 Alerts that halt training
`mask/sample_legality_violations > 0`; `mask/oracle_disagreements > 0`; `mask/empty_mask_count > 0`;
NaN/Inf in any loss; `approx_kl > 10 × target_kl`; `explained_variance < 0` sustained 20 updates;
`policy/normalized_entropy < 0.02` before 25% of training; `eval/vs_random < 0.6` after 20% of
training. Each halt requires a diagnosis entry in `PROGRESS.md` before resuming.

### 13.4 Sweeps
Only after Gate 7, only at Stage A/B budgets. Sweep metric: dev-anchor score plus population
metrics — **never** anything from the sealed suite. Summarized in `DECISIONS.md`.

---

## 14. PHASES AND GATES

Strictly in order. A phase is GREEN only when its gate command's exit code and log path are recorded
in `PROGRESS.md`.

**Phase 0 — Scaffolding.** Repo layout below; pinned deps; ruff/mypy/pytest/coverage/mutation wired;
`make check`; pre-commit; offline-CI job; `STATE.json`, `PROGRESS.md`, `DECISIONS.md`,
`BLOCKERS.md`, `logs/` initialized.
*Gate 0:* `make check` passes on a real minimal suite; deliberately introduce a lint error and a
failing test to prove the gate can go red; revert.

```
checkers-ppo/
├── GOAL.md  STATE.json  PROGRESS.md  DECISIONS.md  BLOCKERS.md  README.md
├── Makefile  pyproject.toml  .pre-commit-config.yaml
├── logs/{iterations,test-output,gates}/  logs/SUMMARY.md
├── src/checkers/
│   ├── rules/   state.py board.py moves.py oracle_moves.py zobrist.py notation.py terminal.py
│   ├── env/     checkers_env.py vec_env.py masking.py encoding.py serialize.py
│   ├── agents/  random_agent.py greedy_agent.py minimax_agent.py policy_agent.py
│   ├── rl/      networks.py masked_categorical.py gae.py buffer.py ppo.py selfplay.py league.py
│   ├── eval/    arena.py elo.py population.py suites.py power.py
│   ├── config.py trainer_state.py schedules.py logging_wandb.py seeding.py cli.py
├── tests/{rules,env,rl,eval,property,metamorphic,integration,golden}/
└── scripts/  train.py evaluate.py final_eval.py play_human.py reproduce.sh
```

**Phase 1 — Rules verification.** Produce `docs/RULES.md` per §3.1: per-rule source clauses, ENGINE
VARIANT labels for R6.3/R6.4/R6.5, and the R6.7 termination proof sketch.
*Gate 1:* every rule ID has a source clause or an explicit variant label; unverified items filed as
P1 blockers.

**Phase 2 — State model, board, move generation.** Implement §5.1, R1–R5, R7, plus the independent
oracle generator.
*Gate 2:* rules tests green; **R4.5 delayed-removal divergence fixtures** pass; dual-generator
differential agrees over 5M positions and BFS depth D; metamorphic suite green; ≥20 published
transcripts replay exactly; mutation score ≥85% on `rules/`; coverage ≥98%.

**Phase 3 — Terminal conditions and hashing.** Implement R6 and §5.3.
*Gate 3:* boundary fixtures for R6.1, R6.2, R6.3 (39 vs. 40 per player), R6.5 (511 vs. 512);
key-separation tests; `position_key()` raises mid-sequence; incremental-vs-recomputed hash property
test; termination proof committed.

**Phase 4 — Environment, encoding, masking.** Implement §6 and §5.2.
*Gate 4:* 5M-step fuzz with zero invariant violations, zero mask disagreements, zero empty masks;
the N7 aliasing regression test green; canonical-rotation tests green; `IllegalActionError` test;
mid-sequence serialize/restore round-trip.

**Phase 5 — Baselines and arena.** `random`, `greedy`, `minimax(d)`, `arena.py`, `elo.py`,
`population.py`, `power.py`.
*Gate 5:* the §11.3 criteria (power-justified N, CIs, no catastrophic inversion, tactical superset
check, non-monotonicity reported if present); Elo and payoff-matrix code validated against
hand-worked examples.

**Phase 6 — RL core, verified offline.** `masked_categorical.py`, `gae.py`, `buffer.py`,
`networks.py`, `ppo.py`.
*Gate 6:* T1–T8 all green; masked-categorical dtype tests green; D2/D3 determinism green.

**Phase 7 — Self-play loop + W&B.** `selfplay.py`, `league.py`, `logging_wandb.py`, `train.py`.
*Gate 7:* a 30-minute smoke on 3 seeds that logs every §13.2 metric, records zero mask violations,
beats `random_agent` ≥0.90 over a power-justified match, and passes the §12.8 resume test including
a mid-sequence checkpoint.

**Phase 8 — Ablations, then full runs.** Stage-A screening; the mandatory Stage-B ablations (§8.5,
§10.2); then Stage-C: 3 seeds at full budget for the baseline and ≤3 promoted variants. Produce
`docs/PPO_CHECKLIST.md`, `docs/ML_TEST_SCORE.md`, `docs/METRICS.md`, README.
*Gate 8:* §19 Tier 1 and Tier 2 green; Tier 3 evaluated once per candidate or reported NOT
EVALUATED.

**Phase 9 — OPTIONAL, only after Gate 8.** MCTS/ExIt, architecture search, larger league. No change
to Phase 2–4 code without re-running all prior gates.

---

## 15. HARNESS OPERATION AND CONTEXT BUDGET

### 15.1 Why this section exists
An append-only `PROGRESS.md` containing raw test output, re-read every iteration, grows without
bound and is the most likely cause of a runaway or out-of-memory long session. This is treated as a
first-class operational risk, not a documentation preference.

### 15.2 Rules
- `PROGRESS.md` is **rewritten** each iteration and capped at 400 lines (§0.3).
- Raw output goes to `logs/`, and `logs/` is never read wholesale.
- Every 25 iterations, `logs/SUMMARY.md` is regenerated as a compaction: phases completed, gates
  passed, bugs found and their root causes, open risks. This is long-term memory.
- Reported evidence = command + exit code + counts + log path (+ ≤30-line excerpt when red).
- If `PROGRESS.md` exceeds its cap, compacting it is the iteration's first action, ahead of
  engineering work.
- If the agent notices it has re-read the same large artifact three iterations running, it stops and
  writes a compaction instead.

---

## 16. FAILURE PLAYBOOK *(`[KARPATHY]` ladder — do not skip rungs)*
1. **Rules?** Run the golden, metamorphic, differential, and transcript-replay suites. Most "the
   agent learns something strange" bugs are move-generation or sign bugs.
2. **Sequence state?** Check `captured_pending`, `moving_square`, and the continuation mask. This is
   the highest-density bug region in this design.
3. **Perspective sign?** Re-verify §7.3 with T3/T6/T7. Print a five-step game's `σ_t`, `δ_t`, `A_t`
   by hand and compare.
4. **Mask?** Assert mask equality against the oracle for the exact failing states; confirm the
   stored mask, not a recomputed one, is used in the update.
5. **Data?** Dump a rollout to a W&B Table and *read the games*. Legal? Sane? Terminating for the
   reason you think?
6. **Capacity?** Run T1/T2. If supervised memorization fails, the bug is in the loss, optimizer, or
   network — not in RL.
7. **PPO health?** `approx_kl`, `clipfrac` (healthy ≈0.05–0.3), `explained_variance` (should rise
   above 0), `grad_norm` (not pinned at the clip forever), `trainable_frac`.
8. **Only then** hyperparameters — one change, new run, Stage A → B (§8.5).

Each descent is recorded in the iteration log with the rung that found the bug.

---

## 17. CHECKERS-SPECIFIC FAILURE MODES *(each gets a permanent test, written first)*
- Men jumping backward (R4.3.2).
- Continuing a jump after promotion (R5.2).
- **Removing captured pieces immediately instead of at sequence end (R4.5)** — the v1.0 defect.
- Re-jumping an already-marked piece (R4.5).
- A marked piece failing to block a landing square (R4.5) — the subtler half of the same rule.
- Mandatory capture applied per piece instead of per player (R4.2).
- A majority-capture rule leaking in from International draughts (R4.6).
- **Continuation state omitted → two different states sharing an observation** (§5.1, §6.2).
- **`no_progress` incremented per environment step instead of per completed move** (R6.3, §5.2).
- Repetition counted on a mid-sequence state (R6.4, §5.3).
- Board orientation flipped for one colour, silently mirroring moves.
- **σ_t missing from the advantage recursion, not just from the bootstrap** (§7.3).
- Rollout truncation mid-sequence bootstrapping with the wrong sign (§7.5, T8).
- Stalemate scored as a draw instead of a loss (R6.2).

---

## 18. DELIVERABLES
1. The repository, all phase gates GREEN in `STATE.json`.
2. `README.md`: what it is, install, test, train, reproduce; W&B links; results with CIs; an explicit
   **"Engine variants and departures from WCDF"** section; an honest limitations section naming any
   UNVERIFIED rule and any NOT RUN test tier.
3. `docs/RULES.md` — rule → source clause → tests, plus the R6.7 termination proof.
4. `docs/PPO_CHECKLIST.md` — `[HUANG37]` items adjudicated, each with its Tier.
5. `docs/ML_TEST_SCORE.md` — self-score against `[MLTS]`.
6. `docs/METRICS.md` — every metric with its exact formula and healthy range (especially
   `policy/normalized_entropy`).
7. `DECISIONS.md` — every non-obvious choice with its authority Tier and evidence Stage.
8. `logs/SUMMARY.md` — the compacted build history, including failures.
9. W&B: ≥3 seeded baseline runs, the mandatory Stage-B ablations, model artifacts.
10. `scripts/play_human.py` — a human plays the trained agent in the terminal with ACF notation and
    visible pending-capture marks.
11. `reports/final_eval_<sha>.json`, or a written statement that Tier 3 was NOT EVALUATED.

---

## 19. ACCEPTANCE — THREE SEPARATE TIERS

v1.0 let a poor result pass through an "or honestly diagnose it" escape clause, which made the
performance bar non-binding while the mission still claimed a strength result. The tiers are now
separated and each is reported independently.

### Tier 1 — Engineering acceptance (BINDING)
- [ ] `make check` exits 0 from a clean clone with **egress blocked** after dependency install.
- [ ] ≥400 tests, all passing, none skipped or xfailed; coverage gates met; mutation score ≥85% on
      `rules/`.
- [ ] `mypy --strict`, `ruff check`, `ruff format --check` all clean.
- [ ] Every rule ID R1.1–R7.3 appears in a passing test; `docs/RULES.md` traceability complete;
      every ENGINE VARIANT labelled as such in code, docs, and README.
- [ ] R4.5 delayed-removal fixtures pass, including ≥1 position where delayed and immediate removal
      demonstrably differ — or a committed proof that no such position exists.
- [ ] §5.3 key-separation tests pass; `position_key()` raises mid-sequence.
- [ ] Dual-generator differential green over ≥5M positions; ≥20 published transcripts replay exactly;
      metamorphic suite green; external perft cited **or** explicitly reported unavailable, with the
      internal perft file labelled REGRESSION BASELINE.
- [ ] Phase-gate fuzz (5M steps) clean; release soak reported as run or NOT RUN.
- [ ] T1–T8 green, including the hand-computed two-player GAE test with both σ values.
- [ ] D2 CPU bitwise determinism green; D3 GPU tolerance green; §12.8 resume test green including a
      mid-capture-sequence checkpoint.
- [ ] `mask/sample_legality_violations`, `mask/oracle_disagreements`, and `mask/empty_mask_count` are
      exactly 0 across all runs.
- [ ] All §18 deliverables exist and are non-placeholder.
- [ ] No §2 DO NOT violated; the agent states this explicitly and lists near-misses.
- [ ] `scripts/reproduce.sh` reproduces the headline numbers from scratch on the reference stack.
- [ ] `BLOCKERS.md` has no unresolved P0.

### Tier 2 — Learning acceptance (BINDING)
- [ ] ≥3 seeds trained at full budget; all results as mean ± 95% CI with game counts.
- [ ] ≥0.95 score vs. `random_agent` and ≥0.85 vs. `greedy_agent`, each over a power-justified,
      colour-balanced match.
- [ ] ≥0.60 score vs. `minimax(2)` — a **dev** anchor, so this is a legitimate non-leaking target.
- [ ] The §11.5 improvement criteria are met (Theil–Sen slope, Elo gain, no sustained regression,
      anchor scores near their best).
- [ ] Population metrics reported: payoff matrix, three-cycle count, exploitability proxy.
- [ ] The mandatory Stage-B ablations are run and their conclusions recorded — including any that
      came out against the design's expectations (e.g. if league arm A0 wins).

### Tier 3 — Strength (NON-BINDING, reported either way, never retried)
- [ ] `final_eval.py` run at most once per candidate, ≤3 candidates, against `$SEALED_EVAL_DIR`;
      score, suite hash, and invocation count recorded — **or** `NOT EVALUATED` if no sealed
      directory was supplied.
- [ ] The result is reported plainly. A low score here is not a failure of the project. Concealing
      it, retrying against the sealed set, or converting Tier 3 into an unfalsifiable narrative is.

---

## 20. FINAL WORD TO THE AGENT

Two things will tempt you. The first is to move faster by trusting untested code, loosening a gate
"just for now," or declaring a phase done because it looks done. Don't. The second is subtler and
newer: this document is long, detailed, and confident — and its previous version was still wrong in
more than forty places, including a rule of checkers, the definition of the environment state, and
the central equation of the learning algorithm. Read it as a spec to be checked, not scripture to be
obeyed. If a rule here contradicts the rules of checkers, or an equation here does not survive being
worked by hand, you have found a defect in the spec, not a reason to write that defect into the
code. File the blocker and stop. The finish line is a system that is *actually* correct — not one
that is merely compliant.
