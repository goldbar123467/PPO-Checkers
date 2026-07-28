# Blockers

### BLOCK-001 [P1] §4 R6.6 — no-draw-by-agreement departure is not labelled

Claim in GOAL.md: "R6.6 No draw by agreement."

Why it is wrong: WCDF rule 1.32 explicitly allows a draw when both players agree. Omitting agreed
draws is reasonable for an autonomous self-play MDP, but it is a deliberate rules departure and
must be labelled ENGINE VARIANT under §3.1, just like R6.3–R6.5.

Evidence: WCDF *Rules of Draughts (Checkers)*, rule 1.32, page 5;
<https://wcdf.net/rules/rules_of_checkers_english.pdf>, archived source hash
`aa1d1235632046c05db7621437f16c33bc7b86b472ccaa039a7a41b897b180b7`.

Phases affected: 1, 3, 4, 8.

Proposed correction: change the heading to "R6.6 — ENGINE VARIANT: no draw by agreement" and add
"Declared departure from WCDF 1.32; autonomous agents cannot negotiate."

Status: OPEN (implementation will conservatively label the departure; human resolution still
required before all gates can be declared GREEN).

### BLOCK-002 [P1] R4.5/Gate 2 — required landing-block divergence is geometrically impossible

Claim in GOAL.md: R4.5 says a marked capture can block a later landing and Gate 2, §12.2, and
§19 E5 require a delayed-removal position where immediate and delayed removal admit different
legal continuations. R4.5 also says that if no such position exists, it must be proved and
recorded.

Why the required fixture cannot exist in American Checkers: WCDF 1.18–1.21 define every jump as
two diagonal squares over the intervening enemy. Thus each jump changes both coordinates by ±2.
Throughout one sequence the mover stays in its original `(row mod 2, column mod 2)` class, while
every captured midpoint is in the opposite class. No captured square can therefore equal any
later landing square. A pending capture also cannot be jumped again under WCDF 1.20; removing it
immediately makes it empty, which likewise cannot be jumped. The two removal timings differ in
mid-sequence state/observation occupancy, but not in the legal continuation set when the no-repeat
rule is implemented.

Evidence: WCDF *Rules of Draughts (Checkers)*, rules 1.18–1.21, page 3;
<https://wcdf.net/rules/rules_of_checkers_english.pdf>, archived source hash
`aa1d1235632046c05db7621437f16c33bc7b86b472ccaa039a7a41b897b180b7`;
exhaustive 32-square geometry test
`tests/rules/test_captures.py::test_r4_5_landing_on_a_pending_square_is_geometrically_impossible`.

Phases affected: 2, 4, 8.

Proposed correction: replace the impossible divergence-fixture requirement with (a) an exact
mid-sequence fixture proving the marked piece remains in occupancy and cannot be jumped twice,
(b) the coordinate-parity proof, and (c) a mutation test that kills immediate removal by checking
the full state/observation rather than a nonexistent legal-set difference.

Status: OPEN (the engine implements delayed removal exactly; Phase 2 cannot honestly be labelled
GREEN while the contradictory fixture remains binding).

### BLOCK-003 [P1] §12.4 D — separate mirror/colour/rotation symmetries do not exist

Claim in GOAL.md: the metamorphic suite must treat "colour swap, 180° rotation, and mirror
symmetry" as transformations that each preserve legal move sets and game-theoretic outcomes.

Why the separate transformations are invalid: on an even 8×8 board, horizontal and vertical
reflections swap dark and light square parity, so they do not map the 32-square playing lattice to
itself. The two diagonal reflections preserve square colour but map promotion rows to columns, so
they do not preserve either player's forward direction or king row. A colour swap without spatial
rotation reverses which direction men should travel without relocating them; a 180° rotation
without a player swap has the same defect. The only nontrivial geometric symmetry preserving the
American Checkers objective is the *composition* of 180° rotation and player/colour swap.

Evidence: WCDF 1.1, 1.4–1.5, 1.15–1.17 define the board, orientation, forward movement, and king
rows; <https://wcdf.net/rules/rules_of_checkers_english.pdf>, archived source hash
`aa1d1235632046c05db7621437f16c33bc7b86b472ccaa039a7a41b897b180b7`. The D4 audit and valid
composed transition-equivariance test are in `tests/metamorphic/test_rules_symmetry.py`.

Phases affected: 2, 4, 8.

Proposed correction: require the composed `rotate180 + swap(PlayerId)` metamorphism, plus its
involution and transition-equivariance checks. Remove separate mirror, colour-only, and
rotation-only claims.

Status: OPEN (the valid combined symmetry is tested through BFS depth 4; Phase 2 cannot honestly
be labelled GREEN while the impossible separate requirements remain binding).

### BLOCK-004 [P1] Gate 2 — recorded results cannot generally be derived in Phase 2

Claim in GOAL.md: Gate 2 requires at least 20 published transcripts replayed end to end with every
move legal and "the recorded result reproduced," while Phase 3 is the first phase allowed to
implement terminal conditions. The specification also removes agreement draws in R6.6.

Why literal result derivation is unavailable: published checkers scores routinely end at
resignation, adjudication, or draw agreement while legal moves remain. None of the selected 20
decisive published records ends in a no-piece/no-legal-move board; all result tags therefore come
from the publisher, not a result function derivable from the move list. Phase 2 cannot implement
R6 out of order, and even Phase 3 cannot infer a resignation or agreement from board state that
does not encode it.

Evidence: the hash-pinned source and deterministic extraction are documented in `docs/RULES.md`;
`tests/golden/test_published_transcripts.py` replays 515 complete moves as 561 legal environment
steps with unique compressed-capture resolution. The final-state audit found legal moves in all
20 final positions. WCDF 1.31–1.32 separately recognize resignation and agreement.

Phases affected: 2 and 3.

Proposed correction: Gate 2 should require every published move to replay legally and the source
result tag to be preserved exactly. Gate 3 should additionally derive results only for transcripts
that actually end in an engine-observable terminal position; resignation/agreement/adjudication
must remain explicitly source-recorded outcomes.

Status: OPEN (20 move-legality replays and exact result-tag preservation pass; no claim of
board-derived outcomes is made).

### BLOCK-005 [P1] §5.3 — state key omits fields that change terminal transitions

Claim in GOAL.md: `state_key` is Zobrist over everything in §5.1 that affects legal transitions,
but its explicit field list contains placement, side, `capture_in_progress`, `moving_square`, and
`captured_pending` only.

Why the explicit list is wrong: R6.3 makes `no_progress` determine whether a state draws, and R6.5
makes `ply` determine whether it draws. Two otherwise identical states at counter 39 versus 40 or
ply 511 versus 512 therefore have different terminal transitions but collide deterministically
under the listed key. `sequence_origin` is also propagated into subsequent full states during a
capture continuation, so omitting it prevents the key from identifying the complete §5.1 state.
This contradicts §5.3's own no-semantic-collision objective and the Markov-state rationale.

Evidence: `GOAL.md` R6.3, R6.5, §5.1, and §5.3; the Phase 3 key-separation tests construct the
counter and ply collisions directly.

Phases affected: 3, 4, 5, and every cache/checkpoint consumer.

Proposed correction: include `sequence_origin`, both `no_progress` counters, and `ply` in
`state_key`; retain the narrower placement-plus-side `position_key` solely for boundary repetition.

Status: OPEN (the implementation will use the source-correct complete key; Phase 3 cannot be
labelled GREEN until the read-only field list is amended).

### BLOCK-006 [P1] Gate 4/N7 — logit regression precedes the network phase

Claim in GOAL.md: Gate 4 requires "the N7 aliasing regression test green," while N7 specifically
requires two sequence-distinct states to produce different **logits**. The network is not
implemented until Phase 6, whose strict phase scope includes §9 and the N1–N7 architecture.

Why it cannot be completed in Phase 4: Phase 4 implements §6 and §5.2 only. There is no policy
network whose logits can be tested without implementing Phase 6 out of order. Different encoded
observations are necessary but do not mathematically guarantee different logits for arbitrary
weights; a constant or degenerate network can map distinct tensors to the same output.

Evidence: `tests/env/test_encoding.py::test_n7_pending_and_forced_planes_prevent_observation_aliasing`
proves the Phase 4 representation is non-aliased. The actual different-logits test remains a Phase
6 obligation against the implemented N1–N6 model.

Phases affected: 4 and 6.

Proposed correction: Gate 4 should require different observations plus different legal masks for
the boundary/mid-sequence pair. Gate 6 should require the N7 different-logits regression once the
network exists.

Status: OPEN (the representation-level regression passes; Phase 4 cannot honestly be labelled
GREEN until the gate is retargeted or Phase 6 supplies the logit test).
