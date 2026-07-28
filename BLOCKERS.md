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
