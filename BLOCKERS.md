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
