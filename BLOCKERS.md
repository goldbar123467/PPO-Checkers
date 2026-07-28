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
