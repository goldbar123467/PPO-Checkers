# American Checkers Rules, Authority, and Test Traceability

This document is the build-time rules record required by `GOAL.md` §§3.1, 4, and 14. It
paraphrases the primary publication instead of copying it wholesale. Rule clauses are Tier A;
project extensions and engine variants are labelled explicitly.

## Primary Source Provenance

- Publisher: World Checkers Draughts Federation (WCDF), the governing body identified in the
  [WCDF bylaws](https://wcdf.net/bylaws.htm).
- Publication: [*Rules of Draughts (Checkers)*](https://wcdf.net/rules/rules_of_checkers_english.pdf),
  18-page PDF.
- Retrieved: 2026-07-27 America/New_York.
- Size: 177,885 bytes.
- SHA-256: `aa1d1235632046c05db7621437f16c33bc7b86b472ccaa039a7a41b897b180b7`.
- HTTP metadata: `Last-Modified: Mon, 19 Aug 2013 22:58:02 GMT`; ETag
  `"2b6dd-4e454df533280"`.
- Scope used: WCDF Section One, clauses 1.1–1.32.2. Tournament clocks, conduct, and administration
  are outside this engine's game-state contract.

Notation examples are corroborated by the WCDF-published
[*2017 GAYP World Qualifier* game scores](https://wcdf.net/games/WQT_2017_Lebanon_TN_Jim_Loy.pdf),
which use `-` for simple moves and `x` for captures:

- Size: 244,080 bytes.
- SHA-256: `58608fd6fb9759aa258e554c40357dffd05d77a1aea2538233b7c69a1c1712ad`.
- HTTP metadata: `Last-Modified: Tue, 26 Sep 2017 21:18:56 GMT`.

Tests and runtime code never fetch these URLs. The clause mapping and hashes in this committed file
are the offline build output. A fresh source audit can download the same URLs and compare hashes.

## Published Transcript Provenance

Twenty external game scores are selected deterministically from Bob Newell's
[*PDN Downloads*](https://www.bobnewell.net/checkers/pdn/pdndownloads.html) publication page,
which identifies *Tricks, Traps and Shots of the Checkerboard* as Willie Ryan material converted to
PDN by Bob Murr. The pinned archive is
[`tts.zip`](https://www.bobnewell.net/checkers/pdn/tts.zip):

- Retrieved: 2026-07-27 America/New_York; size 19,681 bytes.
- Archive SHA-256: `d1c2eb648e46827cf7eb2441f4ab22964329aa013896a0681932d837bd6de662`.
- Member: `Tricks traps and shots.pdn`.
- Selection: the first 20 non-setup entries named `TTS Game 1` through `TTS Game 20`.
- Generated fixture SHA-256:
  `1ab2a5b530d2ff44d5595e0d7674ec521db12e99fed21cbe13c64f8790974311`.
- License: not stated on the source page. The small fixture contains only factual move/result
  records, is used solely for tests, and is excluded from training data.

The 20 scores contain 515 completed moves, expanding to 561 step-wise environment transitions.
Every move has exactly one legal interpretation; this is external correctness evidence for move
legality. The records contain 11 `0-1` and nine `1-0` source results, but all stop while legal moves
remain (resignation or published analysis cutoff). Those result tags are preserved exactly, not
misrepresented as board-derived outcomes; BLOCK-004 records the Gate 2 wording conflict.

## External Perft Provenance

Aart J. C. Bik's peer-reviewed paper, [*Computing Deep Perft and Divide Numbers for
Checkers*](https://doi.org/10.3233/ICG-2012-35403), ICGA Journal 35(4), 206–213 (2012), publishes
American-checkers perft and divide counts through depth 28. The author's earlier
[start-position table](https://aartbik.blogspot.com/2009/02/perft-for-checkers.html) reports the
first 12 depths and states that two independent engines, Cake and Checkers for Android, agreed.

The offline fixture `tests/golden/data/external_perft.json` records counts, retrieval metadata,
and source hashes. `tests/golden/test_external_perft.py` compares the production transition engine
at depths 0–7, decrementing depth only when a complete checkers move ends so multi-jump environment
steps remain one move. All eight counts match exactly: `1, 7, 49, 302, 1469, 7361, 36768,
179740`. These are external correctness evidence, not an internally generated regression baseline.

## Terminal and Hashing Adjudication

Terminal losses follow WCDF 1.30. Automatic no-progress, optional automatic repetition, and the
512-step cap are the declared R6.3–R6.5 ENGINE VARIANTS above. R6.6 is also labelled ENGINE VARIANT
in implementation and documentation because autonomous agents cannot negotiate, although the
read-only goal omitted that label (BLOCK-001). If a loss and an engine-variant draw boundary
coincide, the loss is evaluated first; this prevents a side with no pieces or no move from escaping
WCDF 1.30 through a training rule.

The Zobrist construction follows Albert L. Zobrist's University of Wisconsin Technical Report 88,
[*A New Hashing Method With Application for Game
Playing*](https://minds.wisconsin.edu/handle/1793/57624) (1970): independent feature values are
combined with XOR, permitting changes to be XORed out and in. This repository freezes a schema
version, master seed, feature indices, and SplitMix64 expansion constants in `zobrist.py`; known
initial keys are permanent tests.

`position_key` follows the narrow §5.3 repetition contract: placement plus side, and only at a move
boundary. The complete `state_key` additionally includes capture state, `sequence_origin`, both
`no_progress` counters, and `ply`. The latter three correct BLOCK-005: omitting counter 40 or ply
512 would give identical cache keys to states with different terminal transitions.

## FROZEN ACF 1–32 orientation

Convention: the first player is `PlayerId.RED`, matching WCDF 1.9 and 1.13. Red occupies 1–12,
moves first, and moves toward increasing square numbers. White occupies 21–32 and moves toward
decreasing square numbers. Code, notation, metrics, and tests use Red/White consistently.
Red moves toward increasing square numbers; White moves toward decreasing square numbers.

Dark-square numbering viewed from Red's side:

```text
             WHITE HOME / RED KING ROW
          32      31      30      29
      28      27      26      25
          24      23      22      21
      20      19      18      17
          16      15      14      13
      12      11      10       9
           8       7       6       5
       4       3       2       1
             RED HOME / WHITE KING ROW
```

Only the shown squares are playable. The diagram is FROZEN: a change requires an explicit rules
adjudication and re-running every rules, notation, transcript, and symmetry gate.
The double corners are Red's squares 1/5 and White's squares 28/32.

## Clause Adjudication Notes

### Delayed removal in a capture sequence

WCDF 1.18 describes removal after a single jump, while the more specific multi-jump clause 1.19
places removal of all captured pieces at the end of the sequence. WCDF 1.20 additionally forbids a
piece from being jumped twice in that sequence. For multi-jumps, 1.19 is the controlling specific
clause: jumped pieces remain physically present until the complete move ends. They therefore still
occupy their squares and cannot be jumped again. This interpretation is implemented as
`captured_pending` and tested against the complete mid-sequence state.

The narrower claim in `GOAL.md` that a marked piece can block a later *landing* is geometrically
impossible for American Checkers. Every short jump changes row and column by ±2, so the moving
piece stays in one coordinate-parity class for its whole sequence. Every jumped midpoint belongs
to the opposite class. A midpoint can never be a later landing. If a captured piece is removed
immediately it is not jumpable; if retained and marked, WCDF 1.20 forbids jumping it again. Hence
removal timing changes the required mid-sequence state and observation, but cannot change the
legal continuation set when the no-repeat rule is honored. BLOCK-002 records the contradiction
between this proof fallback in R4.5 and later gate language demanding a divergence fixture.

### Draw departures

WCDF 1.32, 1.32.1, and 1.32.2 use agreement or a player's demonstration to a referee. Autonomous
self-play has neither negotiation nor a referee, so R6.3–R6.6 are engine rules, not claims about
WCDF play. R6.6 was not labelled as a variant in `GOAL.md`; `BLOCKERS.md` BLOCK-001 records that
specification defect. Code and documentation conservatively label it ENGINE VARIANT.

### Project-only contracts

An environment step, parser API, and FEN-like full-state serialization are software contracts, not
over-the-board WCDF rules. They are marked PROJECT CONTRACT rather than attributed to WCDF.

### Symmetry adjudication

The valid nontrivial game symmetry is 180° rotation *combined with* Red/White identity swap.
Horizontal and vertical mirrors map playable squares to light squares; diagonal mirrors map king
rows to columns. Colour-only and rotation-only transforms reverse man direction without applying
the corresponding player-role change. BLOCK-003 records why §12.4 D's separate symmetry wording
cannot be implemented truthfully.

## Rule-to-Source-to-Test Matrix

Test names are the permanent Phase 2/3 targets. A listed future test is a traceability commitment,
not a claim that the test already passes; gate evidence records actual execution separately.

| Rule ID | Authority and derivation | Covering test |
|---|---|---|
| R1.1 | WCDF 1.1 and 1.5 — 8×8 board and 32 playable squares | `tests/rules/test_board.py::test_r1_1_board_has_exactly_32_playable_dark_squares` Phase 2 |
| R1.2 | WCDF 1.4–1.5 — orientation and official 1–32 references | `tests/rules/test_board.py::test_r1_2_acf_mapping_matches_frozen_unmirrored_diagram` Phase 2 |
| R1.3 | WCDF 1.4 — single and double corners | `tests/rules/test_board.py::test_r1_3_double_corner_is_on_each_players_right` Phase 2 |
| R1.4 | WCDF 1.8 and 1.11 — twelve men per side on 1–12 and 21–32 | `tests/rules/test_state.py::test_r1_4_initial_position_is_exact_and_contains_only_men` Phase 2 |
| R1.5 | WCDF 1.9 and 1.13 — Red convention and first move | `tests/rules/test_state.py::test_r1_5_red_is_explicit_first_player` Phase 2 |
| R2.1 | WCDF 1.13 — turns alternate | `tests/rules/test_moves.py::test_r2_1_completed_moves_alternate_players` Phase 2 |
| R2.2 | WCDF 1.14–1.21 plus PROJECT CONTRACT §5.2 — full move versus step | `tests/rules/test_moves.py::test_r2_2_multijump_is_one_move_and_many_environment_steps` Phase 2 |
| R3.1 | WCDF 1.15 — men move one vacant diagonal forward | `tests/rules/test_moves.py::test_r3_1_man_simple_moves_are_forward_only` Phase 2 |
| R3.2 | WCDF 1.17 — kings move one vacant diagonal either way | `tests/rules/test_moves.py::test_r3_2_king_simple_moves_forward_and_backward` Phase 2 |
| R3.3 | WCDF 1.15 and 1.17 — adjacent vacant destination, hence no flying king | `tests/rules/test_moves.py::test_r3_3_no_flying_king_or_occupied_destination` Phase 2 |
| R4.1 | WCDF 1.18 and 1.21 — short jump over adjacent enemy to vacant beyond | `tests/rules/test_captures.py::test_r4_1_jump_geometry_and_landing_occupancy` Phase 2 |
| R4.2 | WCDF 1.20 and 1.25.1 — capture and completion are compulsory | `tests/rules/test_captures.py::test_r4_2_capture_is_mandatory_across_the_whole_player` Phase 2 |
| R4.3.1 | WCDF 1.21 — king captures forward or backward | `tests/rules/test_captures.py::test_r4_3_1_king_jumps_forward_and_backward` Phase 2 |
| R4.3.2 | WCDF 1.18 and 1.25.4 — uncrowned man never captures backward | `tests/rules/test_captures.py::test_r4_3_2_man_never_jumps_backward` Phase 2 |
| R4.4 | WCDF 1.19–1.20 — same-piece continuation through the final jump | `tests/rules/test_captures.py::test_r4_4_continuation_is_mandatory_for_the_same_piece` Phase 2 |
| R4.5 | WCDF 1.19–1.20 — sequence-end removal and no repeat jump; BLOCK-002 parity correction | `tests/rules/test_captures.py::test_r4_5_marked_piece_remains_occupied_and_cannot_be_jumped_twice` Phase 2 |
| R4.6 | WCDF 1.20 — player may choose any available jump route | `tests/rules/test_captures.py::test_r4_6_no_majority_capture_rule` Phase 2 |
| R5.1 | WCDF 1.16 — man crowns on reaching the far row and turn completes | `tests/rules/test_promotion.py::test_r5_1_man_promotes_at_a_completed_move` Phase 2 |
| R5.2 | WCDF 1.16, 1.19, and 1.25.7 — crowning ends capture turn | `tests/rules/test_promotion.py::test_r5_2_promotion_ends_jump_sequence_before_a_new_king_jump` Phase 2 |
| R5.3 | DERIVED from WCDF 1.16–1.17 — king is a persistent crowned state unless captured | `tests/rules/test_promotion.py::test_r5_3_king_is_never_demoted` Phase 2 |
| R6.1 | WCDF 1.30 — player with no pieces has no move and loses | `tests/rules/test_terminal.py::test_r6_1_no_pieces_loses` Phase 3 |
| R6.2 | WCDF 1.30 — blocked player with no legal move loses | `tests/rules/test_terminal.py::test_r6_2_stalemate_is_loss` Phase 3 |
| R6.3 | ENGINE VARIANT from WCDF 1.32.2 — automatic per-player no-progress counters | `tests/rules/test_terminal.py::test_r6_3_per_player_40_move_boundary` Phase 3 |
| R6.4 | ENGINE VARIANT from WCDF 1.32.1 — optional automatic arena repetition only | `tests/rules/test_terminal.py::test_r6_4_repetition_only_at_move_boundaries` Phase 3 |
| R6.5 | ENGINE VARIANT — 512-step cap is an explicit training-MDP rule | `tests/rules/test_terminal.py::test_r6_5_511_vs_512_step_boundary` Phase 3 |
| R6.6 | ENGINE VARIANT departing from WCDF 1.32 — autonomous agents cannot agree a draw | `tests/rules/test_terminal.py::test_r6_6_no_draw_by_agreement_api` Phase 3 |
| R6.7 | DERIVED finite-resource termination bound below | `tests/rules/test_rule_traceability.py::test_r6_7_termination_bound_is_derived_and_above_ply_cap` Phase 1 |
| R7.1 | WCDF 1.5 plus WCDF 2017 published scores — 1–32 notation with `-` and `x` | `tests/rules/test_notation.py::TestMoveExamples::test_r7_1_acf_simple_jump_and_multijump_examples` Phase 2 |
| R7.2 | PROJECT CONTRACT derived from R7.1 — parse and format round trip | `tests/rules/test_notation.py::test_r7_2_move_notation_round_trip` Phase 2 |
| R7.3 | PROJECT CONTRACT §5.1 — FEN-like format serializes the complete state | `tests/rules/test_notation.py::test_r7_3_full_midsequence_state_round_trips_exactly` Phase 2 |

## R6.7 Termination Proof Sketch

Define a *resetting move* as a completed move that either advances an uncrowned man or captures at
least one piece, thus resetting the actor's R6.3 counter.

1. Initially, each side has four men at row-distances 7, 6, and 5 from its king row. The sum of
   remaining forward row-distance over both colours is therefore
   `2 × 4 × (7 + 6 + 5) = 144`. Every move by an uncrowned man reduces this non-negative integer by
   at least one; capture jumps reduce it faster, and neither kings nor promotions increase it.
   Consequently no more than 144 completed moves can reset because a man advanced.
2. At most 23 opposing pieces can be removed: a legal capture always leaves the capturing piece on
   the board. Thus no more than 23 completed moves can reset because they captured. Counting moves
   that do both twice gives the deliberately conservative bound `144 + 23 = 167` resetting moves.
3. With no reset, turns alternate and each player's counter advances once per own completed move.
   Within at most `2 × 40 = 80` completed moves both counters reach 40 and R6.3 declares a draw.
   A reset can postpone that event, but the finite reset budget can do so only 167 times. Including
   the final interval gives at most `(167 + 1) × 80 = 13,440` completed checkers moves.
4. A simple move is one environment step. Every jump step captures a distinct piece, so at most 23
   jump steps occur in a game. Since one step for every capturing move is already included among
   completed moves, multi-jump continuations add at most `23 - 1 = 22` extra steps. A safe bound is
   therefore `13,440 + 22 = 13,462` environment steps.

This proves finite termination under R6.3 even without repetition. The configured R6.5 cap of 512
steps is far below 13,462, so it is a substantive additional ENGINE VARIANT that truncates some
otherwise legal games; it is not merely a defensive assertion.

## Verification Status

- WCDF primary text: VERIFIED and hash-pinned.
- R1.1–R7.3 source/classification rows: complete.
- R6.7 arithmetic: executable in `tests/rules/test_rule_traceability.py`.
- BLOCK-001 remains open because only a human can amend the read-only `GOAL.md`; downstream code
  follows the conservative source-correct ENGINE VARIANT label.
