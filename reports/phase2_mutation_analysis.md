# Phase 2 Rules Mutation Analysis

Classification: Phase 2 correctness baseline and mutation gate.

## Result

The complete Mutmut 3.6.0 population for `src/checkers/rules/` contained 968 mutants. The final
run produced 927 killed, 39 survived, two timeouts, and no untested, skipped, suspicious,
interrupted, or segfaulting mutants. The conservative score counts only killed mutants:

`927 / 968 = 95.7644628099%`

This exceeds the binding 85% threshold without treating timeouts as kills. If the two detected
infinite-loop timeouts are included, the detected score is `929 / 968 = 95.9710743802%`.

Authoritative artifacts:

- Machine-readable counters: `reports/phase2_mutation_stats.json`, SHA-256
  `9780cf78d57845630759a31c80d5b7b8216c8edbbc917803145ec15b50b123a9`.
- Full run transcript: `logs/test-output/000015-mutmut-full-final.txt`.
- Non-killed mutant list: `logs/test-output/000015-mutmut-results.txt`.
- Pinned tool: `mutmut==3.6.0`; configuration is `[tool.mutmut]` in `pyproject.toml`.
- Tool authority: [Mutmut documentation](https://mutmut.readthedocs.io/en/latest/).

The timeout multiplier is 0.5. The two timed-out mutations replace the bit-iterator's progress
operation with a non-progressing assignment and therefore loop forever. The selected clean rules
suite finishes in seconds; the resulting per-function timeout still gives more than 30× observed
clean runtime to the slowest associated function. The initial unsafe default-timeout observation
is preserved in `logs/test-output/000012-mutmut-full-1.txt` rather than hidden.

## Five Mandatory Semantic Challenges

Mutmut did not generate faithful forms of every semantic mutation required by `GOAL.md` §12.4 C.
For example, deleting `promotion_ends_move` does not itself expose a backward king continuation
because the intermediate piece is still represented as a man. A separate isolated challenge
runner therefore applies the exact five defects and runs the permanent rule test for each.

`reports/phase2_rule_mutation_challenges.json` records an unmodified five-test baseline (exit 0),
source and mutated-source hashes, and these five genuine test failures (each exit 1):

| Injected defect | Permanent killing test |
|---|---|
| Allow backward man jumps | `test_r4_3_2_man_never_jumps_backward` |
| Make captures optional | `test_r4_2_capture_is_mandatory_across_the_whole_player` |
| Remove captured pieces immediately | `test_r4_5_marked_piece_remains_occupied_and_cannot_be_jumped_twice` |
| Permit continuation after promotion | `test_r5_2_promotion_ends_jump_sequence_before_a_new_king_jump` |
| Add a majority-capture rule | `test_r4_6_no_majority_capture_rule` |

The integration test `tests/integration/test_rule_mutation_challenges.py` reruns this proof in a
fresh temporary tree and refuses to count collection/configuration errors as kills.

## Survivor Audit

The 39 survivors are retained and disclosed. Most are equivalent or contract-message mutations:
typing-only `cast()` target changes; parity-equivalent `(row + column) % 2` versus
`(row - column) % 2`; sort-key sentinels that cannot affect ordering because a short jump's
origin/destination uniquely determine its midpoint; symmetric direction iteration followed by a
sort; and exception-message-only changes. Two `promotion_ends_move` mutants are implementation-
equivalent for the representation reason above and are covered by the faithful semantic challenge.

The remainder affect diagnostic metadata or differential-run accounting rather than move
legality. Targeted assertions added during the audit killed 53 previous survivors, including
asymmetric no-progress counters, ply accounting, transition metadata, capture promotion with an
existing king, digest field separation, exact BFS counts, and hexadecimal state parsing. No
survivor is used to waive the score denominator, and no survivor is claimed killed.
