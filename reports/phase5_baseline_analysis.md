# Phase 5 powered baseline analysis

## Decision

**Gate 5 is technically GREEN for source revision
`6deefb959cc995517b5bbe3c452610e99058adc8`.** The frozen four-agent population completed all
six unordered matchups at 784 colour-balanced games per matchup (4,704 games total). The
predeclared catastrophic-inversion and depth-1-versus-depth-3 tactical criteria both passed.

This is an internal engineering baseline, not a claim of checkers mastery. No external engine or
sealed suite was queried, and no trained best response yet exists.

## Match results

Score is `(wins + 0.5 * draws) / games` from the first-named agent's perspective. Each matchup has
392 games with the first agent as Red and 392 with it as White. Bounds are 95% Wilson-style score
intervals; draws are fractional observations, so their finite-sample coverage is approximate.

| First agent | Second agent | W-D-L | Score | 95% score interval | Elo difference (95% interval) |
|---|---|---:|---:|---:|---:|
| greedy | random | 432-20-332 | 0.5638 | [0.5288, 0.5981] | +44.6 [+20.1, +69.1] |
| minimax(1) | random | 437-24-323 | 0.5727 | [0.5378, 0.6069] | +50.9 [+26.3, +75.4] |
| minimax(2) | random | 777-4-3 | 0.9936 | [0.9852, 0.9973] | +877.0 [+728.8, +1025.2] |
| minimax(1) | greedy | 386-28-370 | 0.5102 | [0.4752, 0.5451] | +7.1 [-17.2, +31.4] |
| minimax(2) | greedy | 781-0-3 | 0.9962 | [0.9888, 0.9987] | +966.2 [+778.5, +1153.9] |
| minimax(2) | minimax(1) | 781-1-2 | 0.9968 | [0.9898, 0.9990] | +998.0 [+794.2, +1201.8] |

The decisive Gate 5 comparison is minimax(2) versus minimax(1): 0.9968, well above the
predeclared 0.40 catastrophic-inversion floor. This result applies only to the exact seeded agent
implementations, rules configuration, and standard initial position in the archived run.

## Power and interval scope

The experiment froze a smallest worthwhile score difference of 0.05 from a null score of 0.50,
two-sided alpha 0.05, and target power 0.80 before execution. The
[NIST/SEMATECH normal-approximation equation](https://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm)
gives a raw ceiling of 783 games under the conservative Bernoulli-variance model; the budget was
rounded to the next even number, 784, for exact colour balance. The reported approximate power is
0.80074.

NIST separately recommends a continuity correction. It was **not** applied here. Therefore
`power_justified=true` means only that the declared, uncorrected normal approximation reaches the
target; it is not an exact finite-sample guarantee. Every game has distinct injectively derived
pseudorandom streams, but statistical independence and representativeness are modeling
assumptions, not facts proved by seed uniqueness.

The interval implementation follows the score construction associated with
[Wilson's original 1927 article](https://www.tandfonline.com/doi/abs/10.1080/01621459.1927.10502953).
Elo is a descriptive expected-score transform following the
[official FIDE rating convention](https://handbook.fide.com/chapter/B022024); it is not an
independent strength measurement.

## Population diagnostics

The match table is the complete pairwise payoff population and retains W/D/L, not just scores.
The fitted league ratings are conditional on an approximately transitive
[Bradley-Terry paired-comparison model](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091):

| Agent | League Elo | Conditional 95% interval |
|---|---:|---:|
| random | -262.3 | [-290.8, -233.8] |
| greedy | -218.9 | [-247.8, -190.1] |
| minimax(1) | -212.2 | [-241.1, -183.3] |
| minimax(2) | +693.4 | [+614.4, +772.5] |

The directed payoff matrix contains zero strict three-cycles. The fitted model's weighted residual
RMSE is 10.17 Elo and maximum absolute residual is 92.33 Elo. That is below the declared 100-Elo
diagnostic threshold, but this threshold is project-defined and the maximum is close enough that
the league projection should not be read as ground truth. Fixed-anchor scores for minimax(2) are
0.9936 versus random, 0.9962 versus greedy, and 0.9968 versus minimax(1).

The exploitability proxy is `NOT_EVALUATED`: Gate 5 precedes the Phase 7 trained short-budget best
response required by the definition. Substituting another hand-coded policy would answer a
different question. The external anchor is `NOT_AVAILABLE`.

## Tactical check and non-monotonicity

The exact 50-position development suite has SHA-256
`cf0bf4040185dfb229099f9780f988b9650425833c36390f2427c181729ffd01`. Each case is evaluated by a
fresh policy using the declared seed, making the result independent of manifest order.

| Search depth | Solved | Accuracy |
|---|---:|---:|
| minimax(1) | 15 / 50 | 0.30 |
| minimax(2) | 9 / 50 | 0.18 |
| minimax(3) | 50 / 50 | 1.00 |

Depth 3 is a strict superset of depth 1 and gains 35 cases, so the predeclared tactical gate passes.
The depth-1-to-depth-2 regression is also real and is reported rather than averaged away.

**Diagnosis (hypothesis):** the regression is consistent with horizon effects in the simple,
non-quiescent material evaluator. It is evidence about this evaluator/search combination, not by
itself evidence of a rules-engine defect. The suite was programmatically verified but selected
during construction for depth-3 success; it is neither an unbiased tactical sample nor a sealed
strength test.

## Artifact and execution evidence

- Config: `configs/checkers-baselines-v1.yaml`, SHA-256
  `f111835cbc0bcd576bd9e7d7e4c0bbde739bc7e1b3ee26f0e641b619cd0000e0`.
- Goal: `GOAL.md`, SHA-256
  `dab54331c088a201c1e43e0743866e1780aa84e3b0868b0b7cce34271c17660f`.
- Raw replay-complete archive: `reports/phase5_baseline_games_v1.json.gz`, SHA-256
  `c5ca9d1d446a4462932b80bcc8570b5a0a778c38261f3faa46f08751d19d00b4`.
- Machine report: `reports/phase5_baseline_report_v1.json`, SHA-256
  `2a866255e9ed86b771d130bb8e9a728ce0289d94c7f9a6fb4a1bb543594504c4`.
- First-run log: `logs/gates/phase-5-baseline-run.txt`; checkpoint-resume log:
  `logs/gates/phase-5-baseline-run-resume.txt`.

All six checkpoint files were reloaded in a second invocation. Every comparison reported
`status=resumed`, and the regenerated raw archive and report were byte-for-byte identical to the
first invocation. A first resume attempt refused to start because the first-run progress log made
the worktree dirty; moving that generated log out of the worktree restored the runner's deliberate
clean-tree precondition. This refusal changed no checkpoint or result.

Execution used Python 3.12.3, NumPy 2.5.1, Gymnasium 1.3.0, and PyYAML 6.0.3. Fixed search policies
have no tensor backend, so games ran on CPU. Before, during, and after the run, `nvidia-smi`
identified an NVIDIA GeForce RTX 5070 with 12,227 MiB total, 1,357 MiB used, 10,587 MiB free, and
driver 610.74; no CPU fallback is being hidden because CPU was the declared execution device.

From a clean checkout of the recorded source revision, reproduction is:

```bash
.venv-train/bin/python scripts/evaluate_baselines.py \
  --progress-log logs/gates/phase-5-baseline-run.txt
```

Existing checkpoints must either match all source/config/goal identities or be moved aside so the
runner recomputes them. The runner validates every saved checkpoint immediately, reloads the raw
archive, reloads the report, and exits nonzero unless the technical gate passes.

## Claim boundary

Facts supported by the archived artifacts are the exact W/D/L records, actions, seeds, outcomes,
hashes, tactical selections, and deterministic resume result. Statistical intervals, power, league
Elo, and the horizon-effect diagnosis depend on the assumptions stated above. Every arena game
started from the standard initial position; no opening-ballot distribution was tested. The
[WCDF rules publication](https://wcdf.net/rules/rules_of_checkers_english.pdf) is the primary game
rules authority used by the engine.

`$SEALED_EVAL_DIR` was not supplied or queried. Sealed-suite strength is therefore
`NOT_EVALUATED`, which is the required non-fabricated outcome at this phase.
