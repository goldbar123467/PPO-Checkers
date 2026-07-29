# Evaluation methodology

## What is measured

Score is `(wins + 0.5 × draws) / games` from the neural policy's perspective. Every reported match alternates colors exactly. Approximate 95% score intervals apply the Wilson-style formula to the bounded win/draw/loss score; with fractional draw points, coverage is approximate rather than exact.

The practice evaluation uses 216 fixed opening ballots derived from 302 source sequences. Each policy/opponent pair plays every ballot twice, swapping colors, for 432 games. The ballot artifact SHA-256 is `9b7c1db4c70c4b39140220fccc5eca4d2d891b3176eb4d623550a304277a365b`.

## Opponents

- **Random** samples a legal action. It is a basic sanity floor.
- **Minimax-2** is the project's deterministic depth-two, material-based baseline. It is much stronger than the project's random/greedy hand baselines in the powered Phase 5 arena, but shallow search and horizon effects make it an imperfect strength reference.
- **Human** evaluation has not been run systematically.

Greedy versus sampled labels in the website describe action selection from the neural logits. They are not separate networks and do not correspond to Minimax depth.

## Checkpoint selection

The selected update is the maximum Minimax-2 score among persisted checkpoints that coincide with a full periodic evaluation. Update 4608 won that comparison at 0.9005. The final checkpoint at 6144 scored 0.8611.

This procedure has selection bias: the same ballot suite was used repeatedly during training and to choose the deployment checkpoint. The interval at update 4608 describes game sampling under that evaluation design; it does not correct for repeated checkpoint selection. A future claim about generalization needs a sealed opening set and preferably independent engine/human opponents.

## Separate evidence classes

| Evidence | Appropriate conclusion | Inappropriate conclusion |
|---|---|---|
| PPO loss / entropy / value diagnostics | Optimization remained finite and observable | The model is strong |
| Zero legality/oracle counters | Collected actions agreed with the symbolic constraints | The policy learned all tactics |
| Random match | The policy cleared a weak sanity baseline | Human-level play |
| Minimax-2 match | Relative performance against this exact project baseline | Standard rating, expert level, solved checkers |
| Browser QA | The production model can complete a real exchange | Statistical playing strength |

## Baseline and variance context

Before the long practice run, three independently seeded 30-minute A0 baselines scored 0.6731, 0.6923, and 0.7019 against Minimax-2 (mean 0.6891, sample SD 0.0147). Those runs validated the training system and show that seed variance exists, but they are different budgets/checkpoints and must not be pooled with the deployed practice model.

## Remaining evaluation work

- Freeze a held-out ballot suite before choosing the next candidate.
- Evaluate multiple full-budget seeds and report between-seed uncertainty.
- Add an independently implemented stronger engine and fixed compute budgets.
- Blind human games and record player experience, colors, disconnects, and adjudication rules.
- Keep automated metrics, model-based judgments, and human assessment in separate tables.
