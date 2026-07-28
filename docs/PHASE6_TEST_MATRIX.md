# Phase 6 offline-RL test matrix

This matrix freezes the Gate 6 evidence plan before the remaining implementation. A passing test
must use an independent oracle or a falsifiable directional/property criterion; reproducing the
implementation formula in the assertion is insufficient.

## Authority and claim boundary

- The clipped surrogate comes from the original
  [PPO paper](https://arxiv.org/abs/1707.06347).
- Standard single-agent GAE comes from the original
  [GAE paper](https://arxiv.org/abs/1506.02438).
- The two-player `sigma` extension is project-derived. Citation does not validate it; the hand
  calculation, forced-mate trajectories, single-agent equivalence, and colour-swap tests do.
- Categorical API behavior, deterministic-algorithm controls, and gradient-norm clipping follow
  the current official [PyTorch distribution documentation](https://docs.pytorch.org/docs/stable/distributions.html),
  [determinism API](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html),
  and [gradient-clipping API](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).
- D2 is a same-stack CPU bitwise claim. D3 is a same-machine/same-stack GPU tolerance claim. No
  cross-version, cross-machine, or cross-GPU bitwise claim is planned.

## Frozen matrix

| ID | Independent oracle or falsifier | Required scope | Status |
|---|---|---|---|
| A2–A6 | Legal-set membership, exact illegal-gradient zeros, direct restricted softmax | float32, BF16, `k=1`, all-masked failure, used autocast modes | Foundation GREEN |
| T1 | Fixed 64-state labels generated independently of optimizer/model | policy accuracy >0.99; proves capacity, not PPO | Pending |
| T2 | Fixed 64-state scalar targets | value MSE <1e-3; proves capacity, not PPO | Pending |
| T3 | Literal four-row hand table | both sigma signs, deltas, advantages, returns, ratios, clipped objective, total loss within 1e-6 | GAE half GREEN; PPO half pending |
| T4 | Stored old probabilities plus sign-labelled advantages | selected probabilities monotone in the required direction until clipping; clip fraction at exact bounds | Pending |
| T5 | Per-module gradient aggregation and direct illegal-logit probe | every module reached across the suite; illegal logits exactly zero | Distribution probe GREEN; network/PPO pending |
| T6 | Minimal conventional GAE loop with no perspective transform | all `sigma=+1` results bitwise equal | GREEN |
| T7 | Scripted rule-engine forced wins of lengths 3, 5, and 7 | exact actor-relative targets; one multi-jump terminal path and one R6.2 ending | Pending |
| T8 | Hand-interleaved multi-environment chronology | opponent transitions excluded only after GAE; per-environment adjacency; mid-sequence final bootstrap sign | Pending |
| N2–N7 | Structural module inspection plus equal-state batch-invariance probes | exact GroupNorm residual architecture, output shapes/ranges, no BatchNorm, aliasing logits differ | Pending |
| D2 | Run a frozen ten-update fixture twice from full reseeding | CPU loss/action tensors bitwise equal | Pending |
| D3 | Run the same ten-update fixture twice on the RTX 5070 | actions identical; losses `atol=1e-5`, `rtol=1e-4`; BF16 autocast path included if enabled | Pending |

## Foundation evidence

`tests/rl/test_masked_categorical.py` and `tests/rl/test_gae.py` began RED because their modules did
not exist. The first quality run then exposed strict typing, formatting, and missing validation-
branch coverage. After correction, 41 tests pass and both modules have 100% statement and branch
coverage. Evidence is `logs/test-output/000066-*` through `000069-*`.

The signed hand oracle uses `gamma=0.9`, `lambda=0.8`, rewards `[0,0,0,1]`, values
`[0.2,-0.1,0.3,0.4]`, and signs `[+1,-1,-1,+1]`. Its independently calculated advantages are
`[0.1536928,0.61624,-1.092,0.6]` and returns are
`[0.3536928,0.51624,-0.792,1.0]`.

## Negative controls retained

- Any all-false mask row raises before a distribution is constructed.
- Replacing the first hand-oracle `sigma=+1` with `-1` changes the expected recursion.
- Filtering opponent rows before GAE will break the future T8 adjacency oracle.
- Recomputing masks during PPO update will fail a stored-mask mutation test.
- Introducing BatchNorm will fail both structural inspection and batch-composition invariance.
- Omitting either `sigma` occurrence will fail the hand calculation, colour-swap, or forced-mate
  suite.
