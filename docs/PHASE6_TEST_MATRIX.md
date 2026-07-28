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
| T1 | Fixed 64-state labels generated independently of optimizer/model | policy accuracy >0.99; proves capacity, not PPO | GREEN (1.0000) |
| T2 | Fixed 64-state scalar targets | value MSE <1e-3; proves capacity, not PPO | GREEN (4.69e-14) |
| T3 | Literal four-row hand table | both sigma signs, deltas, advantages, returns, ratios, clipped objective, total loss within 1e-6 | GREEN (all scalars within 1e-12) |
| T4 | Stored old probabilities plus sign-labelled advantages | selected probabilities monotone in the required direction until clipping; clip fraction at exact bounds | GREEN |
| T5 | Per-module gradient aggregation and direct illegal-logit probe | every module reached across the suite; illegal logits exactly zero | GREEN |
| T6 | Minimal conventional GAE loop with no perspective transform | all `sigma=+1` results bitwise equal | GREEN |
| T7 | Scripted rule-engine forced wins of lengths 3, 5, and 7 | exact actor-relative targets; one multi-jump terminal path and one R6.2 ending | GREEN |
| T8 | Hand-interleaved multi-environment chronology | opponent transitions excluded only after GAE; per-environment adjacency; mid-sequence final bootstrap sign | GREEN |
| N2–N7 | Structural module inspection plus equal-state batch-invariance probes | exact GroupNorm residual architecture, output shapes/ranges, no BatchNorm, aliasing logits differ | GREEN |
| D2 | Run a frozen ten-update fixture twice from full reseeding | CPU loss/action tensors bitwise equal | GREEN (exact tuple equality) |
| D3 | Run the same ten-update fixture twice on the RTX 5070 | actions identical; losses `atol=1e-5`, `rtol=1e-4`; BF16 path included | GREEN |

## Foundation evidence

`tests/rl/test_masked_categorical.py` and `tests/rl/test_gae.py` began RED because their modules did
not exist. The first quality run then exposed strict typing, formatting, and missing validation-
branch coverage. After correction, 41 tests pass and both modules have 100% statement and branch
coverage. Evidence is `logs/test-output/000066-*` through `000069-*`.

The signed hand oracle uses `gamma=0.9`, `lambda=0.8`, rewards `[0,0,0,1]`, values
`[0.2,-0.1,0.3,0.4]`, and signs `[+1,-1,-1,+1]`. Its independently calculated advantages are
`[0.1536928,0.61624,-1.092,0.6]` and returns are
`[0.3536928,0.51624,-0.792,1.0]`.

The T8 oracle interleaves two stable environment lanes over three vector steps, including a
non-trainable middle transition. Full-chronology advantages are computed before filtering; the
policy view contains exact flattened source indices `[0,3,4]`. A separate rollout ending during a
capture sequence proves that `sigma=+1` transforms the stored bootstrap in the actor's unchanged
frame. The buffer has 100% statement/branch coverage in `logs/test-output/000074-*`.

The exact network contains 470,410 parameters (1.794 MiB of FP32 weights). With frozen trunk
features, its policy head reaches 64/64 fixed-action accuracy and its value head reaches MSE
`4.69e-14` on 64 fixed scalar targets. These are capacity checks, not PPO evidence. All parameters
are graph-connected, every stem/block/head has nonzero aggregate gradient on the T5 batch, and
masked logits retain exact-zero gradients. Evidence is `logs/test-output/000080-*` and `000081-*`.

GroupNorm has no cross-sample statistics, but a convolution applied at batch size 1 versus 4 can
use a different accumulation order. Batch-composition outputs are therefore checked at D3's
`atol=1e-5, rtol=1e-4`, while train versus eval mode on the identical input is bitwise equal.

The literal T3 PPO oracle uses chosen-action ratios `[1.1,0.9,1.3,0.7]` and independently frozen
expected policy loss `0.06898048`, value MSE `0.35025398359296`, entropy
`0.6677927263741105`, k3 KL `0.02609025383118571`, clip fraction `0.5`, and total loss
`0.23742954453273896`. Every scalar matches within `1e-12`. T4 repeatedly optimizes two independent
rows and verifies positive-advantage probability never decreases, negative-advantage probability
never increases, both cross their 1.2/0.8 ratio bounds, clip fraction reaches 1, and both then
plateau. Evidence is `logs/test-output/000087-*`.

The D2/D3 fixture performs ten genuine Adam updates and freshly samples both the current action and
its stored behaviour log-probability on every update. Each repeat reconstructs the network,
optimizer, observations, masks, and all RNG streams from the root seed. CPU action/loss traces are
bitwise identical; same-machine RTX 5070 action traces are identical and all loss diagnostics pass
`atol=1e-5, rtol=1e-4`. A native CUDA BF16 distribution also proves finite entropy, legal samples,
and exact-zero illegal gradients. The transcript pins Python 3.12.3, PyTorch 2.13.0+cu130, CUDA
13.0, cuDNN 9.20, MKL, driver 610.74, GPU UUID/model, and enabled deterministic flags in
`logs/gates/phase-6-determinism.txt`; focused strict coverage is 100% in `000091-*`.

T7 freezes three states discovered by deterministic search, then relies only on the production
rules/environment path during the test. At every state, the next expected ACF step is the engine's
entire one-element legal set. The length-3 path has actors `W,R,W`, signs `-,-,-`, ends by R6.2,
and has exact targets `[+1,-1,+1]`. The distinct length-5 path has actors `W,R,R,R,R`, signs
`-,+,+,+,-`, and its final four steps are one terminal capture sequence; targets are
`[-1,+1,+1,+1,+1]`. The length-7 path has signs `+,-,+,-,-,-,-` and targets
`[+1,+1,-1,-1,+1,-1,+1]`. These are literal actor-relative outcomes with zero baselines and
`gamma=lambda=1`. Removing the recursive `sigma` makes all three target tests fail (`000096-*`),
and the restored suite passes exactly (`000097-*`).

## Negative controls retained

- Any all-false mask row raises before a distribution is constructed.
- Replacing the first hand-oracle `sigma=+1` with `-1` changes the expected recursion.
- Filtering opponent rows before GAE breaks the T8 adjacency oracle.
- Recomputing masks during PPO update will fail a stored-mask mutation test.
- Introducing BatchNorm will fail both structural inspection and batch-composition invariance.
- Omitting either `sigma` occurrence will fail the hand calculation, colour-swap, or forced-mate
  suite.
