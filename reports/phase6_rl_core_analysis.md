# Phase 6 offline RL-core analysis

## Decision

**Gate 6 is GREEN.** On source revision
`165c72101c71a77317b1848a316b4b4a22c46654`, the complete repository gate passed 760/760
collected tests plus eight additional property tests, with no skips or expected failures. Ruff
formatting/lint and strict mypy passed across 88 source paths. Coverage was 3,842/3,842 statements
and 1,358/1,358 branches (100%). The gate began at 2026-07-28 03:28:11 EDT and ended at
03:30:23 EDT.

This is an **offline engineering verification**, not a learning experiment. It establishes the
declared mathematical, structural, masking, buffering, and same-stack reproducibility contracts.
It does not show that PPO learns checkers, beats an opponent, or generalizes. Those claims require
Phase 7 and Phase 8 evidence.

## Falsifiable objective and authority

The implementation was accepted only if every predeclared row in
`docs/PHASE6_TEST_MATRIX.md` passed and the full repository gate remained green. The standard PPO
clipped surrogate is grounded in Schulman et al.,
[*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). Standard GAE is
grounded in Schulman et al.,
[*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438),
whose equations define the TD residual and exponentially weighted advantage sum. The additional
same-actor/opponent `sigma` transform is project-derived for step-wise two-player play; the cited
GAE paper does **not** validate that extension.

Invalid-action masking follows Huang and Ontañón,
[*A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*](https://arxiv.org/abs/2006.14171),
and the current official PyTorch
[`Categorical`](https://docs.pytorch.org/docs/stable/distributions.html) API. Network and optimizer
behavior use the official PyTorch
[`GroupNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html),
[`clip_grad_norm_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html),
and [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) documentation.
Production checkers legality and terminal outcomes remain traced to primary WCDF clauses in
`docs/RULES.md`.

## Verified components

### Masked categorical

One wrapper is used for collection and updates. It rejects an all-false row, replaces illegal
logits with `torch.finfo(dtype).min` before categorical normalization, and derives samples,
log-probabilities, and entropy from the restricted distribution. Float32 and BF16 cases, including
one legal action, are finite and legal. Illegal logits receive exactly zero gradient. PPO consumes
the legal mask stored at collection; a test mutates the source mask after storage and proves the
stored copy remains authoritative.

### Signed GAE and chronology

Time is the leading dimension, and the complete `(time, environment)` chronology is retained until
after signed GAE. The implementation applies `sigma` to both the next value and recursively
propagated advantage, masks both at a true terminal, and applies the final sign when a rollout ends
mid-capture. Only after GAE are non-trainable opponent transitions filtered from policy/value
optimization views.

The literal four-row T3 oracle has advantages
`[0.1536928, 0.61624, -1.092, 0.6]` and returns
`[0.3536928, 0.51624, -0.792, 1.0]`. The T8 two-lane oracle preserves source indices `[0,3,4]`
after filtering. The conventional all-`sigma=+1` case is bitwise equal to a separate single-agent
GAE loop.

T7 executes three frozen boundary states through the production rules engine and environment. At
every state the expected step is the engine's entire one-element legal set:

| Length | Actors | Signs | Exact targets | Ending |
|---:|---|---|---|---|
| 3 | `W,R,W` | `-,-,-` | `[+1,-1,+1]` | R6.2 no legal move |
| 5 | `W,R,R,R,R` | `-,+,+,+,-` | `[-1,+1,+1,+1,+1]` | four-jump terminal capture sequence |
| 7 | `W,W,R,R,W,R,W` | `+,-,+,-,-,-,-` | `[+1,+1,-1,-1,+1,-1,+1]` | no pieces |

With zero baselines and `gamma=lambda=1`, each target is exactly the final outcome in that
transition actor's frame. Temporarily deleting recursive `sigma` makes all three numerical tests
fail with all-positive targets; restoring it makes them pass.

### Network and PPO

The exact network is an 8-to-64 convolutional stem, six shared GroupNorm residual blocks, and
separate policy/value heads. It contains 470,410 parameters (1.794 MiB of FP32 weights) and no
BatchNorm. T1 fits all 64 independently fixed policy labels (accuracy 1.0); T2 fits 64 fixed scalar
targets (MSE `4.69e-14`). These are capacity/gradient-flow checks, not generalization or PPO-quality
evidence. Every parameter is graph-connected, every major module has nonzero aggregate gradient on
the purpose-built T5 suite, and N7's aliased state pair produces different logits.

PPO uses normalized trainable-only advantages, the stored old log-probabilities/masks, the clipped
policy surrogate, plain value MSE, masked entropy, k3 approximate KL, strict clip fraction, and
global norm clipping before Adam. The independent T3 loss oracle is:

| Quantity | Exact expected value |
|---|---:|
| policy loss | `0.06898048` |
| value MSE | `0.35025398359296` |
| entropy | `0.6677927263741105` |
| k3 KL | `0.02609025383118571` |
| clip fraction | `0.5` |
| total loss | `0.23742954453273896` |

Every scalar agrees within `1e-12`. In T4, positive-advantage selected-action probability increases
and negative-advantage probability decreases monotonically until ratios cross 1.2/0.8; clip
fraction becomes one and the probabilities then plateau.

## Determinism and hardware envelope

One unsigned 64-bit root seed deterministically derives separate Python, NumPy, Torch, CUDA, and
per-environment streams. Each D2/D3 repeat reconstructs the network, optimizer, observations,
masks, and RNG state and freshly samples actions/old log-probabilities for ten genuine updates.

- CPU action and seven-diagnostic loss tuples are bitwise identical.
- On the same RTX 5070/software stack, action sequences are identical and diagnostics pass
  `atol=1e-5, rtol=1e-4`.
- Native CUDA BF16 masking is finite/legal and preserves exact-zero illegal gradients.
- Deterministic algorithms are enabled, cuDNN benchmarking is disabled, and deterministic cuDNN
  mode is enabled.

The reference transcript records RTX 5070 UUID `GPU-1e2f280f-31f2-c69a-233e-55627e1aefaf`,
driver 610.74, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20, and MKL. A separate
ten-update CUDA memory probe measured 73.125 MiB peak allocated and 94.000 MiB peak reserved by
PyTorch, far below the 12,227 MiB device capacity. This does not predict Phase 7 rollout memory.

The determinism claim is deliberately limited to the recorded stack. No cross-machine,
cross-driver, cross-GPU, or bitwise-GPU claim is made. PyTorch's own reproducibility documentation
warns that deterministic settings and results are scoped by operations, hardware, and software.

## Failures and corrections retained

- Initial import tests for each new module failed before implementation (`000066`, `000070`,
  `000075`, `000082`, `000088`).
- Strict quality passes exposed real validation-branch, typing, formatting, and lint gaps; final
  focused modules reached 100% statement/branch coverage.
- The first T8 fixture had a hand-oracle sign error and mislabeled frozen/current row; the corrected
  independent actor-frame derivation is retained in the log trail.
- Two PPO test fixtures accidentally downcast float64 old log-probabilities; production validation
  rejected them, and the fixtures were corrected rather than weakening dtype checks.
- T7's first standalone quality commands exposed formatting/lint and an incorrectly scoped mypy
  invocation; the correct full-source strict invocation passes.
- The post-gate memory probe initially called peak reset before CUDA context initialization. Both
  failed instrumentation attempts (`000098`, `000099`) are retained; the initialized probe passes.

No failed run is used as acceptance evidence.

## Evidence integrity

- Consolidated gate: `logs/gates/phase-6.txt`, SHA-256
  `3a0abe5ea6d27b6e85f736ec959be5aa069ad093a29687dda6ed481afb824e14`.
- Determinism stack: `logs/gates/phase-6-determinism.txt`, SHA-256
  `a36393656f60c9adbc05c1e27a64bbfe13790c8d953b5843b046e464e0009905`.
- CUDA memory: `logs/gates/phase-6-memory.txt`, SHA-256
  `61027290d7317c56f417c2b8a9d32300b996c5a122a15a9cf9ca20ee1b9c0017`.
- Focused RED/correction/GREEN trail: `logs/test-output/000066-*` through `000099-*`.
- Exact oracle inventory: `docs/PHASE6_TEST_MATRIX.md`.

## Remaining uncertainty

Phase 6 verifies mechanics only. There is no self-play checkpoint, W&B run, random-agent win-rate,
resume-complete trainer, policy calibration curve, or learning curve yet. Phase 7 must therefore
begin with a short end-to-end smoke and cannot cite this report as evidence of learned playing
strength.
