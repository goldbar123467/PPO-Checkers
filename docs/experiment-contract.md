# PPO Checkers experiment contract

Status: frozen public contract for the reproducible v1 training and evaluation pipeline.

## Objective and hypothesis

The engineering objective is to train, evaluate, export, and serve a compact policy/value network
for American Checkers using PPO self-play on one consumer NVIDIA GPU.

The falsifiable hypothesis is that the trained neural policy will play only legal moves and score
materially above the project's deterministic depth-two Minimax baseline under a fixed,
colour-balanced opening-ballot evaluation. Training loss alone does not satisfy this hypothesis.

## Rules and environment boundary

- `src/checkers/rules` is the symbolic authority for board state, mandatory captures,
  multi-jumps, promotion, terminal outcomes, repetition, and the no-progress rule.
- Rules provenance and engine variants are recorded in [RULES.md](RULES.md).
- The policy never invents legal moves. A Boolean mask restricts its 128 action logits to the
  legal action set produced by the symbolic engine.
- Observations are actor-canonical float32 tensors with shape `8 × 8 × 8`.
- Environment rewards are terminal-only and actor-relative: win `+1`, loss `-1`, draw `0`.
- The configured maximum is 512 environment steps; repetition draws are enabled for the practice
  experiment.

## Model and optimization boundary

- The model is a shared 64-channel residual trunk with six GroupNorm residual blocks, a
  128-logit policy head, and a bounded scalar value head.
- The v1 network has 470,410 parameters.
- PPO uses clipped policy loss, unclipped value MSE, an entropy bonus, generalized advantage
  estimation, gradient clipping, and deterministic seeded minibatch permutations.
- Opponent transitions remain available to the two-player advantage calculation but are excluded
  from the current policy's optimization rows.
- Configuration, optimizer, league, rollout, trainer, and RNG state must be recoverable from a
  full checkpoint. A resume must not silently reset any of them.
- The implementation-detail rationale is frozen in [PPO_CHECKLIST.md](PPO_CHECKLIST.md).

## Reproducibility boundary

- The accepted full profile is `configs/checkers-practice.yaml`.
- A run records the Git revision, configuration hash, seed, Python/package versions, GPU identity,
  metrics, resource telemetry, checkpoint hashes, and runtime state.
- The practice run deliberately pauses at update 1024 for a manual resource and evaluation review
  before resuming from that exact checkpoint.
- Checkpoints, model weights, W&B state, run directories, credentials, and caches are excluded
  from Git.
- Generated evaluation fixtures record provenance, licensing, source hashes, review status, split,
  and content digests.

This document replaces the private build-time specification as the stable public contract hashed
by new validation artifacts. Existing v1 release reports retain their original immutable schema
and digests.

## Evaluation boundary

- Report score as `(wins + 0.5 × draws) / games`.
- Alternate colours exactly.
- The primary v1 suite uses 216 fixed opening ballots twice, once from each colour, for 432 games
  per opponent.
- Random play is a sanity floor. Deterministic depth-two Minimax is an internal comparison, not a
  standard rating or expert-strength proxy.
- Checkpoint selection uses only persisted checkpoints with complete periodic evaluation.
- Preserve adverse results. The final checkpoint is not assumed to be the best checkpoint.
- Keep automated metrics, human results, and model-based judgments separate.

## Artifact and deployment boundary

- Validate an exported model-only bundle by strict reload and inference parity against its source
  checkpoint before release.
- Publish the bundle with a SHA-256 sidecar and model card; do not place weights in Git.
- The Python service is authoritative for rules and inference. The browser renders state and sends
  human move intent; it does not contain a second rules engine.
- Production binds the Python service to loopback behind Caddy. Games are ephemeral and no account
  or persistent user-data system exists in v1.

## Claim limitations

The v1 evidence may support claims about legal play and performance against the declared internal
baselines. It does not support claims of solved checkers, expert human strength, a standard Elo,
generalization to sealed openings, or low between-seed variance. Those require independent
opponents, sealed evaluation data, multiple full-budget seeds, and systematic human games.
