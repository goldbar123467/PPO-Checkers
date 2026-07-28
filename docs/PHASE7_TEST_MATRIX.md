# Phase 7 self-play, logging, and resume test matrix

This matrix freezes Gate 7 before trainer implementation. Focused tests may use tiny configurations;
the timed gate rows require immutable real-run artifacts and cannot be satisfied by mocks.

## Interpretation and claim boundary

- Gate 7's “30-minute smoke on 3 seeds” is interpreted conservatively as at least 1,800 recorded
  training seconds **per seed**, not 30 minutes divided among seeds. The separate §12.11
  approximately-five-minute `make smoke` remains a developer preflight.
- A0 current-policy self-play is the Phase 7 baseline. A1–A3 are mandatory Stage-B experiments in
  Phase 8, so Phase 7 must make league state resumable without selecting a winning arm early.
- The learned-policy random gate uses a predeclared two-sided normal-approximation plan with
  `null_score=0.85`, `alternative_score=0.90`, `alpha=0.05`, and power 0.80: 364 games, rounded to
  an even colour-balanced schedule. Gate success requires observed score at least 0.90; the interval
  is reported but is not silently treated as an exact test of a composite `score >= 0.90` claim.
- Every runtime test/smoke is offline. W&B offline files are local evidence; they are not proof of
  successful cloud synchronization.
- Phase 7 is a smoke/baseline. No convergence, generalization, or Stage-C strength claim follows.

## Frozen focused matrix

| ID | Independent oracle or falsifier | Required result | Status |
|---|---|---|---|
| C1 | Literal valid/invalid configuration table | frozen `RunConfig`; all divisibility/domain/device/timing failures raise | GREEN |
| C2 | Hand schedule endpoints | LR reaches exactly start/zero; entropy reaches start/end at declared fraction; config never mutates | GREEN |
| L1 | Hand FIFO sequence | initial snapshot pinned; capacity 20; deterministic A0/A1/A2/A3 selection; snapshot tensors cloned | GREEN |
| S1 | One-step engine trace | shared policy samples only legal actions; action/mask/oracle agreement counters stay zero | Pending |
| S2 | Forced multijump lane | complete chronology, actor/sign/completion fields, and mid-sequence rollout boundary are exact | Pending |
| S3 | Terminal/reset lane | terminal reward retained once; next row is a fresh seeded game; colour-role schedule remains balanced | Pending |
| U1 | Literal permutation/epoch ledger | every trainable row used once per epoch, no opponent row enters policy loss, rollout discarded after configured epochs | Pending |
| U2 | Directional integration | one complete collect/update changes parameters and increments exact global/update counters | Pending |
| M1 | Hand metric tables | every §13.2 formula/range/name, including mean per-state normalized entropy and binned calibration | Pending |
| M2 | Injected fault table | each §13.3 hard alert halts; sustained-alert windows fire only at exact boundaries | Pending |
| W1 | Fake run plus real offline smoke | exact project/name/tags/config/metadata; monotonic logging steps; every §13.2 key observed | Pending |
| W2 | Repository byte scan | no committed API-key-shaped string or credential file | Pending |
| R1 | Checkpoint schema audit | all §12.8 fields present; update-boundary-only atomic save; corrupt/mismatched/untrusted input rejected | Pending |
| R2 | Forked execution oracle | CPU interrupted/resumed updates `k+1…k+10` bitwise equal, including a serialized mid-capture lane | Pending |
| R3 | Same-stack RTX fork | actions equal and losses meet `atol=1e-5, rtol=1e-4`; W&B/log counters do not duplicate | Pending |
| E1 | Tiny offline CLI run | config → collection → PPO → checkpoint reload → local W&B artifact is end-to-end and rerunnable | Pending |

## Timed Gate 7 matrix

| ID | Immutable evidence | Gate condition | Status |
|---|---|---|---|
| G1 | Three seed run manifests and W&B-offline directories | each records at least 1,800 training seconds with deterministic mode enabled | Pending |
| G2 | Metric-completeness audit over all three histories | every §13.2 key logged; all three masking fault counters exactly zero | Pending |
| G3 | Three 364-game colour-balanced random matches | each score at least 0.90 with W/D/L, game count, and 95% interval reported | Pending |
| G4 | Load validation and resume transcript | final checkpoints load/use; R2/R3 exact-resume evidence remains green | Pending |
| G5 | Consolidated `make check` | static gates, coverage, and all repository tests green | Pending |

## Primary implementation sources

- PPO: <https://arxiv.org/abs/1707.06347>
- GAE: <https://arxiv.org/abs/1506.02438>
- Invalid-action masking: <https://arxiv.org/abs/2006.14171>
- PyTorch reproducibility: <https://docs.pytorch.org/docs/stable/notes/randomness.html>
- PyTorch module/checkpoint state: <https://docs.pytorch.org/docs/stable/notes/modules.html>
- PyTorch `torch.load` trust and device guidance:
  <https://docs.pytorch.org/docs/stable/generated/torch.load.html>
- W&B offline setup: <https://docs.wandb.ai/models/ref/cli/wandb-init>
- W&B run resume: <https://docs.wandb.ai/models/runs/resuming>
- W&B tables: <https://docs.wandb.ai/models/tables/log_tables>

## Retained negative controls

- Recomputing a legal mask after stepping must fail the stored-mask trace.
- Filtering historical-opponent rows before GAE must fail chronology.
- Omitting any RNG family, vector snapshot, league snapshot, or logging counter must fail R1/R2.
- Reusing a rollout after `update_epochs` must fail explicitly.
- Logging ratio-of-means normalized entropy must fail M1's heterogeneous-`k` oracle.
- Any shortened seed, missing metric, nonzero mask counter, or underpowered/odd-colour arena cannot
  satisfy the timed gate.

## Foundation evidence

The import suite began RED because `checkers.config`, `schedules`, `trainer_state`, and `rl.league`
did not exist (`000100-*`). The first behavioral run exposed two independent issues: a test built
invalid YAML text for Python `None`, and the entropy endpoint accumulated a sub-ULP subtraction
residual rather than returning the configured literal endpoint (`000101-*`). Both were corrected.
The first strict audit then exposed formatting/lint, dynamic-YAML typing, NumPy-state typing, and
uncovered validation branches (`000103-*`). The final gate passes 83 tests, strict Ruff/mypy, and
100% of 453 statements plus 176 branches (`000104-*`).

The immutable config enforces whole rollouts, exact minibatch divisibility, even evaluation games,
offline/disabled W&B only, and all numeric/type domains. Trainer state advances only at a complete
rollout boundary. RNG snapshots cover Python, NumPy, Torch CPU, every CUDA device, opponent
selection, minibatch permutation, and stable per-environment streams. League tensors are cloned to
CPU on every boundary; the initial policy remains pinned while unpinned history is FIFO-evicted.
