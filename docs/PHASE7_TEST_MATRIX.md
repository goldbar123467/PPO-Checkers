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
| S1 | One-step engine trace | shared policy samples only legal actions; action/mask/oracle agreement counters stay zero | GREEN |
| S2 | Forced multijump lane | complete chronology, actor/sign/completion fields, and mid-sequence rollout boundary are exact | GREEN |
| S3 | Terminal/reset lane | terminal reward retained once; next row is a fresh seeded game; colour-role schedule remains balanced | GREEN |
| U1 | Literal permutation/epoch ledger | every trainable row used once per epoch, no opponent row enters policy loss, rollout discarded after configured epochs | GREEN |
| U2 | Directional integration | one complete collect/update changes parameters and increments exact global/update counters | GREEN |
| M1 | Hand metric tables | every §13.2 formula/range/name, including mean per-state normalized entropy and binned calibration | GREEN |
| M2 | Injected fault table | each §13.3 hard alert halts; sustained-alert windows fire only at exact boundaries | GREEN |
| W1 | Fake run plus real offline smoke | exact project/name/tags/config/metadata; monotonic logging steps; every §13.2 key observed | GREEN |
| W2 | Repository byte scan | no committed API-key-shaped string or credential file | GREEN |
| R1 | Checkpoint schema audit | all §12.8 fields present; update-boundary-only atomic save; corrupt/mismatched/untrusted input rejected | GREEN |
| R2 | Forked execution oracle | CPU interrupted/resumed updates `k+1…k+10` bitwise equal, including a serialized mid-capture lane | GREEN |
| R3 | Same-stack RTX fork | actions equal and losses meet `atol=1e-5, rtol=1e-4`; W&B/log counters do not duplicate | GREEN |
| E1 | Tiny offline CLI run | config → collection → PPO → checkpoint reload → local W&B artifact is end-to-end and rerunnable | GREEN |

## Timed Gate 7 matrix

| ID | Immutable evidence | Gate condition | Status |
|---|---|---|---|
| G1 | Three seed run manifests and W&B-offline directories | each records at least 1,800 training seconds with deterministic mode enabled | Pending |
| G2 | Metric-completeness audit over all three histories | every §13.2 key logged; all three masking fault counters exactly zero | Pending |
| G3 | Three 364-game colour-balanced random matches | each score at least 0.90 with W/D/L, game count, and 95% interval reported | Pending |
| G4 | Load validation and resume transcript | final checkpoints load/use; R2/R3 exact-resume evidence remains green | Pending |
| G5 | Consolidated `make check` | static gates, coverage, and all repository tests green | PRELIMINARY GREEN — rerun after timed seeds/report |

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

## Interrupted Seed 0 recovery matrix

| ID | Evidence | Acceptance rule | Status |
|---|---|---|---|
| RC1 | Exact update-170 fixture plus two later records | both orphans preserved verbatim; active prefix ends at logging step 186 | GREEN |
| RC2 | No-orphan and seven-orphan fixtures | idempotent preparation; every later record preserved | GREEN |
| RC3 | Ambiguous, malformed, and hash-drift fixtures | fail before final destination; source bytes unchanged | GREEN |
| RC4 | Interrupted partial destination fixture | stale partial detected and recreated atomically; decision recorded | GREEN |
| RC5 | One-update CPU continuation/audit | update/logging steps contiguous, no duplicates, full reload and RNG restore | GREEN |
| RC6 | Read-only monitor fixtures | partial-tail tolerance, diagnostic labels, lifecycle distinctions, source hashes unchanged | GREEN |
| RC7 | One-update RTX 5070 recovery smoke | optimizer/device, collector, league, RNG, masks, telemetry, time counters pass | GREEN |
| RC8 | Production Seed 0 recovery | 1,800 seconds, powered final evaluation, reload, artifacts, reconciliation | GREEN |

The recovery design and commands are frozen in `docs/PHASE7_RECOVERY.md`. RC1–RC6 are engineering
evidence only; they make no policy-strength claim and do not satisfy G1–G5.

RC7 uses the independently prepared `phase7-a0-seed0-c8207ca-recovery-smoke-002` directory at
commit `dd6abcde9e3b1078d627b6456eed23a293f4ac45`. The one-update run reached update 171 and global
step 1,400,832 with contiguous logging step 188. The live monitor observed `RUNNING` and terminal
`FINISHED`; the post-run audit records 66 CUDA parameters, 132 CUDA Adam moments, 66 finite CPU
scalar steps, restored RNG/full state, stable W&B ID, finite telemetry/metrics, and zero mask
faults. It remains bounded setup evidence only.

RC8 uses the separately prepared `phase7-a0-seed0-c8207ca-recovery-001` run at training commit
`7c9f4dcc0780dece342406fc645b53d4ebd10419`. It completed update 264, 2,162,688 transitions,
1,804.556 measured training seconds, 291 contiguous records, a digest-verified/full-state-reloaded
checkpoint, six 364-game final match groups, measured best response, W&B offline artifact, and zero
aggregate legality/oracle faults. The original source hashes remain unchanged. Seeds 1 and 2 are
still required before the three-seed Gate 7 conclusion.

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

The self-play/persistence increment began with missing-module RED tests (`000105`, `000108`,
`000110`, `000112`). Production-engine collection now retains every lockstep row, checks masks
against the independent oracle before sampling, preserves a rollout cut inside a forced jump,
retains terminal reward before per-lane reset, and serializes cumulative game/value diagnostics.
The literal metric inventory contains 55 names; entropy is averaged per eligible state and the
collapse alert excludes all-`k=1` batches, where a zero diagnostic is not evidence of collapse.
PPO permutations have an exact source-index ledger and rollouts are weakly identity-tracked so a
live object cannot be replayed without retaining every historical batch in memory.

Checkpoints are CPU-portable, update-boundary-only, atomically replaced, SHA-256 accompanied, and
loaded through `torch.load(weights_only=True)`. They include model/Adam/config/trainer/RNG/league/
collector/vector/W&B/schedule/AMP fields and reject digest corruption, config drift, malformed
schemas, and an untrusted pickle global. R2 checkpoints after the first jump of a forced two-jump
move; actions, epoch ledgers, every scalar metric, collector state, model tensors, and all RNG
families reproduce bitwise for updates 2–11 (`000113`). A repository-wide intermediate run passed
872 tests at 95.26% total coverage (`000111`); validation-branch closure remains before Gate 7.

The same fork now passes on the RTX 5070 with identical actions and the declared numeric tolerance;
the first run exposed and permanently fixed a real `cuda` versus `cuda:0` ownership bug (`000114`,
`000115`). A real W&B SDK offline run persists all 55 required names (`000116`). The tiny CLI test
crosses a periodic-evaluation checkpoint boundary, resumes with one W&B ID and exact monotonic W&B
and JSONL counters, performs a measured short-budget best response, emits payoff/rendered-game
tables, logs a versioned artifact, reloads the final checkpoint, and is rerunnable (`000117` and
the consolidated suite).

The five-minute CUDA setup validation completed 49 updates / 401,408 transitions and 305.202
measured training seconds in 316.890 wall seconds. It wrote checkpoints/evaluations every ten
updates, observed all required metrics, retained win/loss/draw game rows, verified its final SHA-256,
and reloaded the checkpoint to select a legal opening action (`000118`). Its two-game final anchor
is a diagnostic only, not powered Gate-7 evidence; `000120` repeats the digest/model/action audit
and labels the setup checkpoint as predating the final exploitability config field. The
post-implementation `make check` passes
format, Ruff, strict mypy, 889 tests with no skips/xfails, 93.88% total line/branch coverage, and the
eight-property fuzz target (`000119`).

Recovery engineering after the interrupted Seed 0 run first passed all 901 tests but failed the
unchanged coverage gate at 90.58%. The adversarial branch audit then closed recovery, lifecycle,
monitor, and telemetry paths. The clean pre-CUDA-smoke `make check` passes formatting, Ruff, strict
mypy, 925 tests, 92.40% total line/branch coverage, and all eight property/fuzz tests. This is setup
validation only; RC7 and the timed Gate 7 evidence remain pending (`logs/iterations/000029.md`).
