# Phase 7 interrupted-run recovery protocol

This protocol applies to the Seed 0 interruption after update 172 was logged but only update 170
was checkpointed. It is recovery engineering, not an algorithm, model, environment, evaluation, or
hyperparameter change.

## Source and trust boundary

The original run directory is read-only recovery input. The recovery tool verifies the checkpoint
SHA-256 sidecar with the weights-only checkpoint loader, reconstructs the model, optimizer, league,
collector, trainer, and RNG records, and compares the checkpoint counters to every JSONL record.
It refuses nested/in-place destinations, source hash drift, malformed JSONL, non-contiguous logging,
duplicate logical records, regressing counters, non-finite metrics, and a non-unique boundary.

For the known interruption, checkpoint logging step 187 uniquely selects source line 187:
`periodic_evaluation:update-170:logging-step-186`. Source lines 188 and 189 are later training
records for updates 171 and 172. They remain useful diagnostics but cannot be active training
history because no corresponding model, Adam, league, collector, trainer, or RNG state survived.

## Preparation

Prepare a sibling recovery directory with:

```bash
.venv-train/bin/python scripts/recover_checkers_run.py \
  --source-run runs/checkers-ppo/phase7-a0-seed0-c8207ca \
  --checkpoint runs/checkers-ppo/phase7-a0-seed0-c8207ca/checkpoints/update-000170.pt \
  --output-dir runs/checkers-ppo/phase7-a0-seed0-c8207ca-recovery-001
```

Preparation uses a sibling partial directory and one atomic directory rename. A rerun verifies an
existing completed preparation. A stale partial directory is detected, discarded only within the
exact destination-specific partial path, and recorded in the final manifest. Source checkpoint and
metrics hashes are checked before analysis, before/during copies, and after materialization.

The result contains the exact checkpoint-aligned prefix as active `metrics.jsonl`, the untouched
suffix as `recovery/orphaned-metrics.jsonl`, copied checkpoint/config/evaluations, source hash
records, a narrative report, and `CHECKERS_PPO_RECOVERY_1` machine manifest. The manifest records
byte offsets, line numbers, logical identifiers, per-record hashes, provenance, host/CUDA/software
information, every copied/generated/transformed file, and the explicit exclusion rationale.

## Bounded smoke and production separation

The CUDA smoke is prepared into its own sibling directory from the same immutable update-170
source. It runs exactly one update, then:

```bash
.venv-train/bin/python scripts/audit_recovery_smoke.py \
  --run-dir runs/checkers-ppo/phase7-a0-seed0-c8207ca-recovery-smoke \
  --expected-updates 1
```

The audit requires a byte-identical aligned prefix, update 171, contiguous logging step 187, a
monotonic transition/time boundary, same W&B ID, a fully reloadable checkpoint, restored RNG,
collector/league/trainer agreement, optimizer parameters and moment tensors on the configured
device, finite metrics, host/GPU telemetry, and zero legality/oracle faults. Its output is
explicitly classified as a bounded recovery smoke and is never merged into the production
recovery run.

For standard non-capturable Adam, parameters plus first/second-moment tensors must be on the
configured training device, while each finite scalar `step` tensor may remain on CPU according to
PyTorch's optimizer-state placement policy. The audit records both classes and their devices.

After smoke acceptance, production is prepared again from the original update-170 inputs and
resumed without `--max-updates`. The trainer revalidates source hashes, commit/working-tree state,
destination/checkpoint provenance, active-prefix hash, W&B identity, and JSONL next step before it
initializes W&B or takes an environment action.

## Monitoring and logging

The monitor is read only:

```bash
.venv-train/bin/python scripts/monitor_run.py \
  --run-dir runs/checkers-ppo/phase7-a0-seed0-c8207ca-recovery-001
```

It reads atomic local artifacts and process/NVIDIA telemetry, tolerates a partial final JSONL line,
requires both checkpoint and sidecar before calling a checkpoint durable, labels periodic
evaluations `DIAGNOSTIC_ONLY`, and distinguishes running, idle/waiting, stopped, crashed, and
finished lifecycle states. Unsupported sensors and unrecorded reward mean are displayed as `N/A`.
New lifecycle records bind a PID to its exact Linux boot-relative process start tick from `/proc`,
so wall-clock corrections cannot create a false crash or accidentally match a reused PID. Legacy
records fall back to a narrow timestamp comparison.

Training appends CPU, RAM, process, disk, GPU utilization/memory/temperature/power/clock telemetry
to both local JSONL and the stable offline W&B continuation. Recovery hashes, commits, update, and
source W&B identity are W&B summary fields without local absolute paths. Local checkpoint, JSONL,
recovery, evaluation, and manifest artifacts remain authoritative.

## Acceptance boundary

A prepared directory is not a resumed run. A passing smoke is not a timed baseline. Seed 0 remains
incomplete until 1,800 measured training seconds, final checkpoint/reload, powered 364-game
evaluation, best-response proxy, manifest, W&B artifact, local reconciliation, and repository-wide
quality gate all pass. Seeds 1 and 2 remain prohibited until that Seed 0 boundary is documented.

Seed 0 met this boundary in `phase7-a0-seed0-c8207ca-recovery-001`: update 264, 2,162,688
transitions, 1,804.556 measured seconds, final checkpoint SHA-256
`32c147481d3a8507ee4d0e10b643eeb038e2eeb861385e1de7038afed703dbf0`, and six complete
364-game final match groups. RC8 is accepted; this does not by itself complete the three-seed gate.
