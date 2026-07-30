# Training and reproduction

## Experiment classification

This is a final practice-scale PPO self-play experiment for the web opponent, not supervised fine-tuning, preference optimization, or foundation-model pretraining. The falsifiable objective was to complete the frozen 6,144-update seed-0 profile, pause for review at update 1,024, preserve exact resume state, evaluate every 96 updates on fixed color-balanced ballots, and export the strongest fully evaluated persisted checkpoint.

## Frozen configuration

The tracked source of truth is [`configs/checkers-practice.yaml`](../configs/checkers-practice.yaml).

| Variable | Value |
|---|---:|
| Seed | 0 |
| Device / dtype | CUDA / float32 |
| Deterministic algorithms | true |
| Updates | 6,144 |
| Environments × steps | 64 × 128 |
| Transitions per update | 8,192 |
| PPO minibatches × epochs | 8 × 4 |
| Learning rate | 3e-4, linearly annealed |
| Gamma / GAE lambda | 1.0 / 0.95 |
| Clip coefficient | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 → 0.001 over first half |
| Gradient norm cap | 0.5 |
| Target KL | 0.02 |
| Adam epsilon | 1e-5 |
| Maximum plies | 512 |
| Checkpoint cadence | 256 updates |
| Evaluation cadence | 96 updates |
| Evaluation per opponent | 216 ballots × 2 colors = 432 games |
| Self-play pool | FIFO capacity 20; snapshot every 20 updates |

There is no training dataset. Experience is generated online by the rules environment and current-policy self-play. The tracked ballot files are evaluation fixtures, not imitation targets.

## Recorded environment

The selected run used commit `495ff829e15373c3bb5117dd13933b3a8cdfa492` on one NVIDIA GeForce RTX 5070 with 12,227 MiB, WSL2 Linux, CPython 3.12.3, and PyTorch 2.13.0. Package versions and wheel hashes are frozen in `uv.lock`.

The run recorded 77,845.005 seconds of rollout/optimization time and 82,170.568 seconds across the two invocation wall counters. It processed 50,331,648 transitions. Recorded maxima were 11,923 MiB GPU memory, 4,592,087,040 bytes process RSS, 115.28 W, and 60 °C. These measurements are run evidence, not guarantees for another machine.

## Setup validation

Use a clean clone on a CUDA machine. The profile refuses a dirty Git worktree. Preserve at least 35 GB free and stop other VRAM-heavy services.

```bash
uv sync --locked --all-groups
nvidia-smi
make doctor
make check
```

The mandatory practice preflight performs online/offline W&B equivalence, exact split-resume comparison, legality counters, schedule checks, the 864-game evaluation, and batched-versus-sequential arena comparison. It explicitly requires the W&B key in the environment; enter it without putting it in shell history:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY && printf '\n'
export WANDB_API_KEY

PYTHONPATH=src .venv/bin/python scripts/preflight_practice.py \
  --config configs/checkers-practice.yaml \
  --output-dir runs/practice-preflight-reproduction
```

Do not begin the long run unless `preflight_report.json` says `accepted: 1`, the GPU path is real, and the output artifact reloads.

## Exact long-run commands

```bash
run_dir=runs/checkers-practice-seed0-reproduction

PYTHONPATH=src .venv/bin/python scripts/train.py \
  --config configs/checkers-practice.yaml \
  --output-dir "$run_dir"
```

The process deliberately exits at update 1,024 with `status: paused_for_approval`. Review the evaluation, resource metrics, mask counters, checkpoint sidecar, W&B status, and free disk. Resume only the same directory and checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --config configs/checkers-practice.yaml \
  --output-dir "$run_dir" \
  --resume "$run_dir/checkpoints/update-001024.pt"
```

Resume validation rejects changed configuration, incompatible Git provenance, non-contiguous metric history, or a checkpoint outside the prepared recovery boundary. An interrupted run should resume from its last complete SHA-verified checkpoint; never fabricate missing metric rows or silently change hyperparameters.

## Selection and export

Because checkpoints are saved every 256 updates and evaluations every 96, only their intersections are eligible for evidence-backed selection. For the accepted run these were updates 768, 1024, 1536, 2304, 3072, 3840, 4608, 5376, and 6144. Update 4608 had the best recorded Minimax-2 score.

```bash
PYTHONPATH=src .venv/bin/python scripts/export_checkers_policy.py \
  --config configs/checkers-practice.yaml \
  --checkpoint "$run_dir/checkpoints/update-004608.pt" \
  --output models/checkers/policies/checkers-practice-update-004608.pt
```

The exporter validates the full checkpoint and its SHA-256 sidecar, strips optimizer/league/collector/RNG state, reloads the public CPU bundle strictly, then requires exact logits, value, and greedy-action parity on 12 fixed positions.

To rebuild the compact public report from retained authoritative evidence:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_checkers_release_report.py \
  --run-dir "$run_dir" \
  --bundle models/checkers/policies/checkers-practice-update-004608.pt \
  --output reports/checkers_practice_release_v1.json
```

## Reproducibility boundary

Same-stack deterministic action traces and exact resume are tested. Cross-driver, cross-GPU, or universal bitwise identity is not claimed. A one-seed result cannot estimate between-seed variance. Re-running also creates a new W&B run and may differ in total wall time even when actions match.
