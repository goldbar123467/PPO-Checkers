# Phase 7 three-seed A0 baseline analysis

## Outcome

Gate 7 is green. All three deterministic A0 seeds exceeded 1,800 measured PPO training seconds,
retained complete local/W&B-offline evidence, reloaded their final checkpoints, logged every
required metric, produced zero mask correctness faults, and exceeded the predeclared 0.90 random
opponent score threshold in 364 colour-balanced games.

This is a Phase 7 smoke/baseline result. It is not evidence of convergence, generalization, solved
checkers, or Stage-C strength. Periodic two-game evaluations are diagnostic only; all results below
come from the final powered evaluations.

## Controlled experiment

The falsifiable objective was: each of three independent seeds must run for at least 1,800 measured
training seconds under the frozen A0 configuration and independently score at least 0.90 against
random over 364 alternating-colour games. Seed YAMLs differ only in `experiment_id` and `seed`.
Seeds 1 and 2 ran at the same clean commit `ea5f62f4485ca538890e9d6ac25b77e46c69c46b`.
Recovered Seed 0 ran at clean commit `7c9f4dcc0780dece342406fc645b53d4ebd10419`; subsequent changes
before Seeds 1/2 affected recovery reporting and monitor PID identity, not PPO, model, environment,
evaluation, or hyperparameters.

All work ran locally on one NVIDIA GeForce RTX 5070 (12,227 MiB), driver 610.88, Python 3.12.3,
PyTorch 2.13.0+cu130, CUDA 13.0, and cuDNN 9.2. No seeds overlapped.

## Per-seed results

| Seed | Timed seconds | Updates | Transitions | Random W/D/L | Random score [95% CI] | Greedy | Minimax-2 | Tactical | Best response [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,804.556 | 264 | 2,162,688 | 362/2/0 | 0.9973 [0.9846, 0.9995] | 0.9986 | 0.6731 | 0.24 | 0.25 [0.2083, 0.2969] |
| 1 | 1,801.904 | 241 | 1,974,272 | 363/1/0 | 0.9986 [0.9870, 0.9999] | 1.0000 | 0.6923 | 0.32 | 0.25 [0.2083, 0.2969] |
| 2 | 1,800.139 | 248 | 2,031,616 | 363/0/1 | 0.9973 [0.9846, 0.9995] | 1.0000 | 0.7019 | 0.30 | 0.50 [0.4489, 0.5511] |

Each random match used 182 games with the current policy as red and 182 as white. Every seed also
completed 364 games in each of `vs_greedy`, `vs_minimax2`, `current_vs_initial`,
`sampled_vs_random`, and `best_response_vs_frozen`: 6,552 final games overall.

## Across-seed summary

Values are arithmetic mean and sample standard deviation across three independently seeded runs;
the small sample is reported without pretending to estimate a population precisely.

| Metric | Mean | Sample SD | Range |
|---|---:|---:|---:|
| vs random | 0.9977 | 0.0008 | 0.9973–0.9986 |
| vs greedy | 0.9995 | 0.0008 | 0.9986–1.0000 |
| vs minimax-2 | 0.6891 | 0.0147 | 0.6731–0.7019 |
| tactical accuracy | 0.2867 | 0.0416 | 0.24–0.32 |
| best-response proxy | 0.3333 | 0.1443 | 0.25–0.50 |

The random gate is comfortably satisfied and low-variance across these seeds. Performance against
minimax-2 is materially lower than against random/greedy, and tactical accuracy remains only
24–32%. The best-response proxy has visibly larger between-seed spread. These are useful Phase 8
targets; they should not be hidden by the green random-opponent gate.

## Integrity and reproducibility audit

- Total measured training: 5,406.599 seconds; total transitions: 6,168,576.
- Metric histories contain 291/266/273 records with exact logging sequences, exact training update
  sequences, finite numeric values, every required scalar key, and payoff rows for each seed.
- `sample_legality_violations`, `oracle_disagreements`, and `empty_mask_count` sum to exactly zero
  independently for every seed.
- Final checkpoint SHA-256 values are `32c147…03dbf0`, `1c0c05…07829f`, and
  `8610c7…59e4be`. Sidecars match and full reloads restore CUDA model/Adam, counters, collector,
  league, and RNG state.
- Best-response training left each frozen policy hash unchanged. Exploitability remains a declared
  short-budget proxy, not an exact game-theoretic exploitability computation.
- W&B was offline. IDs are `qex4drmv`, `4gflz3ms`, and `it1rtdfv`; completed manifests record local
  versioned checkpoint artifacts. Local checkpoint/JSONL/evaluation/manifests are authoritative;
  cloud synchronization is not claimed.
- The interrupted Seed 0 source and its two orphan metric records remain preserved. Its active
  history begins from the unique full-state update-170 checkpoint boundary, as documented in
  `docs/PHASE7_RECOVERY.md`.

## Gate decision

G1–G5 pass. The post-report `make check` completed with clean formatting, Ruff, and strict mypy;
930 tests passed with 92.27% total line/branch coverage against the unchanged 92% threshold, and
all eight property/fuzz tests passed. The machine-readable companion is
`reports/phase7_gate_report_v1.json`.
