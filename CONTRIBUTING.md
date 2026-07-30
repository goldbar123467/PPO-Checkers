# Contributing

Small, evidence-backed changes are welcome. Open an issue before changing rules, observation/action encoding, PPO semantics, evaluation design, or artifact formats; those boundaries require migration and reproducibility review.

## Development gate

```bash
uv sync --locked --all-groups
npm --prefix web/checkers ci
make check
npm --prefix web/checkers audit --audit-level=moderate
npm --prefix web/checkers test
npm --prefix web/checkers run typecheck
npm --prefix web/checkers run build
```

Add tests for behavior changes. Keep rules symbolic and server-authoritative. Do not update generated metrics by hand or claim strength from training loss. Record config, seed, revision, environment, and artifact hashes for experiments.

## Data and secrets

Do not submit model weights, checkpoints, optimizer state, caches, raw/reviewed datasets, run directories, or credentials. Tiny reviewed fixtures with provenance and compatible licensing are acceptable. Use synthetic fixtures in CI; the production policy is a GitHub Release asset and is not downloaded by tests.

By contributing, you agree that your contribution is licensed under this repository's MIT License.
