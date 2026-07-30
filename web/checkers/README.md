# Red House Checkers Harness

This Vite + React + TypeScript client plays American checkers against the repository's trained PPO policy. All inference and rules execute in the local Python process; the browser receives only position snapshots, legal human moves, and non-secret model provenance.

## Prerequisites

- The repository `.venv` environment created by `uv sync --locked --all-groups`.
- Node.js 22 and npm.
- The released model-only `update-004608.pt` bundle and its `.sha256` sidecar at the paths below.
- Loopback ports 8765 and 5173 available for development.

The default is update 4608 because it is the strongest persisted, fully evaluated checkpoint from the completed run. The later update 6144 checkpoint is not silently substituted.

## Download the released policy

```bash
mkdir -p models/checkers/policies
gh release download checkers-policy-v1 \
  --pattern 'checkers-practice-update-004608.pt*' \
  --dir models/checkers/policies

policy=models/checkers/policies/checkers-practice-update-004608.pt
test "$(sha256sum "$policy" | cut -d ' ' -f1)" = "$(tr -d '\n' < "$policy.sha256")"
```

## One-time model export

From the repository root:

```bash
.venv/bin/python scripts/export_checkers_policy.py \
  --config configs/checkers-practice.yaml \
  --checkpoint runs/checkers-practice-seed0-495ff82/checkpoints/update-004608.pt \
  --output models/checkers/policies/checkers-practice-update-004608.pt
```

The exporter validates the full checkpoint and sidecar, extracts only `CheckersNetwork` weights and immutable provenance, writes a new sidecar, strictly reloads the bundle on CPU, and verifies logits/value/greedy-action parity on 12 fixed positions. Both output files are generated artifacts ignored by Git.

## Development

Terminal 1, from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_checkers_web.py \
  --bundle models/checkers/policies/checkers-practice-update-004608.pt \
  --port 8765
```

Terminal 2:

```bash
npm --prefix web/checkers ci
npm --prefix web/checkers run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to `http://127.0.0.1:8765`. To change only the development API target, set `CHECKERS_API_TARGET` before starting Vite.

## Single-server production-local mode

```bash
npm --prefix web/checkers run build
PYTHONPATH=src .venv/bin/python scripts/serve_checkers_web.py \
  --bundle models/checkers/policies/checkers-practice-update-004608.pt \
  --static-dir web/checkers/dist \
  --port 8765
```

Open `http://127.0.0.1:8765`. The Python process refuses non-loopback bind configuration and loads/verifies the model before it starts accepting games.

## Controls

- Choose Red to move first or White to let the policy open.
- Greedy mode uses the trained neural policy and deterministically takes its highest-probability legal action. It is not Minimax-2.
- Sampled mode uses the same trained neural policy with temperature-one sampling. A fresh random 32-bit seed is generated automatically for every game and displayed read-only, making that game's samples reproducible.
- Minimax-2 was an evaluation opponent for measuring the trained checkpoint and is not loaded by the web harness.
- Select a ringed piece, then a gold destination. Forced multi-jumps keep the required piece selected.
- A new match discards the current in-memory game. Restarting the server discards all games.

## Verification

```bash
.venv/bin/ruff check src/checkers/web scripts/export_checkers_policy.py scripts/serve_checkers_web.py tests/web
.venv/bin/pytest -q tests/web --no-cov
npm --prefix web/checkers audit --audit-level=moderate
npm --prefix web/checkers run test
npm --prefix web/checkers run typecheck
npm --prefix web/checkers run build
```

The focused pytest command disables the repository-wide coverage threshold because it intentionally selects only `tests/web`; the final project gate remains the full repository test suite with its configured 92% branch threshold.

## Limitations

Version 1 has a public single-player deployment but no accounts, peer-to-peer play, game persistence, undo, clocks, analysis, editing, or training controls. The browser never implements checkers legality, and local servers bind only to loopback. See [`docs/CHECKERS_WEB_HARNESS_CONTRACT.md`](../../docs/CHECKERS_WEB_HARNESS_CONTRACT.md) for the frozen acceptance contract.
