# IMSA West Checkers AI

This is a deliberately simple, game-first React interface for the repository's trained PPO checkers policy. The supplied IMSA West logo and its official blue, light-blue, orange, and white palette define the visual system. No generated artwork is loaded or copied into the production build.

## Student experience

- Choose Orange to move first or White to let the policy open.
- Press one Start button; deterministic play is the only visible policy mode.
- Select an outlined piece and then a dotted destination.
- Play with mouse, touch, the board's keyboard controls, or the equivalent legal-move buttons.
- Read four short steps explaining representation, self-play, PPO, and checkpoint testing.
- Inspect honest project results with an explicit warning that they are not a human skill rating.

The internal API still calls the first side `red`; the interface calls those pieces Orange to match IMSA West. The browser never decides legality or runs inference. `GameService`, `CheckersEnv`, the action mask, and the loaded update-4,608 policy remain authoritative.

## Run locally

From the repository root:

```bash
npm --prefix web/checkers ci
npm --prefix web/checkers run build
PYTHONPATH=src .venv-checkers/bin/python scripts/serve_checkers_web.py \
  --bundle models/checkers/policies/checkers-practice-update-004608.pt \
  --static-dir web/checkers/dist \
  --port 8765
```

Open `http://127.0.0.1:8765/`.

For development, run the Python server without `--static-dir`, then run `npm --prefix web/checkers run dev`; Vite proxies `/api` to port 8765.

## Controls

- Pointer/touch: choose an outlined checker, then a dotted square.
- Keyboard board: arrow keys navigate dark squares; Enter or Space selects; Escape clears.
- Keyboard alternative: open “Legal move list” and use its standard buttons.
- Stronger board contrast is available beside the game.
- Mandatory captures and multi-jumps are enforced by the Python rules engine.

Games live only in the server's bounded in-memory store and disappear after expiry or restart. Undo, accounts, analytics, browser-side training, fabricated probabilities, and Minimax web play are intentionally absent.

## Verification

```bash
npm --prefix web/checkers run lint
npm --prefix web/checkers run typecheck
npm --prefix web/checkers run test:coverage
npm --prefix web/checkers run build
npm --prefix web/checkers run test:e2e:a11y
npm --prefix web/checkers run test:e2e:responsive
npm --prefix web/checkers run test:e2e:touch
npm --prefix web/checkers run test:e2e:visual
npm --prefix web/checkers audit --audit-level=moderate
```

The complete Python regression remains the source of truth for rules, environment behavior, policy loading, and the HTTP service:

```bash
.venv-checkers/bin/ruff check src/checkers/web tests/web scripts/serve_checkers_web.py
.venv-checkers/bin/pytest -q
```

See [`docs/IMSA_GAME_FIRST_RELEASE.md`](../../docs/IMSA_GAME_FIRST_RELEASE.md) for the observed release results and [`docs/CHECKERS_WEB_HARNESS_CONTRACT.md`](../../docs/CHECKERS_WEB_HARNESS_CONTRACT.md) for the backend contract.
