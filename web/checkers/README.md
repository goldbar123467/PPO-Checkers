# PPO Checkers — web app

A static Vite + React + TypeScript site where you play American checkers against the repository's trained PPO policy. The rules engine and the neural network both run in the browser, so the build is plain static files (deployed on Vercel).

## How it is put together

```text
src/
  engine/            framework-free game and model code
    rules.ts         bitboard rules: legal steps, captures, promotion, terminal outcomes
    encoding.ts      8×8×8 actor-canonical observation and the 128-slot action space
    game.ts          step-wise environment with ACF notation and repetition counting
    network.ts       GroupNorm ResNet forward pass over raw float32 weights
    policy.ts        masked greedy / sampled action selection
  lib/
    session.ts       one human-vs-policy game, rendered as immutable snapshots
    policy.worker.ts Web Worker that loads the weights and runs inference
    policyRunner.ts  worker client (falls back to the page thread if workers are unavailable)
    loadPolicy.ts    downloads and SHA-256-verifies the weights
  hooks/useCheckers.ts  React state for the model, the game, and the AI's turns
  components/        board, setup panel, status, network insight, move history
  model/             policy.bin (1.88 MB float32) + policy.json manifest
```

Each file in `src/engine` is a port of the matching Python module in `src/checkers` (`rules/moves.py`, `rules/terminal.py`, `env/encoding.py`, `env/masking.py`, `env/checkers_env.py`, `rl/networks.py`, `agents/policy_agent.py`). Player 0 is Red and moves first. The Python engine calls player 1 White; the site shows those pieces as Black.

The AI always plays its highest-logit legal move. After each move the sidebar shows the value head's estimate from the AI's perspective, the softmax probability of the chosen move among the legal moves, and the on-device inference time. The value is the raw network output, not a calibrated win probability.

## Parity with Python and PyTorch

`scripts/export_browser_policy.py` (repository root) writes the weights, the manifest, and `src/test/fixtures/parity.json`. The fixture holds 43 games (random, greedy, and sampled agents; all five termination rules) plus 36 positions with PyTorch logits and values. `src/test/engine.parity.test.ts` requires the TypeScript engine to:

- match Bik's published completed-move perft counts through depth 6;
- replay every fixture game with identical legal-action lists, notation, final states, and outcomes;
- encode identical observations and legal actions;
- match PyTorch logits and values within 1e-4 and reproduce every recorded greedy decision.

On the Python side, `tests/web/test_browser_export.py` replays the same fixture through the Python engine and the committed weights, so neither side can drift.

## Develop

```bash
npm ci
npm run dev        # http://127.0.0.1:5173
npm run build
npm run preview    # production build with the vercel.json headers, http://127.0.0.1:4173
```

## Controls and accessibility

- Pointer or touch: tap a highlighted piece, then a dotted square.
- Keyboard: arrow keys move between dark squares, Enter or Space selects, and Escape clears.
- "Legal move list" offers every legal move as a standard button.
- High-contrast board toggle, visible focus, skip link, live turn announcements, reduced-motion and forced-colors support.

## Verify

```bash
npm run lint
npm run typecheck
npm run test:coverage   # unit, UI, and parity tests
npm run test:e2e        # Playwright: gameplay, keyboard, touch, axe WCAG, viewports, CSP, recovery
npm audit --audit-level=moderate
```

Playwright runs against `npm run build && npm run preview`, so it exercises the real Web Worker, the real weights, and the production Content-Security-Policy. If Playwright's own browser download is unavailable, point it at a local Chromium with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome`. Visual baselines in `e2e/visual.spec.ts-snapshots` are rendered on Linux Chromium.
