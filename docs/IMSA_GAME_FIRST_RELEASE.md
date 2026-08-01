# IMSA West Game-First Release

Reviewed: 2026-08-01

## Release objective

Replace the routed clay storybook with a simpler student experience: the real checkers game first, IMSA West identity throughout, and a short evidence-backed explanation below it. The Python rules, game service, API schemas, and selected update-4,608 policy were preserved.

## What ships

- One responsive React page at `/`; old deep links receive the same game-first application through the static fallback.
- The exact user-supplied IMSA West logo at `web/checkers/public/assets/imsa-west-logo.png` (SHA-256 `e989471e7211928ad751dd560194c75fc7fa6ac9738881e735b476b69444c05b`).
- The logo's blue `#0033A1`, light blue `#95ABD6`, orange `#FF6C0E`, and white palette.
- A deterministic game setup with only side choice and Start/New Game.
- The existing server-authoritative board with mouse, touch, keyboard-grid, and legal-move-button input.
- Concise explanations of representation, self-play, PPO, checkpoint testing, and the rules/policy boundary.
- Authentic update-4,608 project evidence with its evaluation limitations.

No generated artwork, storybook routes, route progress, cinematic animation, theme framework, public evidence JSON, or unused UI-kit code is in the project frontend or production build. Retired storybook materials remain locally recoverable under the ignored `cache/retired-checkers-storybook-20260801/` directory.

## Observed validation

| Gate | Result |
|---|---|
| Frontend lint | Pass, zero warnings |
| TypeScript application + browser tests | Pass |
| Unit/component tests | 23/23 across four files |
| Frontend coverage | 84.34% statements/lines, 84.78% branches, 76.59% functions |
| Chromium release tests | 19/19: identity, real move, side selection, keyboard, axe, preferences, ten viewports, recovery, two visual baselines |
| Touch Chromium | Pass: one touch sequence produces one model-backed move |
| Firefox | Pass: page and real game start without console errors |
| WebKit | Pass using the documented cache-local Linux media-library shim |
| Automated accessibility | Zero axe WCAG A/AA violations before and during play |
| Responsive set | All ten required sizes from 320×568 through 2560×1440 pass overflow and game-visibility checks |
| Production build | Pass; 340 KB total on disk and only HTML, logo, CSS, and JavaScript |
| Dependency audit | `npm audit --omit=dev`: zero vulnerabilities |
| Python web lint | Pass; backend source is unchanged by this simplification |
| Python web tests | 45/45 pass (`--no-cov`; the full suite below is the coverage gate) |
| Python regression | Prior complete gate remains valid: 986 passed, 92.39% coverage; no Python behavior changed afterward |

The browser test submitted ACF 9→13 through the visible legal-move alternative. The real server returned both the human step and policy reply, and the UI returned to the student's turn.

## Performance profile

`npm --prefix web/checkers run measure:production` runs Chromium 149 at 1366×768 with 4× CPU slowdown, 40 ms loopback latency, 10 Mbps down, and 5 Mbps up. The schema-4 report is `reports/checkers_web_performance_20260801.json`.

| Measurement | Observed | Budget |
|---|---:|---:|
| LCP | 508 ms | ≤ 2,500 ms |
| CLS | 0.0001 | ≤ 0.05 |
| Start click to playable board | 137.1 ms | ≤ 1,000 ms |
| Human move through policy reply | 114.7 ms | ≤ 1,000 ms |
| Controlled Event Timing latency | 40 ms | ≤ 200 ms |
| Initial transfer | 328,956 bytes | ≤ 384,000 bytes |
| Logo transfer | 30,330 bytes | ≤ 65,536 bytes |
| Heap growth across six games | 257,344 bytes | ≤ 10 MiB |

These are local engineering measurements, not public-field Core Web Vitals or a human-strength benchmark.

## Reproduce

```bash
npm --prefix web/checkers ci
npm --prefix web/checkers run lint
npm --prefix web/checkers run typecheck
npm --prefix web/checkers run test:coverage
npm --prefix web/checkers run build
PYTHONPATH=src .venv-checkers/bin/python scripts/serve_checkers_web.py \
  --bundle models/checkers/policies/checkers-practice-update-004608.pt \
  --static-dir web/checkers/dist \
  --port 8765
```

Open `http://127.0.0.1:8765/`.

## Bounded limitations

- The server exposes one selected policy bundle, not a checkpoint picker.
- Undo and persisted games are intentionally unavailable in server-authoritative play.
- The endpoint does not expose probabilities or value estimates, so the UI does not invent them.
- The evaluation suite helped select the checkpoint; its numbers are not sealed-test evidence.
- The production server is intentionally loopback-only.
