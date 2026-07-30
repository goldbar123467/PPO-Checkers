# Local Checkers Web Harness Contract

Status: frozen implementation contract
Experiment type: engineering integration and model inference validation
Date: 2026-07-29

## Objective and falsifiable hypothesis

Build a localhost-only Vite + React + TypeScript harness in which a person can start a game, choose a side, and play legal American checkers against a previously trained local policy.

The hypothesis is: a model-only export of the selected PPO checkpoint can be loaded once by a local Python server and can complete a human-versus-model browser game while producing exactly the same masked greedy action as the source checkpoint on fixed validation states.

## Strict limitations

1. The Python implementation in `src/checkers` is the only authority for rules, legal actions, forced captures, multi-jumps, promotion, repetition, and terminal outcomes. TypeScript must not implement a second rules engine.
2. The server binds to `127.0.0.1` by default. Version 1 has no remote access, accounts, authentication, telemetry, cloud calls, multiplayer, matchmaking, or chat.
3. The trained policy is loaded locally before a game can start. Browser code never receives PyTorch weights and never executes arbitrary model output.
4. The default policy is the best persisted evaluated checkpoint, `update-004608.pt`, not the numerically last checkpoint. The original checkpoint is immutable.
5. Runtime inference uses a validated model-only bundle. It must not restore an optimizer, rollout buffer, RNG state, or training environment on each request.
6. Human and model moves are accepted only by mapping an origin/destination pair to the current server-generated legal-action map. An illegal or out-of-turn move returns a structured 4xx error and leaves game state unchanged.
7. Model inference defaults to greedy masked action selection on CPU. Optional seeded sampling is allowed only when explicitly selected before a new game.
8. Games are in-memory and disappear when the server stops. Version 1 has no undo, save/load, analysis tree, opening book, board editor, clock, or training controls.
9. Generated imagery is decorative only. The playable board, pieces, labels, focus states, and legal-move cues remain semantic HTML/CSS for precision and accessibility.
10. GitHub source study is clean-room: inspect one permissively licensed checkers UI for interaction concepts, record its URL/revision/license, and write new code and assets without copying source or bundled art.
11. Checkpoints, model bundles, generated build output, dependency trees, and runtime game state remain untracked. No credentials or private values enter source, logs, URLs, or browser storage.
12. The existing run variables remain authoritative where relevant: `State`, `PlayerId`, `CheckersEnv`, `Step`, the 128-action encoding, `max_plies: 512`, and `repetition_draws: true`.

Deployment addendum: the later public release preserves the contract's loopback process boundary. Caddy is the only public origin listener, the Python service still binds to `127.0.0.1`, and neither weights nor rule authority moved into the browser.

## Exact deliverables

| ID | Deliverable | Acceptance criterion |
| --- | --- | --- |
| D1 | Frozen contract | This file exists before implementation and its scope/criteria are not weakened during the build. |
| D2 | Reference record | A report identifies one GitHub checkers-board repository, exact inspected revision and license, and the interaction ideas used without copied code/assets. |
| D3 | Original visual asset | One image-generation-produced background is stored under the web app's `public/assets`, documented, and used decoratively. |
| D4 | Safe policy export | A CLI exports a model-only bundle, records source checkpoint SHA-256/config/revision/update, writes its own SHA-256 sidecar, reloads it strictly, and verifies fixed-state action/logit parity with the source model. |
| D5 | Local Python API | A typed localhost server loads the bundle at startup and implements health/model metadata, game creation, and legal human move endpoints using the existing environment. |
| D6 | Browser game | A Vite + React + TypeScript UI provides a responsive 8x8 board, side choice, deterministic/sampled policy choice, selection/legal/last-move cues, turn/result feedback, model-ready gating, move history, and new-game flow. |
| D7 | Automated checks | Focused Python tests cover bundle validation and game flow; frontend tests cover interaction/error states; Ruff, pytest, TypeScript checking, and production build all pass. |
| D8 | End-to-end proof | A real local server is exercised in a browser: readiness, game start, at least one legal human move, one model reply, illegal/non-playable interaction resistance, responsive layout, console errors, and production asset loading are checked. |
| D9 | Reproduction guide | README instructions name prerequisites, export/start commands, default checkpoint, local URLs, controls, limitations, and verification commands. |

## Architecture contract

```text
browser (React/TypeScript)
        | JSON /api on localhost
        v
Python HTTP adapter -> GameService -> CheckersEnv / legal_action_map
                                \-> PolicyAgent -> CheckersNetwork
                                                     ^
                                       validated model-only bundle
```

- Development: Vite serves the UI and proxies `/api` to the loopback Python server.
- Production-local: the Python server serves the Vite `dist` directory and the same `/api` routes.
- Board coordinates sent over the API are the repository's 0-based `(row, col)` coordinates. Display labels use the existing 1–32 notation helpers.
- A completed human turn triggers model steps until control returns to the human or the game terminates, which preserves atomic forced multi-jumps.
- Error responses use `{ "error": { "code": string, "message": string } }`; game mutation endpoints return a complete current snapshot.

## Verification gates

The work is complete only when all applicable commands pass and their observed outputs are recorded in the implementation report:

```bash
.venv/bin/ruff check src/checkers/web scripts tests/web
.venv/bin/pytest -q tests/web
npm --prefix web/checkers run test
npm --prefix web/checkers run typecheck
npm --prefix web/checkers run build
```

Browser verification must use the actual exported checkpoint bundle, not a mock. A failed gate remains a reported failure; it is not silently removed from this contract.
