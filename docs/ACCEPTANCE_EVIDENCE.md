# Acceptance Evidence Matrix

This matrix turns the user request and `GOAL.md` into falsifiable claims. `PROVEN` requires a
current artifact plus a command/result that covers the whole claim. `OPEN` means no completion is
claimed. Internally generated agreement is never promoted to external correctness evidence.

## User-Level Obligations

| ID | Claim | Authority / source | Required proof | Status / current evidence |
|---|---|---|---|---|
| U1 | The downloaded goal is moved into `ml-lab`. | User request | Source path absent; canonical destination present with recorded hash. | PROVEN: source path absent; `GOAL.md` SHA-256 `dab54331…c17660f`; Phase 0 test. |
| U2 | The complete goal is read and used as the controlling specification. | User request; `GOAL.md` §0 | All 1,232 lines accounted for; phase order and state files reflect the document. | PROVEN for initial read: `logs/iterations/000001.md`; ongoing rereads are phase-scoped per §15. |
| U3 | Work loops until every binding objective is complete. | User request; `GOAL.md` §§0, 14, 19 | `STATE.json`, iteration logs, and every matrix row proven; no scope reduction. | OPEN. |
| U4 | Work/project execution continues through 2026-07-28 12:00 America/New_York. | User request | Timestamped work/run evidence reaches or passes the deadline; no early completion claim. | OPEN; deadline not reached. |
| U5 | Every completed objective is provable with a source. | User request; `GOAL.md` §§3, 12.4 | Each claim links definitional authority and authoritative runtime/test evidence. | OPEN; this matrix is the tracking control. |

## Phase Gates

| Gate | Binding source | Proof required | Status / evidence |
|---|---|---|---|
| 0 — Scaffolding | `GOAL.md` §14 Phase 0 | Exact pins, required layout/state, quality tools, offline CI, green `make check`, injected lint/test failures. | PROVEN: `logs/gates/phase-0.txt` exit 0; injected-red logs; `logs/gates/phase-0-gpu-doctor.txt`. |
| 1 — Rules verification | §3.1, §4, §14 Phase 1; primary WCDF/ACF rules | Per-rule primary clause or ENGINE VARIANT label, tests, R6.7 proof, P1 for unavailable text. | BLOCKED by BLOCK-001; technical gate exit 0 at `logs/gates/phase-1.txt`. |
| 2 — State/moves | §§4–5, §§12.2–12.5, §14 Phase 2 | Fast/oracle generators, R4.5 divergence, 5M differential, BFS, metamorphic, 20 published transcripts, mutation ≥85%, rules coverage ≥98%. | BLOCKED only by BLOCK-002/003/004 wording. Feasible technical gate proven: 5M + BFS 7, 20 legal transcripts, valid composed symmetry, published perft through depth 7, 95.76% killed-only mutation, and 100% rules coverage. |
| 3 — Terminal/hash | R6, §5.3, §14 Phase 3 | Boundary/key/property tests and committed termination proof. | OPEN. |
| 4 — Environment | §§5.2, 6, §14 Phase 4 | 5M-step fuzz, zero mask failures, canonical/aliasing/illegal-action/restore tests. | OPEN. |
| 5 — Baselines/arena | §11.2–11.4, §14 Phase 5 | Power-justified balanced matches and hand-worked Elo/payoff validation. | OPEN. |
| 6 — RL core | §§7–9, §12.6–12.7, §14 Phase 6; PPO/GAE primary papers | T1–T8, dtype masking, CPU/GPU determinism. | OPEN. |
| 7 — Self-play/W&B | §§10, 12.8, 13, §14 Phase 7 | Three 30-minute smokes, all metrics, zero mask failures, powered random score ≥0.90, exact resume evidence. | OPEN. |
| 8 — Ablations/full runs | §§8.5, 10.2, 11.5, 14 Phase 8 | Stage A/B/C evidence, three full seeds, all Tier 1/2 rows proven. | OPEN. |
| 9 — Optional | §14 Phase 9 | Only after Gate 8; prior gates rerun after protected changes. | NOT REQUIRED. |

## Section 18 Deliverables

| ID | Deliverable | Proof | Status |
|---|---|---|---|
| D1 | Repository with all binding phase gates GREEN. | `STATE.json` plus gate logs. | OPEN. |
| D2 | Complete checkers `README.md`. | Content audit against §18.2 and reproduced commands. | OPEN. |
| D3 | `docs/RULES.md`. | Rule/source/test matrix plus R6.7 proof. | IN PROGRESS: source matrix/proof verified; planned covering tests remain Phase 2/3 work. |
| D4 | `docs/PPO_CHECKLIST.md`. | All applicable `[HUANG37]` items adjudicated with tiers. | OPEN. |
| D5 | `docs/ML_TEST_SCORE.md`. | Evidence-linked `[MLTS]` self-score. | OPEN. |
| D6 | `docs/METRICS.md`. | Every §13.2 metric formula and range. | OPEN. |
| D7 | `DECISIONS.md`. | Non-obvious choices, authority tier, evidence stage. | IN PROGRESS. |
| D8 | `logs/SUMMARY.md`. | Compacted history including failures. | IN PROGRESS. |
| D9 | W&B runs/artifacts. | At least three seeded baselines, Stage-B ablations, load-validated artifacts. | OPEN. |
| D10 | `scripts/play_human.py`. | Human-play smoke with ACF notation and pending marks. | OPEN. |
| D11 | Final sealed report or explicit NOT EVALUATED. | Invocation ledger/report satisfying §11.6. | OPEN. |

## Tier 1 — Engineering Acceptance

| ID | Requirement from §19 | Authoritative proof | Status |
|---|---|---|---|
| E1 | Clean-clone `make check` with egress blocked. | Fresh clone/namespace transcript after dependency install. | OPEN. |
| E2 | ≥400 passing, none skipped/xfail; coverage and mutation thresholds. | Collection report, coverage JSON, mutation report. | PARTIAL: 95.76% mutation and current coverage gates proven; current suite is below 400 tests. |
| E3 | Strict mypy and Ruff lint/format clean. | Final `make check` log over full checkers scope. | PROVEN for Phase 2 scope at `logs/gates/phase-2.txt`; final rerun remains required. |
| E4 | Every R1.1–R7.3 traced and passing; variants labelled everywhere. | `docs/RULES.md` plus node-ID audit. | OPEN. |
| E5 | R4.5 delayed-removal divergence. | Primary clause, golden fixture, fast/oracle passing tests. | BLOCKED by BLOCK-002; exact pending occupancy/no-repeat tests and parity impossibility proof pass. |
| E6 | State/position key separation. | Dedicated passing tests. | OPEN. |
| E7 | 5M differential, 20 transcripts, metamorphic, external perft or explicit unavailable label. | Saved reports with cited external sources. | PARTIAL/BLOCKED: 5M differential, 20 legal published scores, valid composed symmetry, and Bik external perft through depth 7 proven; BLOCK-003/004 prevent the invalid symmetry/result wording. |
| E8 | 5M fuzz; release soak run or NOT RUN. | Saved fuzz/soak reports. | OPEN. |
| E9 | T1–T8, including hand-computed signed GAE. | Passing tests with derivation fixtures. | OPEN. |
| E10 | D2/D3 and exact mid-sequence resume. | Determinism/resume transcripts and metadata. | OPEN. |
| E11 | All mask violation metrics exactly zero. | Aggregated immutable run metrics. | OPEN. |
| E12 | All §18 deliverables substantive. | Deliverable-by-deliverable content audit. | OPEN. |
| E13 | No §2 violation; near-misses listed. | Final audit against Git/files/logs. | OPEN. |
| E14 | `scripts/reproduce.sh` reproduces headline values. | Fresh execution and tolerance comparison. | OPEN. |
| E15 | No unresolved P0 blocker. | `BLOCKERS.md` audit. | OPEN. |

## Tier 2 — Learning Acceptance

| ID | Requirement from §19 | Authoritative proof | Status |
|---|---|---|---|
| L1 | Three full-budget seeds with mean ± 95% CI and game counts. | Immutable configs, checkpoints, metrics, evaluation report. | OPEN. |
| L2 | Score ≥0.95 random and ≥0.85 greedy. | Powered colour-balanced arena results. | OPEN. |
| L3 | Score ≥0.60 minimax(2). | Powered colour-balanced arena result. | OPEN. |
| L4 | All §11.5 trend criteria. | Saved evaluation series and bootstrap analysis. | OPEN. |
| L5 | Payoff matrix, three-cycles, exploitability proxy. | Population report and validated analysis tests. | OPEN. |
| L6 | Mandatory Stage-B ablations, including adverse findings. | Three-seed arm reports and `DECISIONS.md`. | OPEN. |

## Tier 3 — Non-Binding Strength Report

| ID | Requirement from §19 | Authoritative proof | Status |
|---|---|---|---|
| S1 | At most one sealed invocation per candidate and three candidates, or NOT EVALUATED. | External suite hash plus invocation ledger/report. | OPEN. |
| S2 | Result reported plainly without retries or concealment. | Final report and invocation audit. | OPEN. |
