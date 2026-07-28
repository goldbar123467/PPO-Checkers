# Checkers Environment Contract and Evidence

This document records the implemented Phase 4 software contract, its authority, and its executable
evidence. Over-the-board rules remain sourced clause-by-clause in `docs/RULES.md`; environment
choices are PROJECT CONTRACTS from `GOAL.md` unless an external API or algorithm source is named.

## External API and Algorithm Sources

- [Gymnasium `Env` API](https://gymnasium.farama.org/api/env/) defines the reset/step interface,
  the five-value step result, seeding through `super().reset(seed=seed)`, and the distinction
  between `terminated` and `truncated`. The implementation is pinned to `gymnasium==1.3.0`; the
  wheel SHA-256 in `uv.lock` is
  `6b8c159a8540dcbcb221722d7efda24d78ebbcbc3bd2ea1c2611aa2a34471fc2`.
- [Gymnasium fundamental spaces](https://gymnasium.farama.org/api/spaces/fundamental/) documents
  `Discrete` and its optional sampling mask. The environment exposes `Discrete(128)` but rejects
  any masked action with `IllegalActionError`, as required by `GOAL.md` §6.1.
- Huang and Ontañón, [*A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms*](https://arxiv.org/abs/2006.14171), is the primary algorithm source for masking the
  policy distribution to valid actions. Phase 4 produces and stores exact Boolean masks; the
  dtype-aware differentiable distribution remains a Phase 6 obligation.

The official pages and primary paper were rechecked on 2026-07-28. The installed Gymnasium 1.3.0
runtime signatures were also inspected locally before implementation.

## Frozen API

`CheckersEnv` is a Gymnasium `Env` with `Discrete(128)` actions and a float32
`Box(8, 8, 8)` observation. `reset(seed=...)` returns `(observation, info)`. `step(action)` returns
`(observation, reward, terminated, truncated, info)`. The info mapping has exactly:

- `legal_mask`: a fresh Boolean array of shape `(128,)`;
- `actor`: the explicit player who selected this transition;
- `move_completed`: whether this step ended the complete checkers move;
- `checkers_move_san`: canonical ACF text only at a move boundary;
- `outcome`: the rules outcome only after termination.

An illegal, out-of-range, wrongly typed, or post-terminal action raises `IllegalActionError` before
state mutation. Game rules set `terminated`; they never set `truncated`. At a terminal state the
mask is intentionally all false because no subsequent `step` is accepted. Every nonterminal state
has at least one true mask entry.

## Observation and Action Encoding

The eight planes exactly implement `GOAL.md` §6.2: actor men/kings, opponent men/kings including
pending captures, pending-capture marks, forced-piece one-hot, actor no-progress counter divided by
40, and ply divided by `max_plies`. White-to-move states rotate 180° and swap actor/opponent planes.
Light squares are zero in planes 0–4. The baseline deliberately omits the opponent counter; the
ninth-plane Stage-B ablation remains open.

Actions are `canonical_square * 4 + canonical_direction`. The frozen direction order is
`(+row,-column)`, `(+row,+column)`, `(-row,-column)`, `(-row,+column)` in Red's world frame. A White
action rotates the origin and reverses both direction signs. Short-jump geometry makes origin plus
direction sufficient to recover destination and captured square uniquely.

Planes 4 and 5 distinguish a forced continuation from the otherwise identical boundary
placement. The representation-level N7 regression passes. BLOCK-006 records why Gate 4's stronger
different-**logits** demand must wait for the Phase 6 network.

## Move Boundaries, Rewards, and Hashes

One `step` is one simple move or one jump. During a multi-jump, actor identity is retained,
notation remains absent, repetition is not counted, and no-progress counters do not change. The
final jump emits the complete path, removes all pending pieces, updates the actor counter once, and
changes side. `ply` increments on every environment step.

Reward is zero except on the transition into a terminal state, where it is the outcome from that
transition's actor perspective. Losses and project draw variants use the source adjudication in
`docs/RULES.md`. The wrapper maintains `state_key` incrementally and exposes boundary-only
`position_key`; full recomputation is checked throughout fuzzing.

## Serialization and Vectorization

`CHECKERS_ENV_1` JSON snapshots include current state, reset state, immutable rule configuration,
sorted official-position visit counts, and the complete partial capture path needed to finish ACF
notation after restore. Parsing validates exact fields and cross-field invariants. Restore parses
and validates everything before mutating the live environment.

`CheckersVectorEnv` advances each lane by one environment step. It prevalidates the whole action
batch so one illegal or terminal lane cannot partially mutate earlier lanes. Its
`CHECKERS_VECTOR_ENV_1` snapshot embeds every lane and restores atomically, including a mix of
boundary and mid-sequence states.

## Requirement-to-Evidence Matrix

| Objective | Authority | Executable evidence | Result |
|---|---|---|---|
| Gym reset/step and terminated/truncated API | `GOAL.md` §6.1; official Gymnasium `Env` API | `tests/env/test_environment.py` | PASS |
| Canonical eight-plane observation | `GOAL.md` §6.2; valid rotation+player symmetry in `docs/RULES.md` | `tests/env/test_encoding.py` | PASS |
| 128-ID bijection and canonical directions | `GOAL.md` §6.3; WCDF short-step geometry | `tests/env/test_actions.py` | PASS |
| Actor-relative terminal rewards | `GOAL.md` §§6.4, 7.1; R6 adjudication | `tests/env/test_environment.py` | PASS |
| Exact masks and illegal-action failure | `GOAL.md` §§6.1, 6.5; Huang–Ontañón | `tests/env/test_actions.py`, `test_environment.py` | PASS |
| Exact mid-sequence resume | `GOAL.md` §§5.1–5.2, 7.5 | `tests/env/test_serialization.py` | PASS |
| Step-lockstep vector resume | `GOAL.md` §7.5 | `tests/env/test_vec_env.py` | PASS |
| 5M randomized phase gate | `GOAL.md` §§12.4–12.5 and Phase 4 | immutable report below | PASS |
| N7 different observations | `GOAL.md` §6.2 | named N7 encoding regression | PASS |
| N7 different logits | `GOAL.md` N7, but network is Phase 6 | BLOCK-006 | BLOCKED until Phase 6 |

The consolidated Phase 4 technical suite contains 322 tests total, including 276 under
`rules/` and `env/`, with 100% statement/branch coverage over all current `checkers` modules.

## Five-Million-Step Gate

The immutable report is `reports/phase4_environment_fuzz_5m_seed20260728.json`, SHA-256
`1472e4ea1da80f591ee248748d066fdb05bea72cc78f3a0f5f9aecebb0f479ed`. It records clean Git
revision `66e1c4b50a7d3761a84da209943c901bbd5d376d`, seed `20260728`, exact dependency/hardware
metadata, the pinned GOAL hash, and hashes for every consumed implementation file.

Observed over exactly 5,000,000 transitions: 120,154 terminal games, 1,015,426 capture steps,
121,260 continuation steps, 224,842 promotions, and 500 serialize/reload checks, nine of them
mid-sequence. Invariant violations, mask disagreements, and empty nonterminal masks were all zero.
The report explicitly labels this randomized regression evidence, not a formal proof.
