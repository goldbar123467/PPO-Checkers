# PPO implementation-detail checklist

This checklist adjudicates the 37 details catalogued by Huang, Dossa, Raffin, Kanervisto, and
Wang, [*The 37 Implementation Details of Proximal Policy Optimization*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
(ICLR Blog Track, 2022). That source is Tier B implementation evidence, not binding law. The
checkers-specific algorithm in `GOAL.md` has priority where the task differs from Atari, continuous
control, LSTM, or `MultiDiscrete` environments.

Status meanings: **ADOPTED** is implemented directly; **ADAPTED** preserves the purpose with an
explicit checkers/two-player change; **REJECTED** is intentionally not used; **N/A** does not apply
to this environment. Code and test paths are repository-relative.

## Thirteen core details

| # | Detail | Status | Authority | Adjudication and evidence |
|---:|---|---|---|---|
| 1 | Vectorized architecture | ADOPTED | Tier B | Synchronous 64-lane collection in `env/vec_env.py` and `rl/selfplay.py`; lockstep/mid-sequence tests in `tests/rl/test_selfplay.py`. |
| 2 | Orthogonal weights; constant biases | ADOPTED | Tier B/default | `rl/networks.py` uses gain √2 in hidden affine layers, 0.01 policy output, 1.0 value output, and zero biases; `tests/rl/test_networks.py`. |
| 3 | Adam epsilon | ADOPTED | Tier B/default | `RunConfig.adam_eps=1e-5`; optimizer construction in `train.py`; configuration/checkpoint tests. |
| 4 | Adam learning-rate annealing | ADOPTED | Tier B/default | Pure linear schedule in `schedules.py`; optimizer groups updated without mutating config; schedule tests. |
| 5 | GAE and value bootstrap | ADAPTED | LAW + Tier B | `rl/gae.py` implements actor-relative two-player GAE with σ on every recursion edge, terminal zero bootstrap, and rollout-boundary bootstrap; T3/T6/T7/T8 tests. |
| 6 | Shuffled mini-batch updates | ADOPTED | Tier B | `RolloutUpdater` makes a full seeded permutation per epoch and records every source index; exactly-once and completeness tests in `tests/rl/test_train.py`. |
| 7 | Per-mini-batch advantage normalization | ADOPTED | Tier B/default | `compute_ppo_loss` normalizes with epsilon 1e-8 within each minibatch; PPO numerical tests. |
| 8 | Clipped surrogate objective | ADOPTED | LAW | Literal PPO-Clip minimum in `rl/ppo.py`; hand-computed and directional tests T3/T4. |
| 9 | Value-function loss clipping | REJECTED | GOAL §8.2 default | Plain MSE is intentional; Engstrom-era clipping is not assumed beneficial. `tests/rl/test_ppo.py` locks the formula. |
| 10 | Overall loss and entropy bonus | ADAPTED | LAW/default | `policy + 0.5·value − ent_coef·entropy`; entropy coefficient linearly anneals 0.01→0.001 over the first 50% by project hypothesis. |
| 11 | Global gradient clipping | ADOPTED | Tier B/default | `clip_grad_norm_` at 0.5 with non-finite failure in `rl/ppo.py`; gradient tests. |
| 12 | Debug variables | ADAPTED | Tier B | All source metrics plus mask, policy, value, game, anchor, and population diagnostics are frozen in `metrics.py`; KL uses the lower-variance nonnegative k3 estimator and all 55 keys are completeness-audited. |
| 13 | Shared or separate policy/value network | ADAPTED | Tier B + project hypothesis | Phase-7 A0 uses a shared GroupNorm residual trunk with separate heads. Shared versus separate trunk remains a mandatory Stage-B ablation; no superiority claim is made yet. |

## Nine Atari-specific details

| # | Detail | Status | Authority | Adjudication and evidence |
|---:|---|---|---|---|
| 14 | No-op reset | N/A | Atari-specific | Checkers has one exact initial position and seeded stochastic policies; random opening moves would change the game definition. |
| 15 | Max-and-skip / frame max | N/A | Atari-specific | Every legal checkers substep is semantically significant; skipping would violate the rules. |
| 16 | Episodic life | N/A | Atari-specific | Checkers has no life counter; terminal conditions are R6.1–R6.5 only. |
| 17 | Fire reset | N/A | Atari-specific | No reset action exists. |
| 18 | Warp frame | N/A | Atari-specific | Input is a canonical 8×8×8 symbolic board, not pixels. |
| 19 | Sign reward clipping | REJECTED | Project law | Rewards are already exactly −1/0/+1 at terminal and zero otherwise; no reward wrapper or shaping is permitted. |
| 20 | Frame stacking | N/A | Atari-specific | The complete Markov state, including forced continuation and pending captures, is encoded directly. |
| 21 | Shared Nature CNN | ADAPTED | Architecture hypothesis | A shared residual CNN is used, but Nature-CNN geometry is inappropriate for an 8×8 symbolic board; architecture is specified in `rl/networks.py`. |
| 22 | Scale image bytes to [0,1] | N/A | Atari-specific | Observation planes are natively bounded semantic planes; there are no 0–255 image bytes. |

## Nine continuous-action details

| # | Detail | Status | Authority | Adjudication and evidence |
|---:|---|---|---|---|
| 23 | Normal action distribution | N/A | Continuous-only | The action space is one masked `Discrete(128)` categorical. |
| 24 | State-independent log standard deviation | N/A | Continuous-only | No Gaussian standard deviation exists. |
| 25 | Independent continuous components | N/A | Continuous-only | Each environment step chooses one encoded action ID. |
| 26 | Separate continuous-control MLPs | N/A | Continuous-only | The relevant shared/separate question is already adjudicated in item 13. |
| 27 | Clip continuous action; store raw action | N/A | Continuous-only | Illegal actions have exactly zero probability under the stored mask; no post-sampling clipping occurs. |
| 28 | Running observation normalization | REJECTED | Task-specific | Semantic planes have fixed meanings and bounded ranges; running statistics would make a position depend on training history. GroupNorm normalizes learned features instead. |
| 29 | Observation clipping | N/A | Continuous-only | Encoded planes are already bounded and validated. |
| 30 | Reward scaling | REJECTED | Project law | Would alter the declared terminal-only objective and is prohibited by `GOAL.md` §2.3. |
| 31 | Reward clipping after scaling | REJECTED | Project law | No scaling is performed; terminal rewards are already in the exact target range. |

## Five LSTM details

| # | Detail | Status | Authority | Adjudication and evidence |
|---:|---|---|---|---|
| 32 | LSTM layer initialization | N/A | Recurrent-only | The baseline has no recurrent module. |
| 33 | Zero initial recurrent state | N/A | Recurrent-only | No recurrent state exists. |
| 34 | Reset recurrent state on episode end | N/A | Recurrent-only | No recurrent state exists; vector environments themselves reset exactly. |
| 35 | Sequential recurrent minibatches | N/A | Recurrent-only | Non-recurrent PPO deliberately shuffles complete rollout indices. Full chronology is retained only for two-player GAE before minibatching. |
| 36 | Reconstruct rollout recurrent state | N/A | Recurrent-only | No recurrence exists; stored observations, masks, old values, and old log-probabilities reconstruct the behavior distribution. |

## One MultiDiscrete detail

| # | Detail | Status | Authority | Adjudication and evidence |
|---:|---|---|---|---|
| 37 | Independent MultiDiscrete components | N/A | MultiDiscrete-only | The action is one categorical `(from-square, direction)` ID; legality is represented by one exact 128-bit mask. |

## Four auxiliary details from the same source

These are outside the numbered 37 but materially relevant.

| Detail | Status | Adjudication and evidence |
|---|---|---|
| Clip-range annealing | REJECTED | `clip_coef=0.2` is fixed; changing it requires staged evidence. |
| Parallel gradient update | N/A | One learner and one RTX 5070; environment collection is vectorized, optimizer execution is single-device. |
| KL early stopping | ADOPTED | k3 approximate KL, target 0.02, stops the remaining policy/value minibatches and logs the count; `rl/ppo.py`, `train.py`, and tests. |
| Invalid-action masking | ADOPTED | Algorithm-defining law: stored masks enter both rollout and gradient paths; masked logits have zero probability and exactly zero gradient; masking/oracle violations halt training. |

This checklist describes implementation fidelity only. It is not evidence that the policy learned;
learning claims require the independent seeded evaluations and confidence intervals in the run
reports.
