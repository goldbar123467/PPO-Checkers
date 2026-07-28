# Qwen3-4B × SynthGSM8K RTX 5070 experiment plan

Date prepared: 2026-07-27 (America/New_York)  
Lab: `/home/thecl/ml-lab`  
Status: plan and CPU data validation complete; no baseline inference or training launched

## Objective and claims

This is a staged systems and learning experiment, not a single full-corpus run.

Primary engineering hypothesis:

> Qwen3-4B NF4 QLoRA can complete a deterministic 5,000-example, checkpoint-resumed
> workload on the RTX 5070 without CUDA/Linux OOM, sustained swap activity, hidden CPU
> offload, or unbounded memory growth.

Secondary learning hypothesis:

> With identical prompts, quantization, generation settings, answer parser, and untouched
> GSM8K test records, the reloaded Run B adapter will improve exact-match accuracy over the
> pinned base model.

Training loss alone cannot establish the secondary claim. SynthGSM8K-50K is the training
corpus and systems workload. The untouched `openai/gsm8k` test split is the benchmark.

## Immutable identities

| Artifact | Exact identity |
|---|---|
| Base model | `Qwen/Qwen3-4B` |
| Base revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Training dataset | `clarkkitchen22/SynthGSM8K-50K` |
| Training revision | `ebf8f270d82680fc8b31c15bd1535eafa972da07` |
| Benchmark | `openai/gsm8k`, config `main`, split `test` |
| Benchmark revision | `740312add88f781978c0658806c59bc2815b9866` |
| Historical control model | `clarkkitchen22/Qwen3-4B-GSM8K-Synth-35K` |
| Historical model revision | `124bff2f1736d06cb8765a9fafcfd458ff990962` |

The historical model card publishes settings and aggregate results but no ordered 34,818-row
manifest. Its linked source repository contains generation/filtering code, not the training row
selection. Stage C is therefore a **historical reconstruction**, not an exact reproduction.

## Prepared data evidence

The following CPU-only preparation has already run:

| Artifact | Evidence |
|---|---|
| Run B train | 5,000 records; SHA-256 `638885531d8fb0163a2dbfd91ddb949e606700ba4e5693291ce528383d2bc567` |
| Run B validation | 256 records; SHA-256 `8d69cf7e3c2d4a7ce9d482f385579d935d0b04c9d1084f5fde2d9f1d6600540d` |
| Run B combined | 5,256 records; SHA-256 `e5fccfda245e8be3b527bc5e50495fcd29e20f05a43c2734247f3d30d7ce0059` |
| GSM8K test | 1,319 records; SHA-256 `1f137f749ca8245bdc4baf596aecc785478153e21d8452b54fd6bc2964d4287c` |

All 50,418 source records passed structural transformation and fit the 1,024-token Qwen ceiling;
none were truncated or rejected. The source `id` field is not unique: 37,421 unique values exist,
12,997 values occur twice, and 25,994 rows belong to repeated-ID pairs. Selection therefore uses
the pinned zero-based source row index plus source ID, then sorts by
`sha256(seed + NUL + row_index:id)`. No row was silently discarded because its ID repeated.

Authoritative local manifests:

- `data/processed/synthgsm8k-ebf8f27/run-b-5k/manifest.json`
- `data/processed/gsm8k-740312a/manifest.json`

## Transformation and masking contract

Every accepted training record becomes one native Qwen conversation:

```text
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
{solution with <<...>> calculation annotations removed}
</think>

#### {canonical numeric answer}<|im_end|>
```

Required invariants:

1. `enable_thinking=True` is explicit in preprocessing and masking audits.
2. Prompt and completion remain separate until TRL tokenization.
3. `completion_only_loss=True`; every prompt, user, special, and padding label is `-100`.
4. Every selected record has at least one supervised assistant token.
5. The final-answer parser accepts only the last explicit `#### <number>` marker.
6. Integer-like floats are normalized (`108.0` becomes `108`).
7. Overlength examples are excluded and counted, never silently truncated.
8. The ordered transformed-record digest is stored independently of the Hub revision.

The source has strong automated filtering but known semantic edge cases. Prepared rows are marked
`reviewed_for_hardware_benchmark`, not certified classroom material. The six semantic quality flags
remain unknown until a separate `v2-clean` review. No classroom-quality claim may use this status.

## Experiment stages and gates

### Run A — untouched base baseline

Classification: baseline evaluation. No training.

Run a 32-record deterministic canary, then all 1,319 problems. After that, run the sampled track
with seeds 11, 23, 37, and 51.

Deterministic track:

- `do_sample=False` only for comparison with the historical 85.0% claim.
- It is labeled legacy because Qwen warns against greedy thinking-mode generation.

Qwen-recommended track:

- `enable_thinking=True`
- temperature `0.6`, top-p `0.95`, top-k `20`
- report per-seed accuracy, population standard deviation, majority-vote accuracy, ties,
  invalid answers, output length, latency, throughput, and peak VRAM.

Gate to Run B: both canaries complete on CUDA, all 1,319 benchmark records validate, the parser
does not fall back to “last number,” and the full deterministic baseline is saved locally.

### Run B — controlled 5K learning and resume test

Classification: controlled learning/system test.

Fixed training configuration:

| Setting | Value |
|---|---:|
| QLoRA | NF4, double quantization, BF16 compute |
| Sequence length | 1,024 |
| Microbatch | 1 |
| Gradient accumulation | 16 |
| Effective examples/optimizer step | 16 |
| Train records | 5,000 |
| Validation records | 256, explicit manifest split |
| Target optimizer steps | 313 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Targets | q/k/v/o/gate/up/down projections |
| Optimizer | paged AdamW 8-bit |
| Learning rate | `2e-4`, cosine, 3% warmup |
| Checkpoints | steps 64, 128, then every 64; keep at most 2 |
| Packing | disabled for the controlled comparison |
| CPU offload/fallback | forbidden |

Part 1 stops normally at step 128. Part 2 starts a new local/W&B run, reloads
`checkpoint-128`, and ends at global step 313. Grouping preserves the relationship without
pretending two processes are one uninterrupted W&B run.

Gate to post-training evaluation:

- no CUDA/Linux OOM or hidden CPU parameter/offload;
- all audited adapter gradients finite;
- only LoRA tensors trainable;
- loss finite and lower over a meaningful window, not merely one noisy step;
- memory series has no sustained positive growth after warm-up;
- no sustained swap-in/out samples;
- checkpoint-128 resume reaches global step 313;
- final adapter reloads in a fresh process and generates on the RTX 5070.

Evaluate the adapter on the identical deterministic and sampled tracks. An improvement is reported
only relative to the newly measured base baseline, with absolute counts and invalid-answer rates.

### Run C — historical reconstruction

Classification: reconstruction/ablation, not reproduction.

Only start after Run B and a three-step 4,096-token rank-32 memory canary pass. The profile matches
the published values where available: 34,818 deterministically selected examples, rank 32, alpha
32, dropout 0, batch 1, accumulation 16, three epochs, `2e-4`, cosine, 10 warmup steps, BF16,
4,096 tokens, and the seven projection targets. The unknown historical row selection and possible
Unsloth implementation differences must be listed with every comparison.

### Run D — new full-dataset experiment

Classification: new experiment.

Only start after Run B, the packing/masking canary, and creation of the semantic `v2-clean` review
manifest. Initial profile: one epoch, rank 16, effective batch 32, `1e-4`, 3% warmup, 2,048 tokens,
packing enabled, assistant-only loss, checkpoint limit two. Evaluate at deliberate data milestones;
do not automatically continue to a second epoch.

## W&B policy

- Project: `qwen3-4b-synthgsm8k-5070`.
- Groups: `run-a-base`, `run-b-5k`, `run-b-post`,
  `run-c-historical-reconstruction`, and `run-d-full`.
- W&B stores non-secret configuration and metrics only.
- `WANDB_LOG_MODEL=false`; adapters, checkpoints, datasets, and raw generations stay local.
- TensorBoard remains enabled, and local JSON/JSONL artifacts are authoritative.
- Credential contents and account identity are never inspected or copied into repository files.

The monitor samples every 0.5 seconds: WSL available/used RAM, process RSS, swap use and cumulative
swap I/O, NVML VRAM, GPU/memory utilization, temperature, and power. Trainer logs record loss,
gradient norm, learning rate, input tokens, optimizer steps, runtime, and throughput. Before Run B,
the collator records exact raw, non-padding, padding, and supervised tokens, sequence-length mean
and p95, useful-token throughput, and tokens per optimizer step. Periodic checkpoints record write
duration, size, file count, and an artifact digest. Input tokens alone are not treated as equivalent
to supervised work.

## Resource estimates and stop conditions

Observed three-step compatibility peak at length 512 was approximately 4.35 GiB CUDA allocated and
4.91 GiB reserved, with approximately 3.09 GiB minimum available WSL RAM and no swap I/O. These are
reference measurements, not guarantees for length 1,024–4,096.

Planning estimates:

| Stage | Estimated wall time | Incremental disk | Important uncertainty |
|---|---:|---:|---|
| Run A deterministic full | roughly 2 hours | under 100 MB | generation length dominates |
| Run A sampled, four seeds | roughly 4× deterministic | under 400 MB | stochastic output length |
| Run B training | roughly 20–60 minutes | under 1 GB | 1,024-token padding/evaluation |
| Run C reconstruction | roughly 4–8 hours | under 2 GB | native TRL vs published Unsloth |
| Run D one epoch | roughly 2–6 hours | under 2 GB | packed token throughput |

Stop immediately on CUDA/Linux OOM, any CPU/disk device map, non-finite gradients/loss, sustained
swap activity, repeated checkpoint failure, decreasing WSL available memory without stabilization,
or less than 35 GiB filesystem free. Do not retry with weakened settings without a new recorded
configuration.

## Exact commands

All commands start in WSL:

```bash
cd ~/ml-lab
source scripts/configure-env.sh
```

Rebuild the pinned CPU artifacts if required:

```bash
.venv-train/bin/python -m ml_lab.synthgsm8k prepare-benchmark \
  --output-dir data/processed/gsm8k-740312a

.venv-train/bin/python -m ml_lab.synthgsm8k prepare-training \
  --output-dir data/processed/synthgsm8k-ebf8f27/run-b-5k \
  --train-count 5000 --validation-count 256 --max-length 1024 --seed 42
```

Validate without using the GPU or creating a W&B run:

```bash
.venv-train/bin/python -m ml_lab.train_sft \
  --config configs/synthgsm8k-run-b-part1.yaml --validate-only
.venv-train/bin/python -m ml_lab.train_sft \
  --config configs/synthgsm8k-run-b-resume.yaml --validate-only
.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-deterministic.yaml --validate-only
.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-sampled.yaml --validate-only
```

Run A canary, then full baseline when explicitly chosen:

```bash
.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-deterministic.yaml --limit 32

.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-deterministic.yaml

.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-sampled.yaml
```

Run B part 1 and explicit resume:

```bash
.venv-train/bin/python -m ml_lab.train_sft \
  --config configs/synthgsm8k-run-b-part1.yaml

B1_RUN="$(find runs -maxdepth 1 -type d -name 'synthgsm8k-run-b-part1-*' \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
test -d "$B1_RUN/checkpoints/checkpoint-128"

.venv-train/bin/python -m ml_lab.train_sft \
  --config configs/synthgsm8k-run-b-resume.yaml \
  --resume-from-checkpoint "$B1_RUN/checkpoints/checkpoint-128"
```

Evaluate the newest Run B adapter under identical tracks:

```bash
B2_RUN="$(find runs -maxdepth 1 -type d -name 'synthgsm8k-run-b-resume-*' \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
B2_ADAPTER="models/adapters/$(basename "$B2_RUN")"
test -f "$B2_ADAPTER/adapter_config.json"

.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-deterministic.yaml \
  --adapter "$B2_ADAPTER" --wandb-group run-b-post

.venv-train/bin/python -m ml_lab.gsm8k_eval \
  --config configs/gsm8k-base-sampled.yaml \
  --adapter "$B2_ADAPTER" --wandb-group run-b-post
```

Prepare later-stage data only after their gates are met:

```bash
.venv-train/bin/python -m ml_lab.synthgsm8k prepare-training \
  --output-dir data/processed/synthgsm8k-ebf8f27/run-c-34818 \
  --train-count 34818 --validation-count 0 --max-length 4096 --seed 42

.venv-train/bin/python -m ml_lab.synthgsm8k prepare-training \
  --output-dir data/processed/synthgsm8k-ebf8f27/run-d-full \
  --train-count 50418 --validation-count 0 --max-length 2048 --seed 42
```

The later full commands exist but are intentionally not part of the initial execution sequence:

```bash
.venv-train/bin/python -m ml_lab.train_sft \
  --config configs/synthgsm8k-run-c-historical-reconstruction.yaml

.venv-train/bin/python -m ml_lab.train_sft \
  --config configs/synthgsm8k-run-d-full.yaml
```

Do not run either command until its stated gate is satisfied and reviewed.

## Completion criteria for the 5070 test

Run B is a PASS only if all of the following are evidenced in local artifacts:

- exact model, training dataset, transformed subset, and benchmark identities recorded;
- NF4 transformer modules and all adapter targets on CUDA;
- only adapter parameters trainable;
- assistant-only masking verified for every prepared record;
- checkpoint-128 saved and resume reaches optimizer step 313;
- finite loss and per-step audited gradients;
- bounded CUDA/WSL memory with no OOM or sustained swap activity;
- final adapter saved, reloaded in a fresh process, and used on RTX 5070;
- full base and post-adapter results use identical test data/parser/generation settings;
- local raw results, summaries, runtime samples, config, package versions, and hashes retained;
- W&B logging success is reported separately and never substitutes for local artifacts.

No longer Run C or D job begins automatically after Run B.
