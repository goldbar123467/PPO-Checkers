# ML Lab: RTX 5070 + Ephemeral Vast.ai Workers

This repository is a reproducible Linux-filesystem ML laboratory for WSL2 Ubuntu. The local machine is the source of truth; optional Vast.ai instances are disposable execution workers. It assumes an NVIDIA GeForce RTX 5070 (Blackwell, compute capability 12.0, 12 GB VRAM), Python 3.12, and at least 35 GB continuously free.

The lab separates dependency domains deliberately:

- `.venv-train` contains PyTorch, Transformers, TRL, PEFT, bitsandbytes, Accelerate, data tools, Jupyter, Kaggle, and tests.
- `.venv-vllm` contains vLLM and the exact PyTorch/CUDA packages selected by vLLM.
- Ollama is an external localhost service. It is useful for simple model lifecycle and GGUF-style local workflows; vLLM supplies an OpenAI-compatible, throughput-oriented Hugging Face serving path.

An 8B model cannot be full-parameter trained in 12 GB. Local 8B experiments therefore default to 4-bit NF4 QLoRA with double quantization, BF16 compute when supported, checkpointing, batch size 1, and accumulation. The 4B system must pass the promotion gate in [AGENTS.md](AGENTS.md) before substantive 8B work.

## Architecture and storage

All active files live below `$HOME/ml-lab`, never `/mnt/c`:

```text
source/config/tests  -> Git
raw data             -> data/raw (immutable and untracked)
validated data       -> data/processed (versioned by manifest, untracked payload)
replaceable caches   -> cache/{huggingface,datasets,torch,triton,ollama,temporary}
model artifacts      -> models/{base,adapters,merged,gguf}
run state            -> runs/{checkpoints,logs,metadata,tensorboard,remote}
remote build/plans   -> cloud/
```

Every operational script sources `scripts/configure-env.sh`, which centralizes cache and temporary paths. It does not contain tokens and no shell startup file is modified.

## Environments and diagnostics

Enter the training environment:

```bash
cd ~/ml-lab
source scripts/activate-train.sh
```

Enter the serving environment:

```bash
cd ~/ml-lab
source scripts/activate-vllm.sh
```

Run hardware/package diagnostics and all unit/static checks:

```bash
cd ~/ml-lab
make doctor
make test
make mypy
```

`make doctor` performs real CUDA matrix multiplication, backward, finite-gradient, and BF16 checks, reports peak allocation/reservation, and exercises bitsandbytes rather than treating an import as proof.

## Authentication

Hugging Face is intentionally not configured during setup. Start the supported interactive browser/device flow later—never append a token argument:

```bash
cd ~/ml-lab
make hf-login
```

The helper runs `hf auth login`, then `hf auth whoami`. Tokens belong only in the Hugging Face secure cache, never `.env`, Git, source, Markdown, YAML, or shell history.

Kaggle credentials, when discovered, live at `~/ml-lab/.secrets/kaggle/kaggle.json` with mode `600`; `KAGGLE_CONFIG_DIR` points there. A conventional Ubuntu credential may be moved and securely symlinked back; a Windows credential is copied while the Windows original remains. Validate auth without revealing fields:

```bash
cd ~/ml-lab
make kaggle-test
```

## Jupyter and TensorBoard

Both bind only to localhost:

```bash
cd ~/ml-lab
make jupyter
make tensorboard
```

Select the `ML Lab - RTX 5070` kernel in Jupyter. TensorBoard reads `runs/tensorboard` on `http://127.0.0.1:6006`.

## Dataset format and validation

Training accepts JSON/JSONL or a Hugging Face dataset identifier. A local JSONL may contain one `text` value per record:

```json
{"text":"### User\nExplain recursion.\n### Assistant\nRecursion solves a problem using smaller instances of itself."}
```

Or conversational `messages`:

```json
{"messages":[{"role":"system","content":"Be a careful tutor."},{"role":"user","content":"What is a stack?"},{"role":"assistant","content":"A stack is a last-in, first-out collection."}]}
```

Before a real run, place unreviewed source in `data/raw`, generate a reviewed/versioned product in `data/processed`, and record provenance, license, review status, grade band, categories, duplicate group, and deterministic split. Never train directly from `data/raw`.

## Training

Run the tiny end-to-end GPU setup test (2–5 steps, adapter reload, inference, TensorBoard):

```bash
cd ~/ml-lab
make smoke-train
```

For a 4B NF4 QLoRA run, copy the template, replace the visible model placeholder with an ungated or authorized ID, pin `model_revision`, and set the reviewed dataset:

```bash
cd ~/ml-lab
cp configs/qlora-4b.yaml configs/experiment-4b-001.yaml
$EDITOR configs/experiment-4b-001.yaml
source scripts/activate-train.sh
python -m ml_lab.train_sft --config configs/experiment-4b-001.yaml
```

The starter 4B effective batch is `1 × 8 = 8` sequences per optimizer update (one GPU). Higher sequence length increases activation memory; accumulation trades time for effective batch without making a single microbatch larger.

The 8B template is a compatibility-oriented NF4 QLoRA profile, not permission to skip the gate:

```bash
cd ~/ml-lab
cp configs/qlora-8b.yaml configs/experiment-8b-compat-001.yaml
$EDITOR configs/experiment-8b-compat-001.yaml
source scripts/activate-train.sh
python -m ml_lab.train_sft --config configs/experiment-8b-compat-001.yaml
```

Its effective batch is `1 × 16 = 16` sequences per update and sequence length is reduced to 1024 for VRAM headroom. Resume explicitly:

```bash
python -m ml_lab.train_sft --config configs/experiment-4b-001.yaml \
  --resume-from-checkpoint runs/checkpoints/<run-id>/checkpoint-<step>
```

Only two checkpoints are retained by default. Runs save adapters rather than duplicate base weights unless a reviewed config says otherwise.

### Qwen3-4B SynthGSM8K RTX 5070 experiment

The staged baseline, 5K resume test, historical reconstruction, and gated full-dataset plan is in
[reports/qwen3-4b-synthgsm8k-5070-plan.md](reports/qwen3-4b-synthgsm8k-5070-plan.md).
It pins the model, training corpus, untouched GSM8K benchmark, deterministic transformed split,
Qwen thinking template, strict answer parser, W&B groups, resource metrics, and stop conditions.
Data preparation and configuration validation are complete; the document does not authorize or
automatically launch the substantial GPU stages.

## Local inference services

Serve a supported Hugging Face model through vLLM on localhost:

```bash
cd ~/ml-lab
make serve-vllm MODEL=<organization/model> PORT=8000
# in another terminal
make test-vllm PORT=8000
```

The server defaults to `127.0.0.1`, GPU memory utilization `0.85`, and maximum context 2048. Export `VLLM_API_KEY` in the shell if desired; scripts do not print it.

Start and test Ollama (the setup smoke model is intentionally below 1.5 GB):

```bash
cd ~/ml-lab
make ollama-start
make ollama-test
ollama run qwen3:0.6b
make ollama-stop
```

`make ollama-test` refuses to pass unless `ollama ps` proves GPU use.

## Disk management

```bash
cd ~/ml-lab
make disk
make clean-dry-run
scripts/cleanup-cache.sh --execute
```

Cleanup defaults to dry run and only targets temporary, Triton, and Torch cache contents. It preserves raw data, Hugging Face model cache, adapters, and checkpoints. Training/model scripts warn below 50 GB and refuse below 35 GB unless `ML_LAB_ALLOW_LOW_DISK=1` is set after explicit review.

## Vast.ai workflow

The same `src/ml_lab` package executes locally and remotely; no Python implementation is duplicated in `cloud/`. No paid instance is created during setup. Detailed profiles, container, cost guardrails, data hashing, recovery, and commands are in [cloud/vast/README.md](cloud/vast/README.md). The expected lifecycle is:

```bash
make vast-auth-status
make vast-search PROFILE=cloud/vast/profiles/4b-qlora-24gb.yaml
make vast-plan PROFILE=cloud/vast/profiles/4b-qlora-24gb.yaml
# paid actions require explicit MAX_HOURLY, MAX_HOURS, execution, and cost acceptance
make vast-create PROFILE=cloud/vast/profiles/4b-qlora-24gb.yaml \
  MAX_HOURLY=<price> MAX_HOURS=<hours> EXECUTE=1 ACCEPT_COST=1
```

Always run local smoke, remote smoke, artifact sync/hash verification, and local adapter reload before destruction. Multi-GPU YAML parsing does not count as runtime validation.

## Troubleshooting

### `sm_120` unsupported

```bash
cd ~/ml-lab
source scripts/activate-train.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())'
make doctor
```

If `sm_120` is absent or a kernel reports incompatibility, do not accept CPU fallback. Reinstall the current official stable CUDA 12.8+ backend with `uv pip install --python .venv-train/bin/python --reinstall torch torchvision torchaudio --torch-backend=auto`; use the official nightly only after stable is proven incompatible and record the change.

### CUDA unavailable

```bash
nvidia-smi
cd ~/ml-lab
.venv-train/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

If `nvidia-smi` fails, stop GPU installation and repair Windows/WSL integration from Windows. Do not install an Ubuntu `nvidia-driver-*` package.

### bitsandbytes failure

```bash
cd ~/ml-lab
make doctor
source scripts/activate-train.sh
python -m bitsandbytes
```

An import is not proof; preserve the failing CUDA operation. Until the doctor succeeds, mark QLoRA unavailable rather than silently switching methods.

### CUDA OOM

```bash
nvidia-smi
make ollama-stop
```

Then reduce `max_seq_length`, microbatch size, LoRA rank, or enabled evaluation/generation; keep accumulation if effective batch matters. Confirm no vLLM process owns VRAM. Do not blindly retry.

### WSL systemd unavailable

```bash
systemctl is-system-running
cd ~/ml-lab
scripts/start-ollama.sh
scripts/stop-ollama.sh
```

The lab's user-controlled scripts do not require systemd.

### Ollama using CPU

```bash
nvidia-smi
ollama ps
tail -n 100 ~/ml-lab/runs/logs/ollama.log
```

Stop other GPU workloads, restart Ollama, rerun `make ollama-test`, and do not claim GPU inference unless the processor column shows GPU layers.

### vLLM conflicts or Blackwell kernel failure

```bash
cd ~/ml-lab
.venv-vllm/bin/python -c 'import torch,vllm; print(vllm.__version__, torch.__version__, torch.version.cuda)'
uv pip check --python .venv-vllm/bin/python
tail -n 200 runs/logs/vllm-8000.log
```

Never solve this by mixing `.venv-train` into `.venv-vllm`. Follow the current official CUDA 12.8+ wheel route, then official nightly if the stable wheel specifically lacks Blackwell support.

### Insufficient disk

```bash
cd ~/ml-lab
make disk
make clean-dry-run
scripts/cleanup-cache.sh --execute
```

Do not override the 35 GB floor until artifacts and caches have been reviewed.
