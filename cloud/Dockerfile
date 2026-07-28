# syntax=docker/dockerfile:1.7
# CUDA 13.0 is used because the checked uv.lock resolves PyTorch's CUDA 13 runtime.
# Official NVIDIA multi-platform index digest recorded 2026-07-27.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.8.2

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    VIRTUAL_ENV=/workspace/ml-lab/.venv \
    PATH=/workspace/ml-lab/.venv/bin:/root/.local/bin:${PATH} \
    ML_LAB_HOME=/workspace/ml-lab \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_CACHE=/workspace/cache/huggingface/hub \
    HF_DATASETS_CACHE=/workspace/cache/datasets \
    TORCH_HOME=/workspace/cache/torch \
    TRITON_CACHE_DIR=/workspace/cache/triton \
    TMPDIR=/workspace/cache/temporary \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git htop jq openssh-client python3.12 \
      python3.12-dev python3.12-venv rsync tmux \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

WORKDIR /workspace/ml-lab

# Manifests first: dependency changes invalidate this layer; source edits do not.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.12

COPY README.md ./README.md
COPY src ./src
RUN uv sync --frozen --no-dev --python 3.12

RUN mkdir -p /workspace/data /workspace/models /workspace/runs \
      /workspace/cache/huggingface /workspace/cache/datasets /workspace/cache/torch \
      /workspace/cache/triton /workspace/cache/temporary

HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
  CMD python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() > 0; x=torch.ones(1,device='cuda'); assert x.is_cuda; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1

CMD ["bash"]
