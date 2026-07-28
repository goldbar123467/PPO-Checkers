#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=_common.sh
source "$(dirname -- "$0")/_common.sh"

instance="${1:-}"
if [[ "$instance" == "--instance" ]]; then instance="${2:-}"; fi
require_instance_id "$instance"
command='set -Eeuo pipefail; echo "Kernel: $(uname -r)"; echo "Python: $(python3.12 --version 2>&1)"; echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader; echo "Disk:"; df -h /workspace; if [[ -x /workspace/ml-lab/.venv/bin/python ]]; then /workspace/ml-lab/.venv/bin/python - <<"PY"
import torch
print("torch", torch.__version__)
print("torch_cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("device", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
x=torch.randn(64,64,device="cuda",requires_grad=True); y=(x@x).mean(); y.backward(); torch.cuda.synchronize(); print("cuda_backward", bool(torch.isfinite(x.grad).all()))
PY
else echo "training_environment: not synchronized yet"; fi'
remote_shell "$instance" "$command"
