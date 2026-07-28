"""Secret-safe, executable validation of the ML lab GPU software stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .run_metadata import atomic_write_json, utc_now

PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "bitsandbytes",
    "huggingface-hub",
    "kaggle",
    "kagglehub",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        first_line = (result.stdout or result.stderr).splitlines()
        return {
            "available": result.returncode == 0,
            "version": first_line[0].strip() if first_line else None,
            "exit_code": result.returncode,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": type(exc).__name__}


def _vllm_version(lab: Path) -> dict[str, Any]:
    python = lab / ".venv-vllm" / "bin" / "python"
    if not python.is_file():
        return {"available": False, "version": None, "reason": "vLLM environment absent"}
    code = "import importlib.metadata; print(importlib.metadata.version('vllm'))"
    return _command_version([str(python), "-c", code])


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                continue
    return total


def _cuda_smoke(torch: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": False}
    torch.cuda.reset_peak_memory_stats()
    try:
        device = torch.device("cuda:0")
        left = torch.randn((1024, 1024), device=device, dtype=torch.float32, requires_grad=True)
        right = torch.randn((1024, 1024), device=device, dtype=torch.float32, requires_grad=True)
        output = left @ right
        loss = output.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        gradients_finite = bool(torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all())
        result.update(
            {
                "output_device": str(output.device),
                "backward_succeeded": left.grad is not None and right.grad is not None,
                "gradients_finite": gradients_finite,
                "passed": output.is_cuda and gradients_finite,
            }
        )
        if torch.cuda.is_bf16_supported():
            a = torch.randn((512, 512), device=device, dtype=torch.bfloat16, requires_grad=True)
            b = torch.randn((512, 512), device=device, dtype=torch.bfloat16, requires_grad=True)
            bf16_output = a @ b
            bf16_output.float().mean().backward()
            torch.cuda.synchronize()
            result["bf16"] = {
                "tested": True,
                "passed": bf16_output.dtype == torch.bfloat16 and bool(torch.isfinite(a.grad).all()),
            }
        else:
            result["bf16"] = {"tested": False, "passed": False}
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def _bitsandbytes_smoke(torch: Any) -> dict[str, Any]:
    """Exercise the CUDA NF4 dequantize/matmul/backward path used by QLoRA."""
    result: dict[str, Any] = {"passed": False, "operation": "Linear4bit NF4 forward/backward"}
    try:
        import bitsandbytes as bnb

        layer = bnb.nn.Linear4bit(
            64,
            32,
            bias=False,
            compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            compress_statistics=True,
            quant_type="nf4",
        )
        layer.requires_grad_(False)
        layer = layer.to("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        values = torch.randn(4, 64, device="cuda", dtype=dtype, requires_grad=True)
        output = layer(values)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        finite = values.grad is not None and bool(torch.isfinite(values.grad).all())
        result.update(
            {
                "version": _version("bitsandbytes"),
                "output_device": str(output.device),
                "gradient_finite": finite,
                "passed": output.is_cuda and finite,
            }
        )
    except Exception as exc:
        result.update({"version": _version("bitsandbytes"), "error": f"{type(exc).__name__}: {exc}"})
    return result


def collect_report(lab: Path | None = None) -> dict[str, Any]:
    lab = (lab or Path(os.environ.get("ML_LAB_HOME", Path(__file__).resolve().parents[2]))).resolve()
    disk = shutil.disk_usage(lab)
    report: dict[str, Any] = {
        "timestamp": utc_now(),
        "lab_path": str(lab),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {name: _version(name) for name in PACKAGE_NAMES},
        "vllm": _vllm_version(lab),
        "ollama": _command_version(["ollama", "--version"]),
        "disk": {
            "lab_usage_bytes": _directory_size(lab),
            "filesystem_total_bytes": disk.total,
            "filesystem_free_bytes": disk.free,
        },
        "checks": {},
    }
    try:
        import torch
    except ImportError as exc:
        report["pytorch"] = {"available": False, "error": str(exc)}
        report["checks"] = {"cuda": False, "sm_120": False, "bf16": False, "bitsandbytes": False}
        report["overall_passed"] = False
        return report

    cuda_available = torch.cuda.is_available()
    pytorch: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }
    report["pytorch"] = pytorch
    if not cuda_available:
        report["checks"] = {"cuda": False, "sm_120": False, "bf16": False, "bitsandbytes": False}
        report["overall_passed"] = False
        return report
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    architectures = torch.cuda.get_arch_list()
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    pytorch["gpu"] = {
        "name": properties.name,
        "compute_capability": list(capability),
        "total_vram_bytes": total_vram,
        "free_vram_bytes": free_vram,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "compiled_architectures": architectures,
    }
    sm_120 = capability == (12, 0) and any(item in {"sm_120", "compute_120"} for item in architectures)
    cuda_smoke = _cuda_smoke(torch)
    bnb_smoke = _bitsandbytes_smoke(torch)
    report["cuda_smoke"] = cuda_smoke
    report["bitsandbytes_smoke"] = bnb_smoke
    report["checks"] = {
        "cuda": cuda_available and cuda_smoke.get("passed", False),
        "sm_120": sm_120,
        "bf16": bool(cuda_smoke.get("bf16", {}).get("passed", False)),
        "bitsandbytes": bnb_smoke.get("passed", False),
    }
    report["overall_passed"] = all(report["checks"].values())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the report to this path")
    parser.add_argument("--no-fail", action="store_true", help="report failures without a nonzero exit")
    args = parser.parse_args(argv)
    report = collect_report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        atomic_write_json(args.json, report)
    return 0 if report["overall_passed"] or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

