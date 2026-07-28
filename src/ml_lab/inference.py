"""Local Hugging Face base/adaptor inference and optional verified adapter merge."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .run_metadata import atomic_write_json, utc_now


def load_model(
    model_name_or_path: str,
    *,
    adapter: str | None = None,
    revision: str = "main",
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; CPU fallback is forbidden for this inference check"
        )
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError(
            f"expected compute capability (12, 0), got {torch.cuda.get_device_capability(0)}"
        )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    tokenizer_source = adapter or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=None if adapter else revision
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    device_map = getattr(model, "hf_device_map", {})
    invalid_map = {
        name: location
        for name, location in device_map.items()
        if location not in {0, "cuda", "cuda:0"}
    }
    if invalid_map:
        raise RuntimeError(f"CPU/disk offload is forbidden, but device map contains {invalid_map}")
    cpu_parameters = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "cpu"
    ]
    if cpu_parameters:
        raise RuntimeError(f"inference model parameters remain on CPU: {cpu_parameters[:10]}")
    model.eval()
    return model, tokenizer


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    chat: bool = False,
    system_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    import torch

    if chat:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    options: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        options["temperature"] = temperature
    with torch.inference_mode():
        generated = model.generate(**inputs, **options)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    new_ids = generated[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    tokens = int(new_ids.shape[-1])
    metrics = {
        "generated_tokens": tokens,
        "latency_seconds": duration,
        "tokens_per_second": tokens / duration if duration else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_inference_verified": str(device).startswith("cuda")
        and "RTX 5070" in torch.cuda.get_device_name(0),
    }
    if not metrics["cuda_inference_verified"]:
        raise RuntimeError(f"inference did not use the expected RTX 5070: {metrics}")
    return text, metrics


def merge_adapter(model: Any, tokenizer: Any, destination: Path) -> None:
    lab = Path(os.environ.get("ML_LAB_HOME", Path(__file__).resolve().parents[2])).resolve()
    expected_root = (lab / "models" / "merged").resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(expected_root):
        raise ValueError("merged output must be under models/merged")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"merge destination is non-empty: {destination}")
    if not hasattr(model, "merge_and_unload"):
        raise TypeError("--merge-output requires a PEFT adapter")
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    if not (destination / "config.json").is_file():
        raise RuntimeError("merged artifact verification failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--adapter")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--system-prompt")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--merge-output", type=Path)
    args = parser.parse_args(argv)
    model, tokenizer = load_model(
        args.model,
        adapter=args.adapter,
        revision=args.revision,
        load_in_4bit=args.load_in_4bit,
    )
    text, metrics = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        chat=args.chat,
        system_prompt=args.system_prompt,
    )
    metrics.update(
        {
            "completed_at": utc_now(),
            "model": args.model,
            "revision": args.revision,
            "adapter": args.adapter,
            "load_in_4bit": args.load_in_4bit,
            "chat_template_used": args.chat,
            "prompt": args.prompt,
        }
    )
    print(text)
    print(json.dumps(metrics, sort_keys=True))
    if args.run_dir:
        lab = Path(os.environ.get("ML_LAB_HOME", Path(__file__).resolve().parents[2])).resolve()
        run_dir = args.run_dir.resolve()
        if not run_dir.is_relative_to(lab / "runs") or not run_dir.is_dir():
            raise ValueError("--run-dir must be an existing directory under ML_LAB_HOME/runs")
        (run_dir / "generation.txt").write_text(text + "\n", encoding="utf-8")
        atomic_write_json(run_dir / "inference.json", metrics)
        metadata_path = run_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["fresh_process_inference"] = metrics
            metadata["final_status"] = "passed" if metrics["cuda_inference_verified"] else "failed"
            atomic_write_json(metadata_path, metadata)
    if args.merge_output:
        if args.load_in_4bit:
            raise ValueError(
                "merge from a 4-bit base is intentionally unsupported; reload in BF16 first"
            )
        merge_adapter(model, tokenizer, args.merge_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
