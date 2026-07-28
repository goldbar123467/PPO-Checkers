"""Configurable SFT/LoRA/QLoRA training for local and ephemeral remote GPUs."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import TrainingConfig, load_training_config
from .data_validation import (
    dataset_manifest,
    read_json_records,
    render_chatml,
    validate_records,
)
from .run_metadata import (
    RunMetadataStore,
    atomic_write_json,
    base_runtime_metadata,
    hash_paths,
    make_run_id,
    sha256_json,
    utc_now,
)

PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "bitsandbytes",
    "safetensors",
    "wandb",
)


def _lab_home() -> Path:
    return Path(os.environ.get("ML_LAB_HOME", Path(__file__).resolve().parents[2])).resolve()


def _free_disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def disk_preflight(path: Path, allow_low_disk: bool) -> float:
    free = _free_disk_gib(path)
    if free < 35 and not allow_low_disk:
        raise RuntimeError(
            f"only {free:.1f} GiB free; refusing training below 35 GiB. "
            "Free space or explicitly pass --allow-low-disk after reviewing the risk."
        )
    if free < 50:
        print(f"WARNING: filesystem has only {free:.1f} GiB free (50 GiB warning threshold).")
    return free


def _system_snapshot(lab: Path) -> dict[str, Any]:
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(lab)
    return {
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_in_bytes_since_boot": swap.sin,
        "swap_out_bytes_since_boot": swap.sout,
        "disk_free_bytes": disk.free,
    }


def _assert_exclusive_gpu() -> dict[str, Any]:
    """Fail if a serving process or another CUDA compute process is active."""
    import psutil

    serving: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            name = str(process.info.get("name") or "").lower()
            command = " ".join(process.info.get("cmdline") or []).lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if name == "ollama" or "ollama serve" in command or "vllm" in name or "vllm" in command:
            serving.append({"pid": process.pid, "name": name})
    if serving:
        raise RuntimeError(f"serving process is active; refusing shared-GPU run: {serving}")

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
    if completed.returncode != 0:
        raise RuntimeError(f"cannot inspect CUDA processes: nvidia-smi exit {completed.returncode}")
    compute_processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) < 3 or not pieces[0].isdigit():
            continue
        pid = int(pieces[0])
        item = {"pid": pid, "name": pieces[1], "used_memory_mib": pieces[2]}
        if pid != os.getpid():
            compute_processes.append(item)
    if compute_processes:
        raise RuntimeError(
            f"another CUDA compute process is active; refusing shared-GPU run: {compute_processes}"
        )
    return {"serving_processes": serving, "other_cuda_processes": compute_processes}


class ResourceMonitor:
    """Sample WSL and GPU memory without changing the training configuration."""

    def __init__(self, destination: Path, interval_seconds: float = 0.5) -> None:
        self.destination = destination
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource monitor was already started")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="ml-lab-resource-monitor", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            import psutil
            import pynvml
            import torch

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            process = psutil.Process(os.getpid())
            with self.destination.open("w", encoding="utf-8") as output:
                while not self._stop.is_set():
                    memory = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sample = {
                        "monotonic_seconds": time.monotonic(),
                        "ram_available_bytes": memory.available,
                        "ram_used_bytes": memory.used,
                        "process_rss_bytes": process.memory_info().rss,
                        "swap_used_bytes": swap.used,
                        "swap_in_bytes_since_boot": swap.sin,
                        "swap_out_bytes_since_boot": swap.sout,
                        "gpu_used_bytes": gpu.used,
                        "gpu_free_bytes": gpu.free,
                        "gpu_utilization_percent": utilization.gpu,
                        "gpu_memory_utilization_percent": utilization.memory,
                        "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        "gpu_power_mw": pynvml.nvmlDeviceGetPowerUsage(handle),
                        "cuda_allocated_bytes": (
                            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
                        ),
                        "cuda_reserved_bytes": (
                            int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
                        ),
                    }
                    self.samples.append(sample)
                    output.write(json.dumps(sample, sort_keys=True) + "\n")
                    output.flush()
                    self._stop.wait(self.interval_seconds)
            pynvml.nvmlShutdown()
        except Exception as exc:  # Preserve monitor failure as a measurement failure.
            self._error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("resource monitor did not stop cleanly")
        if self._error:
            raise RuntimeError(f"resource monitor failed: {self._error}")
        if not self.samples:
            raise RuntimeError("resource monitor collected no samples")
        swap_in = [int(sample["swap_in_bytes_since_boot"]) for sample in self.samples]
        swap_out = [int(sample["swap_out_bytes_since_boot"]) for sample in self.samples]
        activity = [
            swap_in[index] > swap_in[index - 1] or swap_out[index] > swap_out[index - 1]
            for index in range(1, len(self.samples))
        ]
        longest = 0
        current = 0
        for active in activity:
            current = current + 1 if active else 0
            longest = max(longest, current)
        return {
            "sample_count": len(self.samples),
            "sample_interval_seconds": self.interval_seconds,
            "minimum_ram_available_bytes": min(
                int(sample["ram_available_bytes"]) for sample in self.samples
            ),
            "peak_swap_used_bytes": max(int(sample["swap_used_bytes"]) for sample in self.samples),
            "peak_process_rss_bytes": max(
                int(sample["process_rss_bytes"]) for sample in self.samples
            ),
            "peak_gpu_used_bytes_nvml": max(
                int(sample["gpu_used_bytes"]) for sample in self.samples
            ),
            "mean_gpu_utilization_percent": sum(
                int(sample["gpu_utilization_percent"]) for sample in self.samples
            )
            / len(self.samples),
            "peak_gpu_temperature_c": max(
                int(sample["gpu_temperature_c"]) for sample in self.samples
            ),
            "peak_gpu_power_mw": max(int(sample["gpu_power_mw"]) for sample in self.samples),
            "peak_cuda_allocated_bytes_sampled": max(
                int(sample["cuda_allocated_bytes"]) for sample in self.samples
            ),
            "peak_cuda_reserved_bytes_sampled": max(
                int(sample["cuda_reserved_bytes"]) for sample in self.samples
            ),
            "swap_in_delta_bytes": swap_in[-1] - swap_in[0],
            "swap_out_delta_bytes": swap_out[-1] - swap_out[0],
            "longest_consecutive_swap_activity_samples": longest,
            "sustained_swap_activity": longest >= 3,
        }


class TokenAccountingCollator:
    """Wrap a TRL collator and count the exact tokens delivered to model steps."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.raw_input_tokens = 0
            self.non_padding_tokens = 0
            self.supervised_tokens = 0
            self.collated_sequences = 0
            self.sequence_lengths: list[int] = []

    def __call__(self, features: Any) -> Any:
        batch = self.inner(features)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]
        raw = int(input_ids.numel())
        if attention_mask is None:
            non_padding = raw
            lengths = [int(input_ids.shape[-1])] * int(input_ids.shape[0])
            supervised = int((labels != -100).sum().item())
        else:
            non_padding = int(attention_mask.sum().item())
            lengths = [int(value) for value in attention_mask.sum(dim=-1).tolist()]
            supervised = int(((labels != -100) & attention_mask.bool()).sum().item())
        with self._lock:
            self.raw_input_tokens += raw
            self.non_padding_tokens += non_padding
            self.supervised_tokens += supervised
            self.collated_sequences += int(input_ids.shape[0])
            self.sequence_lengths.extend(lengths)
        return batch

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lengths = sorted(self.sequence_lengths)
            count = len(lengths)
            p95_index = max(0, math.ceil(0.95 * count) - 1) if count else 0
            return {
                "raw_input_tokens": self.raw_input_tokens,
                "non_padding_tokens": self.non_padding_tokens,
                "supervised_tokens": self.supervised_tokens,
                "collated_sequences": self.collated_sequences,
                "padding_tokens": self.raw_input_tokens - self.non_padding_tokens,
                "padding_fraction": (
                    (self.raw_input_tokens - self.non_padding_tokens) / self.raw_input_tokens
                    if self.raw_input_tokens
                    else None
                ),
                "mean_sequence_length": (sum(lengths) / count if count else None),
                "p95_sequence_length": lengths[p95_index] if lengths else None,
            }


def _dtype(config: TrainingConfig, torch: Any) -> Any:
    if config.compute_dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.compute_dtype]


def _load_dataset(config: TrainingConfig) -> tuple[Any, dict[str, Any]]:
    from datasets import Dataset, DatasetDict, load_dataset

    candidate = Path(config.dataset).expanduser()
    if candidate.is_file():
        records = read_json_records(candidate)
        report = validate_records(
            records,
            require_reviewed=config.require_reviewed_data,
            require_provenance=config.require_provenance,
        )
        if not report.ok:
            details = "; ".join(f"record {i.index}: {i.message}" for i in report.errors[:10])
            raise ValueError(f"dataset validation failed: {details}")
        dataset = Dataset.from_list(records)
        identity = {
            "kind": "local",
            "path": str(candidate.resolve()),
            **dataset_manifest(records),
        }
    else:
        loaded = load_dataset(
            config.dataset,
            config.dataset_config,
            revision=config.dataset_revision,
        )
        if isinstance(loaded, DatasetDict):
            if config.dataset_split not in loaded:
                raise ValueError(
                    f"dataset has splits {sorted(loaded)}, not requested {config.dataset_split!r}"
                )
            dataset = loaded[config.dataset_split]
        else:
            dataset = loaded
        # Validate before training. Large hub datasets are traversed once; this is
        # intentional because unvalidated remote input must not start an expensive run.
        issues = validate_records(
            list(dataset),
            require_reviewed=config.require_reviewed_data,
            require_provenance=config.require_provenance,
        )
        if not issues.ok:
            details = "; ".join(f"record {i.index}: {i.message}" for i in issues.errors[:10])
            raise ValueError(f"dataset validation failed: {details}")
        identity = {
            "kind": "huggingface",
            "identifier": config.dataset,
            "configuration": config.dataset_config,
            "revision": config.dataset_revision,
            "split": config.dataset_split,
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "record_count": len(dataset),
        }
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    return dataset, identity


def _render_dataset(dataset: Any, tokenizer: Any, config: TrainingConfig) -> Any:
    columns = set(dataset.column_names)
    if "text" in columns:
        return dataset
    if "messages" not in columns:
        raise ValueError("dataset must contain text or messages")

    if config.completion_only_loss:

        def split_conversation(example: Mapping[str, Any]) -> dict[str, Any]:
            messages = example["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    "completion-only conversation must contain a prompt and assistant response"
                )
            if not isinstance(messages[-1], Mapping) or messages[-1].get("role") != "assistant":
                raise ValueError(
                    "completion-only conversation must end with one assistant response"
                )
            prompt = messages[:-1]
            completion = [messages[-1]]
            if any(
                message.get("role") == "assistant"
                for message in prompt
                if isinstance(message, Mapping)
            ):
                raise ValueError("compatibility dataset must contain one final assistant turn only")
            transformed: dict[str, Any] = {
                "prompt": prompt,
                "completion": completion,
            }
            if config.chat_template_enable_thinking is not None:
                transformed["chat_template_kwargs"] = {
                    "enable_thinking": config.chat_template_enable_thinking
                }
            return transformed

        return dataset.map(
            split_conversation,
            remove_columns=dataset.column_names,
            desc="Preparing native-template assistant completions",
        )

    def render(example: Mapping[str, Any]) -> dict[str, str]:
        messages = example["messages"]
        try:
            value = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except (AttributeError, ValueError, TypeError):
            value = render_chatml(messages)
        return {"text": value}

    return dataset.map(render, desc="Rendering conversations")


def _split_dataset(dataset: Any, fraction: float, seed: int) -> tuple[Any, Any | None]:
    if "split" in dataset.column_names:
        split_values = {str(value) for value in dataset["split"]}
        unsupported = sorted(split_values - {"train", "validation"})
        if unsupported:
            raise ValueError(f"unsupported explicit dataset split values: {unsupported}")
        train = dataset.filter(lambda example: example["split"] == "train")
        validation = dataset.filter(lambda example: example["split"] == "validation")
        if len(train) == 0:
            raise ValueError("explicit dataset split contains no training records")
        return train, validation if len(validation) else None
    if fraction <= 0 or len(dataset) < 2:
        return dataset, None
    count = max(1, round(len(dataset) * fraction))
    count = min(count, len(dataset) - 1)
    split = dataset.train_test_split(test_size=count, seed=seed, shuffle=True)
    return split["train"], split["test"]


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    try:
        return {item.name for item in dataclasses.fields(cls)}
    except TypeError:
        return set()


def _sft_arguments(config: TrainingConfig, run_dir: Path, has_eval: bool, torch: Any) -> Any:
    from trl import SFTConfig

    accepted = _dataclass_field_names(SFTConfig)
    values: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "logging_dir": str(run_dir / "tensorboard"),
        "run_name": run_dir.name,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "warmup_ratio": config.warmup_ratio,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "logging_first_step": True,
        "include_num_input_tokens_seen": True,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "save_strategy": "steps",
        "optim": config.optimizer,
        "lr_scheduler_type": config.lr_scheduler_type,
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": config.report_to,
        "project": config.wandb_project,
        "bf16": config.compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported(),
        "fp16": config.compute_dtype == "float16",
        "tf32": bool(torch.cuda.is_available()),
        "max_grad_norm": config.max_grad_norm,
        "remove_unused_columns": True,
        "dataset_text_field": "text",
        "packing": config.packing,
        "shuffle_dataset": False,
        "assistant_only_loss": config.assistant_only_loss,
        "completion_only_loss": config.completion_only_loss,
        "use_cpu": config.use_cpu,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    if "max_length" in accepted:
        values["max_length"] = config.max_seq_length
    elif "max_seq_length" in accepted:
        values["max_seq_length"] = config.max_seq_length
    strategy_key = "eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"
    values[strategy_key] = "steps" if has_eval else "no"
    if config.distributed_backend == "fsdp2":
        values["fsdp"] = config.fsdp or "full_shard auto_wrap"
        fsdp_config = dict(config.fsdp_config)
        fsdp_config.setdefault("activation_checkpointing", config.activation_checkpointing)
        fsdp_config.setdefault("cpu_ram_efficient_loading", config.cpu_ram_efficient_loading)
        fsdp_config.setdefault("offload_params", config.cpu_offload)
        fsdp_config.setdefault("state_dict_type", config.state_dict_type)
        values["fsdp_config"] = fsdp_config
    elif config.distributed_backend == "deepspeed":
        if config.deepspeed is None:
            raise ValueError("distributed_backend=deepspeed requires deepspeed configuration")
        values["deepspeed"] = config.deepspeed
    return SFTConfig(**{key: value for key, value in values.items() if key in accepted})


def _configure_reporting(config: TrainingConfig) -> dict[str, Any]:
    """Configure non-secret experiment metadata for optional integrations."""
    if "wandb" not in config.report_to:
        return {"integrations": list(config.report_to), "wandb_enabled": False}
    os.environ["WANDB_PROJECT"] = str(config.wandb_project)
    os.environ["WANDB_LOG_MODEL"] = "true" if config.wandb_log_model else "false"
    os.environ.setdefault("WANDB_WATCH", "false")
    if config.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = config.wandb_group
    if config.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(config.wandb_tags)
    return {
        "integrations": list(config.report_to),
        "wandb_enabled": True,
        "wandb_project": config.wandb_project,
        "wandb_group": config.wandb_group,
        "wandb_tags": list(config.wandb_tags),
        "wandb_log_model": config.wandb_log_model,
    }


def _build_model_and_peft(config: TrainingConfig, torch: Any) -> tuple[Any, Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = _dtype(config, torch)
    quantization_config = None
    if config.mode == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    model_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "trust_remote_code": config.trust_remote_code,
        "dtype": dtype,
        "attn_implementation": config.attn_implementation,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }
    if config.mode == "qlora" and torch.cuda.is_available():
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if config.mode == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    peft_config = None
    if config.mode in {"lora", "qlora"}:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias=config.lora_bias,
        )
        model = get_peft_model(model, peft_config)
    return model, tokenizer, peft_config


def _parameter_audit(model: Any, mode: str) -> dict[str, Any]:
    import bitsandbytes as bnb

    logical_sizes = {
        id(module.weight): int(module.in_features) * int(module.out_features)
        for module in model.modules()
        if isinstance(module, bnb.nn.Linear4bit)
    }
    storage_elements = sum(parameter.numel() for parameter in model.parameters())
    total = sum(
        logical_sizes.get(id(parameter), parameter.numel()) for parameter in model.parameters()
    )
    trainable_named = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = sum(number for _, number in trainable_named)
    if trainable <= 0:
        raise RuntimeError("model has no trainable parameters")
    if mode in {"lora", "qlora"}:
        unexpected = [name for name, _ in trainable_named if "lora_" not in name]
        if unexpected:
            raise RuntimeError(f"unexpected non-adapter trainable parameters: {unexpected[:10]}")
    return {
        "total_parameters": total,
        "packed_storage_elements": storage_elements,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "trainable_tensor_names": [name for name, _ in trainable_named],
    }


def _quantization_and_lora_audit(model: Any, config: TrainingConfig) -> dict[str, Any]:
    import bitsandbytes as bnb

    quantized: list[dict[str, str]] = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            device = str(module.weight.device)
            quantized.append({"name": name, "device": device})
    if config.mode == "qlora" and not quantized:
        raise RuntimeError("QLoRA model contains no bitsandbytes Linear4bit modules")
    off_gpu = [item for item in quantized if not item["device"].startswith("cuda")]
    if off_gpu:
        raise RuntimeError(f"quantized modules are not on CUDA: {off_gpu[:10]}")

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        invalid = {
            name: location
            for name, location in device_map.items()
            if location not in {0, "cuda", "cuda:0"}
        }
        if invalid:
            raise RuntimeError(f"CPU/disk offload is forbidden, but device map contains {invalid}")
    cpu_parameters = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "cpu"
    ]
    if cpu_parameters:
        raise RuntimeError(
            f"model parameters remain on CPU; offload/fallback is forbidden: {cpu_parameters[:10]}"
        )

    adapter_modules = sorted(
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and module.lora_A
    )
    requested = (
        list(config.lora_target_modules)
        if isinstance(config.lora_target_modules, list)
        else [config.lora_target_modules]
    )
    categories = sorted({name.rsplit(".", 1)[-1] for name in adapter_modules})
    missing_categories = sorted(set(requested) - set(categories))
    unexpected_categories = sorted(set(categories) - set(requested))
    if missing_categories or unexpected_categories:
        raise RuntimeError(
            "LoRA target mismatch: "
            f"missing={missing_categories}, unexpected={unexpected_categories}"
        )
    return {
        "quantized_module_count": len(quantized),
        "quantized_modules": quantized,
        "all_quantized_modules_on_cuda": not off_gpu,
        "device_map": dict(device_map) if isinstance(device_map, Mapping) else device_map,
        "cpu_parameter_count": len(cpu_parameters),
        "requested_target_categories": requested,
        "matched_target_categories": categories,
        "lora_target_module_count": len(adapter_modules),
        "lora_target_modules": adapter_modules,
    }


def _audit_assistant_masking(
    trainer: Any,
    tokenizer: Any,
    *,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    dataset = trainer.train_dataset
    supervised_counts: list[int] = []
    prompt_counts: list[int] = []
    sequence_lengths: list[int] = []
    for index, example in enumerate(dataset):
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        if len(input_ids) != len(labels):
            raise RuntimeError(f"record {index}: input/label length mismatch")
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        prompt_encoding = tokenizer.apply_chat_template(example["prompt"], **template_kwargs)
        prompt_ids = list(prompt_encoding["input_ids"])
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"record {index}: native chat-template prompt prefix mismatch")
        if any(label != -100 for label in labels[: len(prompt_ids)]):
            raise RuntimeError(f"record {index}: system/user/non-assistant token is supervised")
        supervised = labels[len(prompt_ids) :]
        if not supervised or not any(label != -100 for label in supervised):
            raise RuntimeError(f"record {index}: no supervised assistant tokens")
        if any(label == -100 for label in supervised):
            raise RuntimeError(
                f"record {index}: assistant response contains unexpectedly masked tokens"
            )
        if supervised != input_ids[len(prompt_ids) :]:
            raise RuntimeError(f"record {index}: assistant labels do not match input tokens")
        supervised_counts.append(len(supervised))
        prompt_counts.append(len(prompt_ids))
        sequence_lengths.append(len(input_ids))

    shortest = min(range(len(dataset)), key=lambda item: len(dataset[item]["input_ids"]))
    longest = max(range(len(dataset)), key=lambda item: len(dataset[item]["input_ids"]))
    batch = trainer.data_collator([dataset[shortest], dataset[longest]])
    padding_mask = batch["attention_mask"] == 0
    padding_tokens = int(padding_mask.sum().item())
    if padding_tokens <= 0:
        raise RuntimeError(
            "mask audit could not create padding; dataset examples need varied lengths"
        )
    if not bool((batch["labels"][padding_mask] == -100).all()):
        raise RuntimeError("padding tokens are not masked with label -100")
    return {
        "verified": True,
        "record_count": len(dataset),
        "records_with_supervised_assistant_tokens": len(supervised_counts),
        "minimum_supervised_tokens": min(supervised_counts),
        "maximum_supervised_tokens": max(supervised_counts),
        "masked_prompt_tokens": sum(prompt_counts),
        "minimum_sequence_tokens": min(sequence_lengths),
        "maximum_sequence_tokens": max(sequence_lengths),
        "padding_tokens_checked": padding_tokens,
        "system_user_nonassistant_labels_are_minus_100": True,
        "padding_labels_are_minus_100": True,
        "native_chat_template_used": True,
        "chat_template_enable_thinking": enable_thinking,
    }


def _create_trainer(
    model: Any, tokenizer: Any, args: Any, train: Any, evaluation: Any
) -> tuple[
    Any,
    list[dict[str, Any]],
    TokenAccountingCollator,
    list[dict[str, Any]],
]:
    import inspect

    from transformers import TrainerCallback
    from trl import SFTTrainer

    gradient_audit: list[dict[str, Any]] = []

    class GradientAuditCallback(TrainerCallback):
        def on_pre_optimizer_step(
            self, callback_args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
        ) -> Any:
            del callback_args, kwargs
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            finite = bool(gradients) and all(
                bool(gradient.isfinite().all()) for gradient in gradients
            )
            gradient_audit.append(
                {
                    "optimizer_step": int(state.global_step) + 1,
                    "gradient_tensor_count": len(gradients),
                    "all_gradients_finite": finite,
                }
            )
            if not finite:
                raise RuntimeError("non-finite or missing adapter gradients before optimizer step")
            return control

    parameters = inspect.signature(SFTTrainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train,
        "eval_dataset": evaluation,
        "callbacks": [GradientAuditCallback()],
    }
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**kwargs)
    uncounted_collator = trainer.data_collator
    token_accounting = TokenAccountingCollator(uncounted_collator)
    trainer.data_collator = token_accounting
    original_get_eval_dataloader = trainer.get_eval_dataloader

    def uncounted_eval_dataloader(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        active_collator = self.data_collator
        self.data_collator = uncounted_collator
        try:
            return original_get_eval_dataloader(*call_args, **call_kwargs)
        finally:
            self.data_collator = active_collator

    trainer.get_eval_dataloader = types.MethodType(uncounted_eval_dataloader, trainer)
    checkpoint_audit: list[dict[str, Any]] = []
    original_save_checkpoint = trainer._save_checkpoint

    def measured_save_checkpoint(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_save_checkpoint(*call_args, **call_kwargs)
        duration = time.perf_counter() - started
        checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
        checkpoint_audit.append(
            {
                "optimizer_step": int(self.state.global_step),
                "path": str(checkpoint.resolve()),
                "duration_seconds": duration,
                "file_count": len(files),
                "size_bytes": sum(path.stat().st_size for path in files),
                "artifact_sha256": hash_paths(files, base=checkpoint),
            }
        )
        return result

    trainer._save_checkpoint = types.MethodType(measured_save_checkpoint, trainer)
    return trainer, gradient_audit, token_accounting, checkpoint_audit


def _save_and_verify_adapter(
    config: TrainingConfig,
    trainer: Any,
    tokenizer: Any,
    adapter_dir: Path,
    torch: Any,
) -> dict[str, Any]:
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    if adapter_dir.exists() and any(adapter_dir.iterdir()):
        raise FileExistsError(
            f"adapter output already exists: {adapter_dir}; choose another path or archive it first"
        )
    save_started = time.perf_counter()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    save_duration = time.perf_counter() - save_started
    saved_files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    saved_artifact = {
        "save_duration_seconds": save_duration,
        "file_count": len(saved_files),
        "size_bytes": sum(path.stat().st_size for path in saved_files),
        "artifact_sha256": hash_paths(saved_files, base=adapter_dir),
    }
    required = adapter_dir / "adapter_config.json"
    if config.mode in {"lora", "qlora"} and not required.is_file():
        raise RuntimeError("adapter_config.json was not saved")
    if not config.verify_adapter_after_train or config.mode == "full":
        return {
            "saved": True,
            "reloaded": False,
            "generation_succeeded": False,
            **saved_artifact,
        }

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = _dtype(config, torch)
    kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "trust_remote_code": config.trust_remote_code,
        "dtype": dtype,
        "attn_implementation": config.attn_implementation,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }
    if config.mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)
    reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    reloaded.eval()
    verify_tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    inputs = verify_tokenizer(config.generation_prompt, return_tensors="pt")
    device = next(reloaded.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = reloaded.generate(
            **inputs,
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=verify_tokenizer.eos_token_id,
        )
    if output.shape[-1] <= inputs["input_ids"].shape[-1]:
        raise RuntimeError("adapter reload generation produced no new tokens")
    # Store only a fact and lengths, not generated text (which can contain dataset material).
    return {
        "saved": True,
        "reloaded": True,
        "generation_succeeded": True,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": int(output.shape[-1] - inputs["input_ids"].shape[-1]),
        **saved_artifact,
    }


def run_training(
    config: TrainingConfig, *, allow_low_disk: bool = False, allow_full_finetune: bool = False
) -> Path:
    if config.model_name_or_path == "REPLACE_WITH_HF_MODEL_ID":
        raise ValueError("replace model_name_or_path placeholder before training")
    if config.model_name_or_path == "Qwen/Qwen3-4B":
        try:
            pinned_revision = (
                len(config.model_revision) == 40 and int(config.model_revision, 16) >= 0
            )
        except ValueError:
            pinned_revision = False
        if not pinned_revision:
            raise ValueError(
                "Qwen/Qwen3-4B compatibility runs require an exact 40-character commit revision"
            )
    if config.mode == "full" and not allow_full_finetune:
        raise PermissionError("full fine-tuning requires explicit --allow-full-finetune")
    lab = _lab_home()
    free_before = disk_preflight(lab, allow_low_disk)
    process_preflight = _assert_exclusive_gpu()
    system_before = _system_snapshot(lab)
    run_id = make_run_id(config.run_name)
    run_dir = (lab / config.runs_dir / run_id).resolve()
    if not run_dir.is_relative_to(lab):
        raise ValueError("runs_dir must resolve inside ML_LAB_HOME")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = base_runtime_metadata(lab, PACKAGES)
    metadata.update(
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "initializing",
            "config": config.as_dict(),
            "configuration_sha256": sha256_json(config.as_dict()),
            "disk_free_gib_before": round(free_before, 3),
            "preflight": {"system": system_before, "processes": process_preflight},
            "started_at": utc_now(),
            "distributed": {
                "backend": config.distributed_backend,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "runtime_validated": int(os.environ.get("WORLD_SIZE", "1")) > 1,
            },
        }
    )
    store = RunMetadataStore.create(run_dir / "metadata.json", metadata)
    atomic_write_json(run_dir / "config.json", config.as_dict())
    started = time.monotonic()
    monitor: ResourceMonitor | None = None
    monitor_stopped = False
    resource_summary: dict[str, Any] | None = None
    try:
        import torch

        if not config.use_cpu and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; CPU fallback is not accepted for this training run"
            )
        if torch.cuda.get_device_capability(0) != (12, 0):
            capability = torch.cuda.get_device_capability(0)
            raise RuntimeError(f"expected RTX 5070 compute capability (12, 0), got {capability}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gpu_free_before, gpu_total = torch.cuda.mem_get_info(0)
        monitor = ResourceMonitor(run_dir / "resource-samples.jsonl")
        monitor.start()
        dataset, dataset_identity = _load_dataset(config)
        model, tokenizer, _ = _build_model_and_peft(config, torch)
        chat_template = tokenizer.chat_template or ""
        tokenizer_information = {
            "class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "chat_template_present": bool(chat_template),
            "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            "chat_template_has_generation_markers": "{% generation" in chat_template,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "enable_thinking": config.chat_template_enable_thinking,
        }
        if not chat_template:
            raise RuntimeError("model tokenizer has no native chat template")
        print(
            json.dumps(
                {
                    "model": config.model_name_or_path,
                    "revision": config.model_revision,
                    "tokenizer": tokenizer_information,
                },
                sort_keys=True,
            )
        )
        raw_train_dataset, raw_eval_dataset = _split_dataset(
            dataset, config.validation_fraction, config.seed
        )
        train_dataset = _render_dataset(raw_train_dataset, tokenizer, config)
        eval_dataset = (
            _render_dataset(raw_eval_dataset, tokenizer, config)
            if raw_eval_dataset is not None
            else None
        )
        audit = _parameter_audit(model, config.mode)
        quantization_audit = _quantization_and_lora_audit(model, config)
        print(
            json.dumps(
                {
                    "total_parameters": audit["total_parameters"],
                    "trainable_parameters": audit["trainable_parameters"],
                    "trainable_fraction": audit["trainable_fraction"],
                    "quantized_module_count": quantization_audit["quantized_module_count"],
                    "lora_target_modules": quantization_audit["lora_target_modules"],
                },
                sort_keys=True,
            )
        )
        reporting = _configure_reporting(config)
        arguments = _sft_arguments(config, run_dir, eval_dataset is not None, torch)
        trainer, gradient_audit, token_accounting, checkpoint_audit = _create_trainer(
            model, tokenizer, arguments, train_dataset, eval_dataset
        )
        masking_audit = (
            _audit_assistant_masking(
                trainer,
                tokenizer,
                enable_thinking=config.chat_template_enable_thinking,
            )
            if config.completion_only_loss
            else {"verified": False, "reason": "completion_only_loss is disabled"}
        )
        if config.completion_only_loss and not masking_audit["verified"]:
            raise RuntimeError("assistant-only masking audit did not pass")
        token_accounting.reset()
        store.update(
            status="training",
            dataset=dataset_identity,
            tokenizer=tokenizer_information,
            parameter_audit=audit,
            quantization_and_lora_audit=quantization_audit,
            masking_audit=masking_audit,
            reporting=reporting,
            gpu_memory_before={"free_bytes": gpu_free_before, "total_bytes": gpu_total},
            train_records=len(train_dataset),
            validation_records=len(eval_dataset) if eval_dataset is not None else 0,
        )
        result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        training_token_counts = token_accounting.snapshot()
        metrics = dict(result.metrics)
        if eval_dataset is not None:
            metrics.update({f"final_{key}": value for key, value in trainer.evaluate().items()})
        if not all(
            not isinstance(value, float) or (value == value and abs(value) != float("inf"))
            for value in metrics.values()
        ):
            raise RuntimeError("training produced non-finite metrics")
        actual_steps = int(trainer.state.global_step)
        if actual_steps <= 0:
            raise RuntimeError("training completed without an optimizer step")
        if config.max_steps > 0 and actual_steps != config.max_steps:
            raise RuntimeError(
                f"expected exactly {config.max_steps} optimizer steps, got {actual_steps}"
            )
        audited_steps = [int(item["optimizer_step"]) for item in gradient_audit]
        if (
            not audited_steps
            or audited_steps != list(range(audited_steps[0], actual_steps + 1))
            or not all(item["all_gradients_finite"] for item in gradient_audit)
        ):
            raise RuntimeError(
                f"gradient audit did not verify every optimizer step: {gradient_audit}"
            )
        training_logs = [dict(item) for item in trainer.state.log_history]
        loss_logs = [item for item in training_logs if "loss" in item]
        if not loss_logs or not all(math.isfinite(float(item["loss"])) for item in loss_logs):
            raise RuntimeError("logged training loss is missing or non-finite")
        gradient_norms = [float(item["grad_norm"]) for item in loss_logs if "grad_norm" in item]
        if not gradient_norms or not all(math.isfinite(value) for value in gradient_norms):
            raise RuntimeError(
                f"logged gradient norms are incomplete or non-finite: {gradient_norms}"
            )
        token_values = [
            float(item.get("num_input_tokens_seen", item.get("num_tokens", 0)))
            for item in training_logs
            if item.get("num_input_tokens_seen", item.get("num_tokens")) is not None
        ]
        observed_tokens = int(max(token_values, default=0))
        training_runtime = float(metrics.get("train_runtime", 0.0))
        metrics.update(
            {
                "optimizer_steps": actual_steps,
                "optimizer_steps_audited_this_process": len(audited_steps),
                "observed_input_tokens": observed_tokens,
                "tokens_per_second": observed_tokens / training_runtime
                if observed_tokens and training_runtime
                else None,
                "logged_losses": [float(item["loss"]) for item in loss_logs],
                "logged_gradient_norms": gradient_norms,
                "logged_learning_rates": [
                    float(item["learning_rate"]) for item in loss_logs if "learning_rate" in item
                ],
                "token_accounting": training_token_counts,
                "raw_input_tokens_per_second": (
                    training_token_counts["raw_input_tokens"] / training_runtime
                    if training_runtime
                    else None
                ),
                "non_padding_tokens_per_second": (
                    training_token_counts["non_padding_tokens"] / training_runtime
                    if training_runtime
                    else None
                ),
                "supervised_tokens_per_second": (
                    training_token_counts["supervised_tokens"] / training_runtime
                    if training_runtime
                    else None
                ),
                "optimizer_steps_per_second": (
                    len(audited_steps) / training_runtime if training_runtime else None
                ),
                "supervised_tokens_per_optimizer_step": (
                    training_token_counts["supervised_tokens"] / len(audited_steps)
                ),
                "checkpoint_audit": checkpoint_audit,
            }
        )
        adapter_dir = (
            (lab / config.adapter_output_dir).resolve()
            if config.adapter_output_dir
            else (lab / "models" / "adapters" / run_id).resolve()
        )
        if not adapter_dir.is_relative_to(lab / "models" / "adapters"):
            raise ValueError("adapter output must resolve under models/adapters")
        verification = _save_and_verify_adapter(config, trainer, tokenizer, adapter_dir, torch)
        resource_summary = monitor.stop()
        monitor_stopped = True
        if resource_summary["sustained_swap_activity"]:
            raise RuntimeError(f"sustained swap activity detected: {resource_summary}")
        duration = time.monotonic() - started
        peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        adapter_size = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())
        atomic_write_json(run_dir / "metrics.json", metrics)
        atomic_write_json(run_dir / "training-log-history.json", training_logs)
        store.update(
            status="completed",
            completed_at=utc_now(),
            duration_seconds=round(duration, 3),
            metrics=metrics,
            gradient_audit=gradient_audit,
            checkpoint_audit=checkpoint_audit,
            resume_audit={
                "requested_checkpoint": config.resume_from_checkpoint,
                "first_optimizer_step_audited": audited_steps[0],
                "final_optimizer_step": actual_steps,
                "optimizer_steps_executed_this_process": len(audited_steps),
            },
            adapter={"path": str(adapter_dir), "size_bytes": adapter_size, **verification},
            resource_summary=resource_summary,
            peak_vram_bytes={"allocated": peak_allocated, "reserved": peak_reserved},
            disk_free_gib_after=round(_free_disk_gib(lab), 3),
        )
        print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "adapter": str(adapter_dir)}))
        return run_dir
    except Exception as exc:
        if monitor is not None and not monitor_stopped:
            try:
                resource_summary = monitor.stop()
                monitor_stopped = True
            except Exception as monitor_exc:
                resource_summary = {"monitor_error": f"{type(monitor_exc).__name__}: {monitor_exc}"}
        duration = time.monotonic() - started
        diagnostic = f"{type(exc).__name__}: {exc}"
        recommendation = None
        if "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower():
            recommendation = (
                "CUDA OOM: reduce max_seq_length first, then LoRA rank; keep batch size 1, "
                "increase gradient accumulation to preserve effective batch, and confirm "
                "no serving process uses VRAM."
            )
            print(recommendation, file=sys.stderr)
        store.update(
            status="failed",
            failed_at=utc_now(),
            duration_seconds=round(duration, 3),
            resource_summary=resource_summary,
            failure={
                "diagnostic": diagnostic,
                "recommendation": recommendation,
                "traceback": "".join(traceback.format_exception(exc)),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument("--allow-low-disk", action="store_true")
    parser.add_argument("--allow-full-finetune", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_training_config(arguments.config)
    if arguments.resume_from_checkpoint:
        config.resume_from_checkpoint = arguments.resume_from_checkpoint
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "effective_batch_size": config.effective_batch_size_per_data_parallel_replica,
                }
            )
        )
        return 0
    run_training(
        config,
        allow_low_disk=arguments.allow_low_disk,
        allow_full_finetune=arguments.allow_full_finetune,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
