"""Typed configuration parsing shared by local and remote training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(slots=True)
class TrainingConfig:
    model_name_or_path: str
    dataset: str
    model_revision: str = "main"
    dataset_config: str | None = None
    dataset_revision: str | None = None
    dataset_split: str = "train"
    mode: Literal["lora", "qlora", "full"] = "qlora"
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    mixed_precision: Literal["bf16", "fp16", "no", "auto"] = "auto"
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.0
    logging_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    validation_fraction: float = 0.1
    packing: bool = False
    assistant_only_loss: bool = False
    completion_only_loss: bool | None = None
    chat_template_enable_thinking: bool | None = None
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    wandb_project: str | None = None
    wandb_group: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_log_model: bool = False
    run_name: str = "sft"
    runs_dir: str = "runs"
    adapter_output_dir: str | None = None
    resume_from_checkpoint: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    low_cpu_mem_usage: bool = True
    max_grad_norm: float = 1.0
    use_cpu: bool = False
    verify_adapter_after_train: bool = True
    generation_prompt: str = "Briefly explain why reproducible experiments use a fixed random seed."
    generation_max_new_tokens: int = 24
    require_reviewed_data: bool = False
    require_provenance: bool = False
    # Distributed execution is still launched by `accelerate launch`; these options
    # let Transformers cooperate with the selected Accelerate/FSDP2/DeepSpeed mode.
    distributed_backend: Literal["none", "fsdp2", "deepspeed"] = "none"
    fsdp: str | list[str] | None = None
    fsdp_config: dict[str, Any] = field(default_factory=dict)
    deepspeed: str | dict[str, Any] | None = None
    activation_checkpointing: bool = True
    cpu_ram_efficient_loading: bool = True
    cpu_offload: bool = False
    state_dict_type: Literal["FULL_STATE_DICT", "SHARDED_STATE_DICT"] = "SHARDED_STATE_DICT"
    adapter_only_checkpointing: bool = True
    gradient_sync_each_batch: bool = True
    final_checkpoint_consolidation: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingConfig:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown training configuration keys: {', '.join(unknown)}")
        try:
            config = cls(**dict(value))
        except TypeError as exc:
            raise ValueError(f"invalid training configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path is required")
        if not self.dataset.strip():
            raise ValueError("dataset is required")
        if self.mode == "qlora" and not self.load_in_4bit:
            raise ValueError("QLoRA requires load_in_4bit=true")
        if self.mode != "qlora" and self.load_in_4bit:
            raise ValueError("load_in_4bit is only valid in qlora mode")
        if self.mode == "full" and self.adapter_only_checkpointing:
            raise ValueError("full mode cannot use adapter_only_checkpointing")
        if self.quant_type != "nf4" and self.mode == "qlora":
            raise ValueError("this lab requires NF4 for QLoRA")
        positive_ints = {
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "generation_max_new_tokens": self.generation_max_new_tokens,
        }
        for name, number in positive_ints.items():
            if number <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.save_total_limit > 2:
            raise ValueError("save_total_limit may not exceed 2 in this disk-constrained lab")
        reporters = set(self.report_to)
        if len(reporters) != len(self.report_to):
            raise ValueError("report_to entries must be unique")
        unsupported_reporters = sorted(reporters - {"tensorboard", "wandb"})
        if unsupported_reporters:
            raise ValueError(
                "unsupported report_to integrations: " + ", ".join(unsupported_reporters)
            )
        if "wandb" in reporters and not (self.wandb_project or "").strip():
            raise ValueError("wandb_project is required when report_to includes wandb")
        if "wandb" not in reporters and any(
            (self.wandb_project, self.wandb_group, self.wandb_tags, self.wandb_log_model)
        ):
            raise ValueError("W&B settings require report_to to include wandb")
        if self.distributed_backend == "fsdp2" and self.deepspeed is not None:
            raise ValueError("FSDP2 and DeepSpeed cannot be enabled together")
        if self.distributed_backend == "deepspeed" and self.fsdp is not None:
            raise ValueError("DeepSpeed and FSDP cannot be enabled together")
        if self.cpu_offload and self.distributed_backend == "none":
            raise ValueError("CPU offload is only a distributed-training option")

    @property
    def effective_batch_size_per_data_parallel_replica(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_config(path: str | Path) -> TrainingConfig:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: expected a YAML mapping")
    return TrainingConfig.from_mapping(value)
