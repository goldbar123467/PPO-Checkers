"""Optional continued pretraining (CPT) on a licensed, cleaned domain corpus.

This is intentionally distinct from SFT. It uses a causal language-model objective
over EOS-delimited packed documents and requires a corpus manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .data_validation import read_json_records
from .run_metadata import RunMetadataStore, base_runtime_metadata, make_run_id, sha256_file, utc_now
from .train_sft import disk_preflight


@dataclass(slots=True)
class ContinuedPretrainConfig:
    model_name_or_path: str
    model_revision: str
    dataset: str
    corpus_manifest: str
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_field: str = "text"
    streaming: bool = False
    sequence_length: int = 2048
    max_steps: int = 1000
    max_tokens: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    validation_fraction: float = 0.01
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    resume_from_checkpoint: str | None = None
    evaluation_material: list[str] = dataclasses.field(default_factory=list)
    run_name: str = "continued-pretrain"

    @classmethod
    def load(cls, path: Path) -> "ContinuedPretrainConfig":
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("continued-pretraining config must be a mapping")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown CPT keys: {', '.join(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_revision in {"", "main"}:
            raise ValueError("CPT requires a pinned, non-main model_revision")
        if self.sequence_length <= 0 or self.max_steps <= 0:
            raise ValueError("sequence_length and max_steps must be positive")
        if self.save_total_limit > 2:
            raise ValueError("save_total_limit may not exceed 2")
        if self.streaming and self.max_steps <= 0:
            raise ValueError("streaming training requires max_steps")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"name", "version", "license", "provenance", "deduplicated", "review_status"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"corpus manifest missing: {', '.join(missing)}")
    if not manifest["license"] or not manifest["provenance"]:
        raise ValueError("corpus license and provenance must be documented")
    if manifest["deduplicated"] is not True or manifest["review_status"] != "approved":
        raise ValueError("CPT corpus must be deduplicated and approved")
    return manifest


def _document_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _evaluation_hashes(paths: Iterable[str]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        for record in read_json_records(Path(path).expanduser()):
            text = record.get("text")
            if isinstance(text, str):
                hashes.add(_document_hash(text))
    return hashes


def validate_local_corpus(config: ContinuedPretrainConfig) -> dict[str, Any]:
    path = Path(config.dataset).expanduser()
    if not path.is_file():
        return {"local": False, "duplicates": None, "contamination": None}
    records = read_json_records(path)
    texts = [record.get(config.text_field) for record in records]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"all corpus records require non-empty {config.text_field!r}")
    hashes = [_document_hash(str(text)) for text in texts]
    duplicate_count = len(hashes) - len(set(hashes))
    if duplicate_count:
        raise ValueError(f"corpus contains {duplicate_count} exact normalized duplicates")
    overlap = set(hashes) & _evaluation_hashes(config.evaluation_material)
    if overlap:
        raise ValueError(f"corpus contaminates evaluation material with {len(overlap)} exact matches")
    return {
        "local": True,
        "records": len(records),
        "duplicates": 0,
        "contamination": 0,
        "sha256": sha256_file(path),
    }


def run(config: ContinuedPretrainConfig, *, allow_low_disk: bool = False) -> Path:
    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    lab = Path(os.environ.get("ML_LAB_HOME", Path(__file__).resolve().parents[2])).resolve()
    disk_preflight(lab, allow_low_disk)
    manifest_path = Path(config.corpus_manifest).expanduser()
    manifest = _load_manifest(manifest_path)
    corpus_validation = validate_local_corpus(config)
    run_dir = lab / "runs" / make_run_id(config.run_name)
    run_dir.mkdir(parents=True)
    store = RunMetadataStore.create(
        run_dir / "metadata.json",
        {
            **base_runtime_metadata(lab, ("torch", "transformers", "datasets", "accelerate")),
            "status": "initializing",
            "task": "continued_pretraining",
            "config": dataclasses.asdict(config),
            "corpus_manifest": manifest,
            "corpus_validation": corpus_validation,
            "started_at": utc_now(),
        },
    )
    source = Path(config.dataset).expanduser()
    if source.is_file():
        loaded = load_dataset("json", data_files=str(source), streaming=config.streaming)
    else:
        loaded = load_dataset(
            config.dataset, config.dataset_config, streaming=config.streaming
        )
    dataset = loaded[config.dataset_split] if isinstance(loaded, DatasetDict) or config.dataset_split in loaded else loaded
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: Mapping[str, list[str]]) -> dict[str, list[list[int]]]:
        # Every document gets an EOS boundary before packing; tokens never concatenate
        # across a document boundary without an explicit separator.
        documents = [text + tokenizer.eos_token for text in batch[config.text_field]]
        return tokenizer(documents, add_special_tokens=False, truncation=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    def pack(batch: Mapping[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for tokens in batch["input_ids"]:
            concatenated.extend(tokens)
        usable = len(concatenated) // config.sequence_length * config.sequence_length
        blocks = [
            concatenated[index : index + config.sequence_length]
            for index in range(0, usable, config.sequence_length)
        ]
        return {"input_ids": blocks, "attention_mask": [[1] * len(block) for block in blocks]}

    packed = tokenized.map(pack, batched=True)
    evaluation = None
    if not config.streaming and config.validation_fraction > 0:
        split = packed.train_test_split(test_size=config.validation_fraction, seed=config.seed)
        packed, evaluation = split["train"], split["test"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    arguments = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        logging_dir=str(run_dir / "tensorboard"),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        eval_strategy="steps" if evaluation is not None else "no",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to=["tensorboard"],
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=packed,
        eval_dataset=evaluation,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    metrics = dict(result.metrics)
    observed_tokens = int(metrics.get("train_samples", 0)) * config.sequence_length
    if "train_loss" in metrics and evaluation is not None:
        evaluation_metrics = trainer.evaluate()
        metrics.update(evaluation_metrics)
        if "eval_loss" in evaluation_metrics:
            metrics["validation_perplexity"] = math.exp(min(20, evaluation_metrics["eval_loss"]))
    store.update(status="completed", completed_at=utc_now(), metrics=metrics, observed_token_count=observed_tokens)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-low-disk", action="store_true")
    args = parser.parse_args(argv)
    config = ContinuedPretrainConfig.load(args.config)
    _load_manifest(Path(config.corpus_manifest).expanduser())
    validation = validate_local_corpus(config)
    if args.validate_only:
        print(json.dumps({"valid": True, "corpus": validation}, sort_keys=True))
        return 0
    run(config, allow_low_disk=args.allow_low_disk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

