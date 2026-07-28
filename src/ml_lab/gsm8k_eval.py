"""Evaluate a pinned Qwen3-4B base or adapter on the untouched GSM8K test split."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .inference import load_model
from .run_metadata import append_jsonl, atomic_write_json, make_run_id, sha256_file, utc_now
from .synthgsm8k import (
    GSM8K_DATASET_REVISION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    canonicalize_number,
    extract_final_answer,
)
from .train_sft import ResourceMonitor, _assert_exclusive_gpu, disk_preflight


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    model: str
    model_revision: str
    benchmark: str
    benchmark_dataset: str
    benchmark_revision: str
    track: str
    seeds: list[int]
    max_new_tokens: int = 512
    limit: int | None = None
    adapter: str | None = None
    wandb_project: str | None = None
    wandb_group: str | None = None

    @classmethod
    def load(cls, path: Path) -> EvaluationConfig:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("evaluation config must be a YAML mapping")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown evaluation config keys: {', '.join(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config

    def validate(self) -> None:
        if self.model != QWEN_MODEL_ID or self.model_revision != QWEN_MODEL_REVISION:
            raise ValueError("evaluation requires the exact pinned Qwen3-4B model revision")
        if self.benchmark_dataset != "openai/gsm8k":
            raise ValueError("evaluation benchmark must be openai/gsm8k")
        if self.benchmark_revision != GSM8K_DATASET_REVISION:
            raise ValueError("evaluation requires the exact pinned GSM8K dataset revision")
        if self.track not in {"deterministic", "sampled"}:
            raise ValueError("track must be deterministic or sampled")
        if not self.seeds or any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must be a non-empty integer list")
        if self.track == "deterministic" and len(self.seeds) != 1:
            raise ValueError("deterministic evaluation requires exactly one seed")
        if self.max_new_tokens <= 0 or (self.limit is not None and self.limit <= 0):
            raise ValueError("max_new_tokens and optional limit must be positive")
        if not (self.wandb_project or "").strip():
            raise ValueError("wandb_project is required for this tracked evaluation")


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            required = {"id", "question", "expected_answer"}
            missing = sorted(required - set(value))
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {', '.join(missing)}")
            records.append(value)
    if not records:
        raise ValueError("benchmark is empty")
    if len({str(item["id"]) for item in records}) != len(records):
        raise ValueError("benchmark ids are not unique")
    return records


def _generation_prompt(question: str) -> str:
    return (
        "Solve this grade-school mathematics problem carefully. Show your reasoning, then "
        "put only the canonical numeric answer on the final line in the form `#### <number>`.\n\n"
        + question.strip()
    )


def _generate_one(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    max_new_tokens: int,
    sampled: bool,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    messages = [{"role": "user", "content": _generation_prompt(question)}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"evaluation model is not on CUDA: {device}")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    options: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sampled,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sampled:
        options.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    count = int(new_tokens.shape[-1])
    return text, {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": count,
        "latency_seconds": duration,
        "tokens_per_second": count / duration if duration else None,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device": str(device),
    }


def _summary(results: Sequence[dict[str, Any]], duration: float) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize zero results")
    correct = sum(bool(item["correct"]) for item in results)
    parsed = sum(item["parsed_answer"] is not None for item in results)
    output_tokens = sum(int(item["performance"]["output_tokens"]) for item in results)
    generation_seconds = sum(float(item["performance"]["latency_seconds"]) for item in results)
    summary = {
        "examples": len(results),
        "correct": correct,
        "exact_match_accuracy": correct / len(results),
        "parsed_answers": parsed,
        "invalid_answer_count": len(results) - parsed,
        "invalid_answer_rate": (len(results) - parsed) / len(results),
        "output_tokens": output_tokens,
        "generation_seconds": generation_seconds,
        "wall_seconds": duration,
        "aggregate_output_tokens_per_second": (
            output_tokens / generation_seconds if generation_seconds else None
        ),
        "mean_output_tokens": output_tokens / len(results),
        "peak_cuda_allocated_bytes": max(
            int(item["performance"]["peak_cuda_allocated_bytes"]) for item in results
        ),
        "peak_cuda_reserved_bytes": max(
            int(item["performance"]["peak_cuda_reserved_bytes"]) for item in results
        ),
    }
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_seed[int(result["seed"])].append(result)
        by_prompt[str(result["prompt_id"])].append(result)
    seed_accuracies = {
        str(seed): sum(bool(item["correct"]) for item in items) / len(items)
        for seed, items in sorted(by_seed.items())
    }
    summary["accuracy_by_seed"] = seed_accuracies
    summary["mean_seed_accuracy"] = statistics.fmean(seed_accuracies.values())
    summary["seed_accuracy_population_stddev"] = (
        statistics.pstdev(seed_accuracies.values()) if len(seed_accuracies) > 1 else 0.0
    )
    if len(by_seed) > 1:
        majority_correct = 0
        majority_ties = 0
        for items in by_prompt.values():
            votes = Counter(
                str(item["parsed_answer"]) for item in items if item["parsed_answer"] is not None
            )
            if not votes:
                continue
            highest = max(votes.values())
            winners = sorted(answer for answer, count in votes.items() if count == highest)
            if len(winners) != 1:
                majority_ties += 1
                continue
            majority_correct += winners[0] == str(items[0]["expected_answer"])
        summary["majority_vote_accuracy"] = majority_correct / len(by_prompt)
        summary["majority_vote_ties"] = majority_ties
    return summary


def evaluate(
    benchmark: Path,
    *,
    adapter: Path | None,
    track: str,
    seeds: Sequence[int],
    max_new_tokens: int,
    limit: int | None,
    wandb_project: str | None,
    wandb_group: str | None,
) -> Path:
    import torch

    if track not in {"deterministic", "sampled"}:
        raise ValueError("track must be deterministic or sampled")
    if track == "deterministic" and len(seeds) != 1:
        raise ValueError("deterministic evaluation requires exactly one seed")
    lab = Path(os.environ.get("ML_LAB_HOME", Path(__file__).resolve().parents[2])).resolve()
    disk_preflight(lab, allow_low_disk=False)
    _assert_exclusive_gpu()
    records = _load_benchmark(benchmark)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]
    run_id = make_run_id(f"gsm8k-{track}-{'adapter' if adapter else 'base'}")
    run_dir = lab / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        "model": QWEN_MODEL_ID,
        "model_revision": QWEN_MODEL_REVISION,
        "adapter": str(adapter.resolve()) if adapter else None,
        "benchmark": str(benchmark.resolve()),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_dataset_revision": GSM8K_DATASET_REVISION,
        "track": track,
        "seeds": list(seeds),
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "load_in_4bit": True,
        "enable_thinking": True,
        "sampling": (
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
            if track == "sampled"
            else {"do_sample": False, "warning": "legacy comparison track"}
        ),
    }
    atomic_write_json(run_dir / "config.json", configuration)
    monitor = ResourceMonitor(run_dir / "resource-samples.jsonl")
    monitor.start()
    wandb_run = None
    started = time.monotonic()
    try:
        if wandb_project:
            import wandb

            os.environ.setdefault("WANDB_LOG_MODEL", "false")
            os.environ.setdefault("WANDB_WATCH", "false")
            wandb_run = wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=run_id,
                config=configuration,
                job_type="evaluation",
            )
        model, tokenizer = load_model(
            QWEN_MODEL_ID,
            adapter=str(adapter) if adapter else None,
            revision=QWEN_MODEL_REVISION,
            load_in_4bit=True,
        )
        results: list[dict[str, Any]] = []
        for seed in seeds:
            for record in records:
                response, performance = _generate_one(
                    model,
                    tokenizer,
                    str(record["question"]),
                    max_new_tokens=max_new_tokens,
                    sampled=track == "sampled",
                    seed=seed,
                )
                parsed = extract_final_answer(response)
                expected = Decimal(str(record["expected_answer"]))
                result = {
                    "prompt_id": record["id"],
                    "seed": seed,
                    "expected_answer": canonicalize_number(expected),
                    "parsed_answer": canonicalize_number(parsed) if parsed is not None else None,
                    "correct": parsed == expected if parsed is not None else False,
                    "response": response,
                    "performance": performance,
                }
                results.append(result)
                append_jsonl(run_dir / "results.jsonl", result)
                if wandb_run and len(results) % 25 == 0:
                    partial = _summary(results, time.monotonic() - started)
                    wandb_run.log(
                        {
                            "evaluation/examples": len(results),
                            "evaluation/accuracy": partial["exact_match_accuracy"],
                            "evaluation/invalid_rate": partial["invalid_answer_rate"],
                            "evaluation/output_tokens_per_second": partial[
                                "aggregate_output_tokens_per_second"
                            ],
                        },
                        step=len(results),
                    )
        duration = time.monotonic() - started
        summary = _summary(results, duration)
        summary["resource_monitor"] = monitor.stop()
        summary["completed_at"] = utc_now()
        summary["gpu_name"] = torch.cuda.get_device_name(0)
        summary["compute_capability"] = list(torch.cuda.get_device_capability(0))
        summary["status"] = "passed"
        atomic_write_json(run_dir / "summary.json", summary)
        if wandb_run:
            wandb_run.log(
                {
                    f"final/{key}": value
                    for key, value in summary.items()
                    if isinstance(value, int | float)
                }
            )
            wandb_run.finish(exit_code=0)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return run_dir
    except BaseException:
        try:
            resource_summary = monitor.stop()
        except RuntimeError as monitor_error:
            resource_summary = {"monitor_error": f"{type(monitor_error).__name__}: {monitor_error}"}
        atomic_write_json(
            run_dir / "failure.json",
            {"failed_at": utc_now(), "resource_monitor": resource_summary},
        )
        if wandb_run:
            wandb_run.finish(exit_code=1)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--track", choices=["deterministic", "sampled"])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-group")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        loaded = EvaluationConfig.load(args.config)
        benchmark = Path(loaded.benchmark)
        adapter = args.adapter or (Path(loaded.adapter) if loaded.adapter else None)
        track = args.track or loaded.track
        seeds = args.seeds or loaded.seeds
        max_new_tokens = args.max_new_tokens or loaded.max_new_tokens
        limit = args.limit if args.limit is not None else loaded.limit
        wandb_project = args.wandb_project or loaded.wandb_project
        wandb_group = args.wandb_group or loaded.wandb_group
    else:
        required = {
            "--benchmark": args.benchmark,
            "--track": args.track,
            "--seeds": args.seeds,
            "--max-new-tokens": args.max_new_tokens,
            "--wandb-project": args.wandb_project,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("without --config, required arguments are: " + ", ".join(missing))
        benchmark = args.benchmark
        adapter = args.adapter
        track = args.track
        seeds = args.seeds
        max_new_tokens = args.max_new_tokens
        limit = args.limit
        wandb_project = args.wandb_project
        wandb_group = args.wandb_group
    assert benchmark is not None
    assert track is not None
    assert seeds is not None
    assert max_new_tokens is not None
    records = _load_benchmark(benchmark)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "record_count": len(records),
                    "benchmark_sha256": sha256_file(benchmark),
                    "track": track,
                    "seeds": seeds,
                },
                sort_keys=True,
            )
        )
        return 0
    output = evaluate(
        benchmark,
        adapter=adapter,
        track=track,
        seeds=seeds,
        max_new_tokens=max_new_tokens,
        limit=limit,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
