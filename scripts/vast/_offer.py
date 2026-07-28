#!/usr/bin/env python3
"""Normalize Vast offer JSON, validate a profile, and emit an immutable plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def scalar_profile(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line or line.startswith((" ", "\t")):
            continue
        key, raw = line.split(":", 1)
        raw = raw.split(" #", 1)[0].strip().strip("'\"")
        if raw.lower() in {"true", "false"}:
            value: Any = raw.lower() == "true"
        else:
            try:
                value = float(raw) if "." in raw else int(raw)
            except ValueError:
                value = raw
        result[key.strip()] = value
    return result


def offers_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        for key in ("offers", "bundles", "results", "instances"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [v for v in nested if isinstance(v, dict)]
        if "id" in value:
            return [value]
    return []


def number(offer: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = offer.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def gb(value: float) -> float:
    return value / 1024.0 if value > 1024 else value


def normalized(offer: dict[str, Any]) -> dict[str, Any]:
    gpu_count = int(number(offer, "num_gpus", default=1))
    per_gpu = gb(number(offer, "gpu_ram", "gpu_mem", default=0))
    total_gpu = gb(number(offer, "gpu_total_ram", default=per_gpu * gpu_count))
    return {
        "offer_id": int(number(offer, "id", "ask_contract_id")),
        "machine_id": int(number(offer, "machine_id")) or None,
        "gpu_model": str(offer.get("gpu_name") or offer.get("gpu_model") or "unknown"),
        "gpu_count": gpu_count,
        "vram_per_gpu_gb": round(per_gpu, 3),
        "total_gpu_vram_gb": round(total_gpu, 3),
        "system_ram_gb": round(gb(number(offer, "cpu_ram")), 3),
        "cpu_cores": round(number(offer, "cpu_cores_effective", "cpu_cores"), 2),
        "disk_available_gb": round(number(offer, "disk_space"), 3),
        "download_mbps": round(number(offer, "inet_down"), 3),
        "upload_mbps": round(number(offer, "inet_up"), 3),
        "reliability": number(offer, "reliability"),
        "verified": bool(offer.get("verified", False)),
        "cuda_max_version": number(offer, "cuda_vers"),
        "compute_capability_x100": int(number(offer, "compute_cap")),
        "direct_ssh_ports": int(number(offer, "direct_port_count")),
        "maximum_rental_duration_days": number(offer, "duration", default=math.inf),
        "hourly_cost_usd": number(offer, "dph_total", "dph"),
        "host_id": offer.get("host_id") or offer.get("machine_id"),
    }


def validate(offer: dict[str, Any], profile: dict[str, Any], max_hourly: float, max_hours: float) -> list[str]:
    failures: list[str] = []
    checks = [
        (offer["gpu_count"] >= int(profile["gpu_count"]), "GPU count"),
        (offer["vram_per_gpu_gb"] >= float(profile["min_vram_per_gpu_gb"]), "VRAM per GPU"),
        (offer["total_gpu_vram_gb"] >= float(profile["min_total_vram_gb"]), "total GPU VRAM"),
        (offer["system_ram_gb"] >= float(profile["min_system_ram_gb"]), "system RAM"),
        (offer["cpu_cores"] >= float(profile["min_cpu_cores"]), "CPU cores"),
        (offer["disk_available_gb"] >= float(profile["disk_gb"]), "disk"),
        (offer["download_mbps"] >= float(profile["min_download_mbps"]), "download bandwidth"),
        (offer["upload_mbps"] >= float(profile["min_upload_mbps"]), "upload bandwidth"),
        (offer["reliability"] >= float(profile["min_reliability"]), "reliability"),
        (offer["cuda_max_version"] >= float(profile["min_cuda_version"]), "CUDA capability"),
        (offer["hourly_cost_usd"] <= max_hourly, "hourly price"),
        (offer["maximum_rental_duration_days"] * 24 >= max_hours, "available rental duration"),
    ]
    if profile.get("require_verified", True):
        checks.append((offer["verified"], "verified host"))
    if profile.get("require_direct_ssh", True):
        checks.append((offer["direct_ssh_ports"] >= 1, "direct SSH"))
    allowed = [x.strip().lower().replace("_", " ") for x in str(profile.get("gpu_models", "")).split(",") if x.strip()]
    if allowed:
        checks.append((offer["gpu_model"].lower().replace("_", " ") in allowed, "GPU model allowlist"))
    failures.extend(name for passed, name in checks if not passed)
    return failures


def load_one(path: Path, offer_id: int | None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    offers = offers_from(value)
    if offer_id is not None:
        offers = [item for item in offers if int(number(item, "id", "ask_contract_id")) == offer_id]
    if len(offers) != 1:
        raise SystemExit(f"Expected exactly one offer; found {len(offers)}")
    return normalized(offers[0])


def display(args: argparse.Namespace) -> None:
    offers = offers_from(json.loads(Path(args.input).read_text(encoding="utf-8")))
    if not offers:
        raise SystemExit("No matching offers")
    headings = ("offer", "GPU", "n", "VRAM/GPU", "RAM", "CPU", "disk", "down/up", "reliability", "CUDA", "$/hr")
    print("\t".join(headings))
    for raw in offers:
        item = normalized(raw)
        print("\t".join(map(str, (
            item["offer_id"], item["gpu_model"], item["gpu_count"], item["vram_per_gpu_gb"],
            item["system_ram_gb"], item["cpu_cores"], item["disk_available_gb"],
            f'{item["download_mbps"]}/{item["upload_mbps"]}', f'{item["reliability"]:.4f}',
            item["cuda_max_version"], f'{item["hourly_cost_usd"]:.4f}',
        ))))


def plan(args: argparse.Namespace) -> None:
    profile_path = Path(args.profile).resolve()
    profile = scalar_profile(profile_path)
    offer = load_one(Path(args.input), args.offer_id)
    failures = validate(offer, profile, args.max_hourly, args.max_hours)
    if failures:
        raise SystemExit("Offer rejected; failed constraints: " + ", ".join(failures))
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **offer,
        "profile": profile.get("profile_name", profile_path.stem),
        "profile_path": str(profile_path),
        "profile_sha256": profile_hash,
        "disk_allocation_gb": profile["disk_gb"],
        "docker_image": profile["docker_image"],
        "accelerate_config": profile["accelerate_config"],
        "training_mode": profile["training_mode"],
        "intended_model_size": profile["model_size"],
        "approved_maximum_hourly_usd": args.max_hourly,
        "approved_maximum_runtime_hours": args.max_hours,
        "maximum_theoretical_cost_usd": round(args.max_hourly * args.max_hours, 4),
        "offer_cost_for_maximum_runtime_usd": round(offer["hourly_cost_usd"] * args.max_hours, 4),
        "remote_paths": {
            "code": "/workspace/ml-lab",
            "data": "/workspace/data",
            "models": "/workspace/models",
            "runs": "/workspace/runs",
            "cache": "/workspace/cache",
        },
        "validated": True,
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise SystemExit(f"Refusing to overwrite existing plan: {destination}")
        destination.write_text(encoded, encoding="utf-8")
        destination.chmod(0o600)
        print(destination)
    else:
        print(encoded, end="")


def verify_plan(args: argparse.Namespace) -> None:
    plan_value = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    required = {"offer_id", "docker_image", "disk_allocation_gb", "approved_maximum_hourly_usd", "approved_maximum_runtime_hours", "maximum_theoretical_cost_usd", "validated"}
    missing = required.difference(plan_value)
    if missing or plan_value.get("validated") is not True:
        raise SystemExit("Invalid plan; missing: " + ", ".join(sorted(missing)))
    print(json.dumps(plan_value, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    show = sub.add_parser("display")
    show.add_argument("--input", required=True)
    show.set_defaults(func=display)
    planner = sub.add_parser("plan")
    planner.add_argument("--input", required=True)
    planner.add_argument("--profile", required=True)
    planner.add_argument("--offer-id", type=int)
    planner.add_argument("--max-hourly", type=float, required=True)
    planner.add_argument("--max-hours", type=float, required=True)
    planner.add_argument("--output")
    planner.set_defaults(func=plan)
    check = sub.add_parser("verify-plan")
    check.add_argument("--plan", required=True)
    check.set_defaults(func=verify_plan)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
