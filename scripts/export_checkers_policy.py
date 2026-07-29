#!/usr/bin/env python3
"""Export and parity-check a small inference policy from a full training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkers.checkpoint import load_checkpoint
from checkers.config import load_run_config
from checkers.env.checkers_env import CheckersEnv
from checkers.env.encoding import encode_observation
from checkers.env.masking import legal_action_map
from checkers.rl.networks import CheckersNetwork
from checkers.web.policy_bundle import (
    PolicyBundleMetadata,
    config_sha256,
    load_policy_bundle,
    save_policy_bundle,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validation_states(
    *, max_plies: int, repetition_draws: bool
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    environment = CheckersEnv(max_plies=max_plies, repetition_draws=repetition_draws)
    environment.reset(seed=0)
    observations: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(12):
        action_map = legal_action_map(environment.state)
        mask = torch.zeros((1, 128), dtype=torch.bool)
        mask[0, tuple(action_map)] = True
        observations.append(
            (torch.as_tensor(encode_observation(environment.state)).unsqueeze(0), mask)
        )
        if environment.terminated:
            break
        actions = tuple(action_map)
        environment.step(actions[(index * 5 + 1) % len(actions)])
    return tuple(observations)


def _verify_parity(
    source: CheckersNetwork,
    exported: CheckersNetwork,
    *,
    max_plies: int,
    repetition_draws: bool,
) -> int:
    source.eval()
    exported.eval()
    verified = 0
    for observation, mask in _validation_states(
        max_plies=max_plies, repetition_draws=repetition_draws
    ):
        with torch.inference_mode():
            source_output = source(observation)
            exported_output = exported(observation)
        if not torch.equal(source_output.logits, exported_output.logits):
            raise RuntimeError("exported policy logits differ from source checkpoint")
        if not torch.equal(source_output.value, exported_output.value):
            raise RuntimeError("exported policy values differ from source checkpoint")
        source_action = source_output.logits.masked_fill(~mask, -torch.inf).argmax(dim=-1)
        exported_action = exported_output.logits.masked_fill(~mask, -torch.inf).argmax(dim=-1)
        if not torch.equal(source_action, exported_action):
            raise RuntimeError("exported greedy action differs from source checkpoint")
        verified += 1
    return verified


def main() -> int:
    """Load the full trusted checkpoint once, export its model, and prove parity."""

    args = _arguments()
    config = load_run_config(args.config.read_text(encoding="utf-8"))
    network = CheckersNetwork().to(torch.device(config.device))
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    loaded = load_checkpoint(
        path=args.checkpoint,
        expected_config=config,
        network=network,
        optimizer=optimizer,
    )
    network = network.cpu()
    metadata = PolicyBundleMetadata(
        bundle_id=f"{config.experiment_id}-update-{loaded.state.update_idx:06d}",
        experiment_id=config.experiment_id,
        update_idx=loaded.state.update_idx,
        global_step=loaded.state.global_step,
        source_checkpoint=args.checkpoint.as_posix(),
        source_checkpoint_sha256=loaded.evidence.sha256,
        source_checkpoint_size_bytes=loaded.evidence.size_bytes,
        source_git_sha=loaded.git_sha,
        source_git_dirty=loaded.git_dirty,
        config_sha256=config_sha256(config),
        max_plies=config.max_plies,
        repetition_draws=config.repetition_draws,
    )
    bundle_sha256, bundle_size = save_policy_bundle(
        path=args.output, network=network, metadata=metadata
    )
    reloaded = load_policy_bundle(args.output)
    verified_states = _verify_parity(
        network,
        reloaded.network,
        max_plies=config.max_plies,
        repetition_draws=config.repetition_draws,
    )
    print(f"bundle={args.output}")
    print(f"bundle_sha256={bundle_sha256}")
    print(f"bundle_size_bytes={bundle_size}")
    print(f"source_checkpoint_sha256={loaded.evidence.sha256}")
    print(f"update={loaded.state.update_idx}")
    print(f"parity_states={verified_states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
