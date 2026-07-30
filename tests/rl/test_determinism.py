"""D1–D3 same-stack determinism tests for the offline PPO core."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from checkers.rl.determinism import SeedStreams, derive_stream_seed, seed_everything
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import CheckersNetwork
from checkers.rl.ppo import PPOConfig, PPOMinibatch, ppo_minibatch_update

NUM_ENVS = 4
UPDATES = 10
ROOT_SEED = 314159
REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires the declared CUDA validation host",
)


@dataclass(frozen=True, slots=True)
class _Trace:
    actions: tuple[tuple[int, ...], ...]
    metrics: tuple[tuple[float, ...], ...]


def test_d1_seed_everything_reproduces_all_local_streams() -> None:
    first_streams = seed_everything(ROOT_SEED, num_envs=NUM_ENVS, deterministic=True)
    first_values = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    second_streams = seed_everything(ROOT_SEED, num_envs=NUM_ENVS, deterministic=True)
    second_values = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    assert isinstance(first_streams, SeedStreams)
    assert first_streams == second_streams
    assert first_values == second_values
    assert len(set(first_streams.env_seeds)) == NUM_ENVS
    assert first_streams.env_seeds == tuple(
        derive_stream_seed(ROOT_SEED, index + 4) for index in range(NUM_ENVS)
    )
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"seed": True}, TypeError, "seed must be an integer"),
        ({"seed": -1}, ValueError, "unsigned 64-bit"),
        ({"num_envs": 0}, ValueError, "num_envs must be positive"),
        ({"num_envs": True}, TypeError, "num_envs must be an integer"),
        ({"deterministic": 1}, TypeError, "deterministic must be bool"),
    ],
)
def test_invalid_seed_configuration_raises(
    arguments: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    complete: dict[str, object] = {
        "seed": ROOT_SEED,
        "num_envs": NUM_ENVS,
        "deterministic": True,
    }
    complete.update(arguments)
    with pytest.raises(error, match=message):
        seed_everything(**complete)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("root", "stream", "error", "message"),
    [
        (ROOT_SEED, -1, ValueError, "stream_index must be non-negative"),
        (ROOT_SEED, True, TypeError, "stream_index must be an integer"),
        (1 << 64, 0, ValueError, "unsigned 64-bit"),
    ],
)
def test_invalid_stream_derivation_raises(
    root: int,
    stream: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        derive_stream_seed(root, stream)


def test_nondeterministic_cpu_only_configuration_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    streams = seed_everything(ROOT_SEED, num_envs=1, deterministic=False)

    assert not streams.deterministic
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic


def _run_ten_updates(device: torch.device) -> _Trace:
    streams = seed_everything(ROOT_SEED, num_envs=NUM_ENVS, deterministic=True)
    network = CheckersNetwork().to(device)
    generator = torch.Generator().manual_seed(streams.env_seeds[0])
    observations = torch.rand((NUM_ENVS, 8, 8, 8), generator=generator).to(device)
    masks = torch.zeros((NUM_ENVS, 128), dtype=torch.bool, device=device)
    for lane in range(NUM_ENVS):
        masks[lane, lane * 4 : lane * 4 + 4] = True
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)
    config = PPOConfig(target_kl=1.0)
    actions_trace: list[tuple[int, ...]] = []
    metrics_trace: list[tuple[float, ...]] = []

    for _ in range(UPDATES):
        with torch.no_grad():
            output = network(observations)
            distribution = MaskedCategorical(logits=output.logits, legal_mask=masks)
            actions = distribution.sample()
            old_logprob = distribution.log_prob(actions)
            old_values = output.value
            returns = torch.clamp(old_values + 0.2 * advantages, min=-1.0, max=1.0)
        minibatch = PPOMinibatch(
            obs=observations,
            legal_mask=masks,
            actions=actions,
            old_logprob=old_logprob,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            source_indices=torch.arange(NUM_ENVS, device=device),
        )
        metrics = ppo_minibatch_update(
            network=network,
            optimizer=optimizer,
            minibatch=minibatch,
            config=config,
        )
        actions_trace.append(tuple(int(action) for action in actions.cpu().tolist()))
        metrics_trace.append(
            (
                metrics.policy_loss,
                metrics.value_loss,
                metrics.entropy,
                metrics.total_loss,
                metrics.approx_kl,
                metrics.clipfrac,
                metrics.grad_norm,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return _Trace(actions=tuple(actions_trace), metrics=tuple(metrics_trace))


def test_d2_cpu_losses_and_actions_are_bitwise_reproducible_for_ten_updates() -> None:
    first = _run_ten_updates(torch.device("cpu"))
    second = _run_ten_updates(torch.device("cpu"))

    assert first.actions == second.actions
    assert first.metrics == second.metrics
    assert len(first.metrics) == UPDATES


@REQUIRES_CUDA
def test_d3_gpu_actions_match_and_losses_meet_same_stack_tolerance() -> None:
    device = torch.device("cuda:0")
    first = _run_ten_updates(device)
    second = _run_ten_updates(device)

    assert first.actions == second.actions
    first_metrics = torch.tensor(first.metrics, dtype=torch.float64)
    second_metrics = torch.tensor(second.metrics, dtype=torch.float64)
    torch.testing.assert_close(first_metrics, second_metrics, atol=1e-5, rtol=1e-4)
    assert len(first.metrics) == UPDATES


@REQUIRES_CUDA
def test_d3_cuda_bfloat16_masked_distribution_is_finite_and_legal() -> None:
    device = torch.device("cuda:0")
    logits = torch.tensor(
        [[-3.0, 0.0, 2.0, 7.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
        device=device,
    )
    distribution = MaskedCategorical(logits=logits, legal_mask=mask)
    actions = distribution.sample((1024,))
    selected = mask.expand(1024, -1, -1).gather(-1, actions.unsqueeze(-1))
    loss = -(distribution.log_prob(torch.tensor([2, 1], device=device))).mean()
    loss.backward()  # type: ignore[no-untyped-call]

    assert selected.all()
    assert torch.isfinite(distribution.entropy()).all()
    assert logits.grad is not None
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
