"""Architecture, invariance, capacity, and gradient tests for the PPO network."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from checkers.env.encoding import encode_observation
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import (
    CheckersNetwork,
    NetworkOutput,
    PolicyHead,
    ResidualBlock,
    ValueHead,
)
from checkers.rules.state import PlayerId, State

BATCH_SIZE = 4
ACTION_COUNT = 128
RESIDUAL_BLOCKS = 6
CHANNELS = 64
OBSERVATION_PLANES = 8
GROUP_COUNT = 8
POLICY_CHANNELS = 2
VALUE_HIDDEN = 64
POLICY_TARGET_ACCURACY = 0.99
VALUE_TARGET_MSE = 1e-3


def _mask(*acf_squares: int) -> int:
    return sum(1 << (square - 1) for square in acf_squares)


def _network() -> CheckersNetwork:
    torch.manual_seed(20260728)
    return CheckersNetwork()


def test_exact_groupnorm_residual_architecture() -> None:
    network = _network()

    assert isinstance(network.stem, nn.Sequential)
    assert isinstance(network.stem[0], nn.Conv2d)
    assert network.stem[0].in_channels == OBSERVATION_PLANES
    assert network.stem[0].out_channels == CHANNELS
    assert network.stem[0].kernel_size == (3, 3)
    assert network.stem[0].padding == (1, 1)
    assert isinstance(network.stem[1], nn.GroupNorm)
    assert network.stem[1].num_groups == GROUP_COUNT
    assert isinstance(network.stem[2], nn.ReLU)

    assert isinstance(network.residual_blocks, nn.ModuleList)
    assert len(network.residual_blocks) == RESIDUAL_BLOCKS
    for block in network.residual_blocks:
        assert isinstance(block, ResidualBlock)
        assert block.conv1.kernel_size == block.conv2.kernel_size == (3, 3)
        assert block.conv1.padding == block.conv2.padding == (1, 1)
        assert block.norm1.num_groups == block.norm2.num_groups == GROUP_COUNT

    assert isinstance(network.policy_head, PolicyHead)
    assert network.policy_head.conv.in_channels == CHANNELS
    assert network.policy_head.conv.out_channels == POLICY_CHANNELS
    assert network.policy_head.conv.kernel_size == (1, 1)
    assert network.policy_head.norm.num_groups == 1
    assert network.policy_head.output.in_features == ACTION_COUNT
    assert network.policy_head.output.out_features == ACTION_COUNT

    assert isinstance(network.value_head, ValueHead)
    assert network.value_head.conv.in_channels == CHANNELS
    assert network.value_head.conv.out_channels == 1
    assert network.value_head.norm.num_groups == 1
    assert network.value_head.hidden.in_features == VALUE_HIDDEN
    assert network.value_head.hidden.out_features == VALUE_HIDDEN
    assert network.value_head.output.in_features == VALUE_HIDDEN
    assert network.value_head.output.out_features == 1
    assert not any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) for module in network.modules()
    )


def test_forward_shapes_value_range_and_batch_composition_invariance() -> None:
    network = _network()
    sample = torch.rand(8, 8, 8)
    companions = torch.rand(BATCH_SIZE - 1, 8, 8, 8)

    network.train()
    alone = network(sample.unsqueeze(0))
    in_batch = network(torch.cat((sample.unsqueeze(0), companions), dim=0))
    network.eval()
    in_eval = network(sample.unsqueeze(0))

    assert isinstance(alone, NetworkOutput)
    assert in_batch.logits.shape == (BATCH_SIZE, ACTION_COUNT)
    assert in_batch.value.shape == (BATCH_SIZE,)
    assert torch.all(in_batch.value >= -1.0)
    assert torch.all(in_batch.value <= 1.0)
    torch.testing.assert_close(alone.logits[0], in_batch.logits[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(alone.value[0], in_batch.value[0], atol=1e-5, rtol=1e-4)
    assert torch.equal(alone.logits, in_eval.logits)
    assert torch.equal(alone.value, in_eval.value)


def test_n7_pending_and_moving_planes_change_policy_logits() -> None:
    mid_sequence = State(
        men=(_mask(18), _mask(14, 22)),
        kings=(0, 0),
        side_to_move=PlayerId.RED,
        capture_in_progress=True,
        moving_square=17,
        sequence_origin=8,
        captured_pending=_mask(14),
        no_progress=(17, 39),
        ply=5,
    )
    same_placement_boundary = State(
        men=mid_sequence.men,
        kings=mid_sequence.kings,
        side_to_move=mid_sequence.side_to_move,
        no_progress=mid_sequence.no_progress,
        ply=mid_sequence.ply,
    )
    observations = torch.from_numpy(
        np.stack(
            [encode_observation(mid_sequence), encode_observation(same_placement_boundary)],
        )
    )

    logits = _network()(observations).logits

    assert not torch.equal(logits[0], logits[1])


def test_orthogonal_gains_and_zero_biases() -> None:
    network = _network()
    for module in network.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0
        if isinstance(module, nn.GroupNorm):
            assert module.weight is not None
            assert module.bias is not None
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))

    stem_conv = cast(nn.Conv2d, network.stem[0])
    stem_rows = stem_conv.weight.flatten(start_dim=1)
    policy_rows = network.policy_head.output.weight
    value_output_row = network.value_head.output.weight
    torch.testing.assert_close(
        torch.linalg.vector_norm(stem_rows, dim=1),
        torch.full((CHANNELS,), 2**0.5),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(policy_rows, dim=1),
        torch.full((ACTION_COUNT,), 0.01),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(value_output_row, dim=1),
        torch.ones(1),
        atol=1e-6,
        rtol=1e-5,
    )


def test_t5_every_parameter_connects_and_each_module_has_nonzero_aggregate_gradient() -> None:
    network = _network()
    observations = torch.rand(BATCH_SIZE, 8, 8, 8)
    legal_mask = torch.zeros(BATCH_SIZE, ACTION_COUNT, dtype=torch.bool)
    legal_mask[:, :4] = True
    actions = torch.tensor([0, 1, 2, 3])
    output = network(observations)
    output.logits.retain_grad()
    distribution = MaskedCategorical(logits=output.logits, legal_mask=legal_mask)
    loss = -distribution.log_prob(actions).mean() + (output.value - 0.5).square().mean()

    loss.backward()

    assert output.logits.grad is not None
    assert torch.equal(
        output.logits.grad[~legal_mask],
        torch.zeros_like(output.logits.grad[~legal_mask]),
    )
    assert all(parameter.grad is not None for parameter in network.parameters())
    modules: list[nn.Module] = [
        network.stem,
        *network.residual_blocks,
        network.policy_head,
        network.value_head,
    ]
    for module in modules:
        aggregate = sum(
            float(parameter.grad.detach().square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert aggregate > 0.0


def _fixed_features(network: CheckersNetwork) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7719)
    observations = torch.rand((64, 8, 8, 8), generator=generator)
    with torch.no_grad():
        return network.encode(observations).detach()


def test_t1_supervised_policy_head_memorizes_64_fixed_states() -> None:
    network = _network()
    features = _fixed_features(network)
    targets = torch.arange(64)
    optimizer = torch.optim.Adam(network.policy_head.parameters(), lr=0.03, eps=1e-5)

    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        logits = network.policy_head(features)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    accuracy = (network.policy_head(features).argmax(dim=-1) == targets).float().mean()
    assert float(accuracy) > POLICY_TARGET_ACCURACY


def test_t2_supervised_value_head_memorizes_64_fixed_states() -> None:
    network = _network()
    features = _fixed_features(network)
    targets = torch.linspace(-0.9, 0.9, 64)
    optimizer = torch.optim.Adam(network.value_head.parameters(), lr=0.01, eps=1e-5)

    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        values = network.value_head(features)
        loss = nn.functional.mse_loss(values, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    final_mse = nn.functional.mse_loss(network.value_head(features), targets)
    assert float(final_mse.detach()) < VALUE_TARGET_MSE


@pytest.mark.parametrize(
    ("observation", "error", "message"),
    [
        (torch.zeros(8, 8, 8), ValueError, "shape"),
        (torch.zeros(2, 7, 8, 8), ValueError, "shape"),
        (torch.zeros(0, 8, 8, 8), ValueError, "non-empty batch"),
        (torch.zeros(2, 8, 8, 8, dtype=torch.int64), TypeError, "floating"),
        (
            torch.full((2, 8, 8, 8), float("nan")),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_observation_tensors_raise(
    observation: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _network()(observation)


def test_non_tensor_observation_raises() -> None:
    with pytest.raises(TypeError, match="observation must be a Tensor"):
        _network()(cast(torch.Tensor, [[0.0]]))
