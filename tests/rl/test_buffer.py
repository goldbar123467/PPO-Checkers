"""T8 and defensive-contract tests for the chronological rollout buffer."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from checkers.rl.buffer import RolloutBuffer, RolloutStep

EXPECTED_TRANSITIONS = 6


def _step(  # noqa: PLR0913
    *,
    values: tuple[float, float] = (0.0, 0.0),
    rewards: tuple[float, float] = (0.0, 0.0),
    dones: tuple[bool, bool] = (False, False),
    sigmas: tuple[int, int] = (-1, -1),
    trainable: tuple[bool, bool] = (True, False),
    time_marker: float = 0.0,
) -> RolloutStep:
    observations = torch.tensor(
        [[time_marker, 0.0], [time_marker, 1.0]],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [[True, False, False, False], [False, True, False, False]],
    )
    return RolloutStep(
        obs=observations,
        legal_mask=masks,
        action=torch.tensor([0, 1]),
        behaviour_logprob=torch.tensor([0.0, 0.0]),
        value=torch.tensor(values),
        reward=torch.tensor(rewards),
        done=torch.tensor(dones),
        actor=torch.tensor([0, 1]),
        sigma=torch.tensor(sigmas, dtype=torch.int8),
        trainable=torch.tensor(trainable),
        policy_id=tuple("current" if flag else "frozen-001" for flag in trainable),
        env_id=torch.tensor([0, 1]),
        move_completed=torch.tensor([True, True]),
    )


def test_t8_full_chronology_drives_gae_before_trainable_filtering() -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=3,
        observation_shape=(2,),
        action_count=4,
    )
    buffer.append(_step(trainable=(True, False), time_marker=0.0))
    buffer.append(
        _step(
            sigmas=(-1, 1),
            trainable=(False, True),
            time_marker=1.0,
        )
    )
    buffer.append(
        _step(
            rewards=(1.0, -1.0),
            dones=(True, True),
            sigmas=(1, 1),
            trainable=(True, False),
            time_marker=2.0,
        )
    )

    batch = buffer.finalize(
        bootstrap_value=torch.tensor([99.0, -99.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert batch.transitions == EXPECTED_TRANSITIONS
    assert torch.equal(batch.env_id, torch.tensor([0, 1, 0, 1, 0, 1]))
    assert torch.equal(batch.advantages, torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0]))
    assert torch.equal(batch.returns, batch.advantages)

    policy = batch.policy_view()
    assert torch.equal(policy.source_indices, torch.tensor([0, 3, 4]))
    assert torch.equal(policy.advantages, torch.tensor([1.0, -1.0, 1.0]))
    assert policy.policy_id == ("current", "current", "current")
    assert policy.legal_mask[:, :2].tolist() == [[True, False], [False, True], [True, False]]

    default_values = batch.value_view()
    all_values = batch.value_view(include_nontrainable=True)
    assert torch.equal(default_values.source_indices, policy.source_indices)
    assert torch.equal(all_values.source_indices, torch.arange(6))
    with pytest.raises(TypeError, match="include_nontrainable must be bool"):
        batch.value_view(include_nontrainable=1)  # type: ignore[arg-type]


def test_t8_mid_capture_rollout_boundary_uses_positive_sigma_bootstrap() -> None:
    buffer = RolloutBuffer(
        num_envs=1,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    step = RolloutStep(
        obs=torch.tensor([[0.0, 1.0]]),
        legal_mask=torch.tensor([[False, False, True, False]]),
        action=torch.tensor([2]),
        behaviour_logprob=torch.tensor([0.0]),
        value=torch.tensor([0.25]),
        reward=torch.tensor([0.0]),
        done=torch.tensor([False]),
        actor=torch.tensor([0]),
        sigma=torch.tensor([1], dtype=torch.int8),
        trainable=torch.tensor([True]),
        policy_id=("current",),
        env_id=torch.tensor([0]),
        move_completed=torch.tensor([False]),
    )
    buffer.append(step)

    batch = buffer.finalize(
        bootstrap_value=torch.tensor([0.75]),
        gamma=1.0,
        gae_lambda=0.95,
    )

    assert batch.move_completed.tolist() == [False]
    torch.testing.assert_close(batch.advantages, torch.tensor([0.5]))
    torch.testing.assert_close(batch.returns, torch.tensor([0.75]))


def test_append_clones_and_detaches_the_stored_mask_and_observation() -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    step = _step()
    step.obs.requires_grad_(True)
    buffer.append(step)
    step.obs.data.fill_(77.0)
    step.legal_mask.fill_(False)

    batch = buffer.finalize(
        bootstrap_value=torch.zeros(2),
        gamma=1.0,
        gae_lambda=0.95,
    )

    assert batch.obs.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert batch.legal_mask.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
    ]
    assert not batch.obs.requires_grad


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"num_envs": 0}, ValueError, "num_envs must be positive"),
        ({"num_envs": True}, TypeError, "num_envs must be an integer"),
        ({"num_steps": 0}, ValueError, "num_steps must be positive"),
        ({"observation_shape": ()}, ValueError, "observation_shape"),
        ({"observation_shape": [2]}, TypeError, "observation_shape"),
        ({"observation_shape": (2, 0)}, ValueError, "observation_shape"),
        ({"observation_shape": (2, True)}, TypeError, "observation_shape"),
        ({"action_count": 0}, ValueError, "action_count must be positive"),
    ],
)
def test_invalid_buffer_configuration_raises(
    arguments: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    complete: dict[str, object] = {
        "num_envs": 2,
        "num_steps": 3,
        "observation_shape": (2,),
        "action_count": 4,
    }
    complete.update(arguments)
    with pytest.raises(error, match=message):
        RolloutBuffer(**complete)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("obs", torch.zeros(2, 3), ValueError, "obs shape"),
        ("obs", [0.0, 1.0], TypeError, "obs must be a Tensor"),
        ("obs", torch.zeros(2, 2, dtype=torch.int64), TypeError, "obs must be floating"),
        ("obs", torch.tensor([[0.0, float("nan")], [0.0, 1.0]]), ValueError, "obs must be finite"),
        ("legal_mask", torch.ones(2, 4), TypeError, "legal_mask must have dtype bool"),
        ("legal_mask", torch.ones(2, 3, dtype=torch.bool), ValueError, "legal_mask shape"),
        ("action", torch.tensor([0.0, 1.0]), TypeError, "action must have dtype int64"),
        ("action", torch.tensor([0, 4]), ValueError, "action must be in"),
        ("action", torch.tensor([0, 0]), ValueError, "action must be legal"),
        ("behaviour_logprob", torch.tensor([0.0]), ValueError, "behaviour_logprob shape"),
        (
            "behaviour_logprob",
            torch.tensor([0.0, float("inf")]),
            ValueError,
            "behaviour_logprob must be finite",
        ),
        (
            "behaviour_logprob",
            torch.tensor([0.0, 0.0], dtype=torch.float64),
            ValueError,
            "share a dtype",
        ),
        ("value", torch.tensor([0, 0]), TypeError, "value must be floating"),
        ("value", torch.tensor([0.0, float("nan")]), ValueError, "value must be finite"),
        ("reward", torch.tensor([0.0]), ValueError, "reward shape"),
        ("reward", torch.tensor([0, 0]), TypeError, "reward must be floating"),
        ("reward", torch.tensor([0.0, float("nan")]), ValueError, "reward must be finite"),
        ("reward", torch.tensor([0.0, 0.5]), ValueError, "reward must contain only"),
        ("done", torch.tensor([0, 0]), TypeError, "done must have dtype bool"),
        ("actor", torch.tensor([0, 1], dtype=torch.int8), TypeError, "actor must have dtype int64"),
        ("actor", torch.tensor([0, 2]), ValueError, "actor must contain only"),
        ("sigma", torch.tensor([1, 1]), TypeError, "sigma must have dtype int8"),
        ("sigma", torch.tensor([1, 0], dtype=torch.int8), ValueError, "sigma must contain only"),
        ("trainable", torch.tensor([1, 0]), TypeError, "trainable must have dtype bool"),
        ("policy_id", ("current",), ValueError, "policy_id length"),
        ("policy_id", ("current", ""), ValueError, "policy_id entries"),
        ("policy_id", ("current", 3), TypeError, "policy_id entries"),
        (
            "env_id",
            torch.tensor([0, 1], dtype=torch.int8),
            TypeError,
            "env_id must have dtype int64",
        ),
        ("env_id", torch.tensor([1, 0]), ValueError, "env_id must equal"),
        ("move_completed", torch.tensor([1, 1]), TypeError, "move_completed must have dtype bool"),
    ],
)
def test_invalid_rollout_step_raises(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    step = replace(_step(), **{field: value})  # type: ignore[arg-type]
    with pytest.raises(error, match=message):
        buffer.append(step)


def test_non_step_and_wrong_device_raise() -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    with pytest.raises(TypeError, match="step must be a RolloutStep"):
        buffer.append(object())  # type: ignore[arg-type]
    wrong_device = replace(_step(), actor=torch.tensor([0, 1], device="meta"))
    with pytest.raises(ValueError, match="buffer device"):
        buffer.append(wrong_device)


def test_buffer_capacity_and_single_finalize_are_enforced() -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    with pytest.raises(ValueError, match="not full"):
        buffer.finalize(
            bootstrap_value=torch.zeros(2),
            gamma=1.0,
            gae_lambda=0.95,
        )
    buffer.append(_step())
    assert buffer.size == 1
    assert buffer.full
    with pytest.raises(OverflowError, match="already full"):
        buffer.append(_step())
    buffer.finalize(
        bootstrap_value=torch.zeros(2),
        gamma=1.0,
        gae_lambda=0.95,
    )
    with pytest.raises(RuntimeError, match="already finalized"):
        buffer.finalize(
            bootstrap_value=torch.zeros(2),
            gamma=1.0,
            gae_lambda=0.95,
        )


@pytest.mark.parametrize(
    ("bootstrap", "error", "message"),
    [
        (object(), TypeError, "bootstrap_value must be a Tensor"),
        (torch.zeros(1), ValueError, "bootstrap_value shape"),
        (torch.zeros(2, dtype=torch.int64), TypeError, "bootstrap_value must be floating"),
        (torch.zeros(2, dtype=torch.float64), ValueError, "dtype must match"),
        (torch.tensor([0.0, float("nan")]), ValueError, "bootstrap_value must be finite"),
        (torch.zeros(2, device="meta"), ValueError, "buffer device"),
    ],
)
def test_invalid_finalize_bootstrap_raises(
    bootstrap: object,
    error: type[Exception],
    message: str,
) -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    buffer.append(_step())
    with pytest.raises(error, match=message):
        buffer.finalize(
            bootstrap_value=bootstrap,  # type: ignore[arg-type]
            gamma=1.0,
            gae_lambda=0.95,
        )


def test_policy_view_rejects_a_rollout_with_no_trainable_transitions() -> None:
    buffer = RolloutBuffer(
        num_envs=2,
        num_steps=1,
        observation_shape=(2,),
        action_count=4,
    )
    buffer.append(_step(trainable=(False, False)))
    batch = buffer.finalize(
        bootstrap_value=torch.zeros(2),
        gamma=1.0,
        gae_lambda=0.95,
    )

    with pytest.raises(ValueError, match="no trainable transitions"):
        batch.policy_view()
    with pytest.raises(ValueError, match="no trainable transitions"):
        batch.value_view()
