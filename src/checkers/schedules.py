"""Pure learning-rate and entropy schedules for immutable run configuration."""

from __future__ import annotations

from checkers.config import RunConfig
from checkers.trainer_state import TrainerState


def _inputs(config: object, state: object) -> tuple[RunConfig, TrainerState]:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if not isinstance(state, TrainerState):
        raise TypeError("state must be a TrainerState")
    return config, state


def schedule_progress(config: RunConfig, state: TrainerState) -> float:
    """Return clamped training progress in `[0, 1]`.

    Args:
        config: Immutable run configuration.
        state: Mutable trainer counters.

    Returns:
        Global-step progress, clamped after the configured budget.

    Raises:
        TypeError: If either argument has the wrong runtime type.
    """

    checked_config, checked_state = _inputs(config, state)
    return min(1.0, checked_state.global_step / checked_config.total_timesteps)


def current_lr(config: RunConfig, state: TrainerState) -> float:
    """Return linearly annealed learning rate without mutating config.

    Args:
        config: Immutable run configuration.
        state: Mutable trainer counters.

    Returns:
        Learning rate at the current global step.

    Raises:
        TypeError: If either argument has the wrong runtime type.
    """

    checked_config, checked_state = _inputs(config, state)
    return checked_config.learning_rate * (1.0 - schedule_progress(checked_config, checked_state))


def current_ent_coef(config: RunConfig, state: TrainerState) -> float:
    """Return entropy coefficient annealed over the declared initial fraction.

    Args:
        config: Immutable run configuration.
        state: Mutable trainer counters.

    Returns:
        Current entropy coefficient.

    Raises:
        TypeError: If either argument has the wrong runtime type.
    """

    checked_config, checked_state = _inputs(config, state)
    progress = min(
        1.0,
        schedule_progress(checked_config, checked_state) / checked_config.ent_anneal_fraction,
    )
    if progress == 1.0:
        return checked_config.ent_coef_end
    span = checked_config.ent_coef_start - checked_config.ent_coef_end
    return checked_config.ent_coef_start - progress * span
