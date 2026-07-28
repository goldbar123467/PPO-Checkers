"""Deterministic full-chronology self-play collection for the Phase 7 A0 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch

from checkers.config import RunConfig
from checkers.env.masking import ACTION_COUNT, action_to_step, step_to_action
from checkers.env.vec_env import CheckersVectorEnv
from checkers.metrics import (
    GameStatistics,
    GameSummary,
    MaskStatistics,
    policy_health,
    value_health,
)
from checkers.rl.buffer import RolloutBatch, RolloutBuffer, RolloutStep
from checkers.rl.determinism import ENV_STREAM_OFFSET, derive_stream_seed
from checkers.rl.masked_categorical import MaskedCategorical
from checkers.rl.networks import OBSERVATION_SHAPE, CheckersNetwork
from checkers.rules.moves import legal_steps
from checkers.rules.oracle import oracle_legal_steps
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome

CURRENT_POLICY_ID = "current"
COLLECTOR_SCHEMA = "CHECKERS_SELFPLAY_COLLECTOR_1"
COLLECTOR_RECORD_FIELDS = frozenset(
    {
        "schema",
        "vector_env",
        "episode_indices",
        "active_games",
        "mask_statistics",
        "game_statistics",
        "calibration_predictions",
        "calibration_outcomes",
    }
)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """One finalized rollout plus interval completions and cumulative health metrics."""

    rollout: RolloutBatch
    metrics: dict[str, float]
    completed_games: tuple[GameSummary, ...]


@dataclass(slots=True)
class _ActiveGame:
    steps: int = 0
    moves: int = 0
    captures: int = 0
    capture_sequences: list[int] = field(default_factory=list)
    active_capture_length: int = 0
    promotions: int = 0
    value_trace: list[tuple[PlayerId, float]] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        """Return a weights-only-checkpoint-safe mutable-state record."""

        return {
            "steps": self.steps,
            "moves": self.moves,
            "captures": self.captures,
            "capture_sequences": list(self.capture_sequences),
            "active_capture_length": self.active_capture_length,
            "promotions": self.promotions,
            "value_trace": [[int(actor), prediction] for actor, prediction in self.value_trace],
        }


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _float_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} values must be numeric")
        checked = float(item)
        if not math.isfinite(checked) or not -1.0 <= checked <= 1.0:
            raise ValueError(f"{name} values must be finite and in [-1, 1]")
        values.append(checked)
    return values


def _integer_list(value: object, name: str, *, positive: bool = False) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result = [_nonnegative_integer(item, name) for item in value]
    if positive and any(item < 1 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _active_game_from_record(value: object) -> _ActiveGame:
    if not isinstance(value, dict):
        raise TypeError("active game record must be a mapping")
    record = cast(dict[str, object], value)
    expected = {
        "steps",
        "moves",
        "captures",
        "capture_sequences",
        "active_capture_length",
        "promotions",
        "value_trace",
    }
    if set(record) != expected:
        raise ValueError("active game record fields are invalid")
    raw_trace = record["value_trace"]
    if not isinstance(raw_trace, list):
        raise TypeError("value_trace must be a list")
    trace: list[tuple[PlayerId, float]] = []
    for item in raw_trace:
        if not isinstance(item, list) or len(item) != 2:  # noqa: PLR2004
            raise TypeError("value_trace entries must be actor/value pairs")
        actor_raw, prediction_raw = item
        if isinstance(actor_raw, bool) or not isinstance(actor_raw, int):
            raise TypeError("value_trace actors must be integers")
        try:
            actor = PlayerId(actor_raw)
        except ValueError as error:
            raise ValueError("value_trace actor is invalid") from error
        prediction = _float_list([prediction_raw], "value_trace")[0]
        trace.append((actor, prediction))
    active = _ActiveGame(
        steps=_nonnegative_integer(record["steps"], "steps"),
        moves=_nonnegative_integer(record["moves"], "moves"),
        captures=_nonnegative_integer(record["captures"], "captures"),
        capture_sequences=_integer_list(
            record["capture_sequences"], "capture_sequences", positive=True
        ),
        active_capture_length=_nonnegative_integer(
            record["active_capture_length"], "active_capture_length"
        ),
        promotions=_nonnegative_integer(record["promotions"], "promotions"),
        value_trace=trace,
    )
    if len(active.value_trace) != active.steps:
        raise ValueError("value_trace length must equal active game steps")
    return active


def _mask_statistics_record(statistics: MaskStatistics) -> dict[str, int]:
    return {
        "sample_legality_violations": statistics.sample_legality_violations,
        "oracle_disagreements": statistics.oracle_disagreements,
        "empty_mask_count": statistics.empty_mask_count,
        "legal_action_total": statistics.legal_action_total,
        "continuation_state_total": statistics.continuation_state_total,
        "state_total": statistics.state_total,
    }


def _mask_statistics_from_record(value: object) -> MaskStatistics:
    expected = set(_mask_statistics_record(MaskStatistics()))
    if not isinstance(value, dict):
        raise TypeError("mask statistics record must be a mapping")
    record = cast(dict[str, object], value)
    if set(record) != expected:
        raise ValueError("mask statistics record fields are invalid")
    values = {name: _nonnegative_integer(record[name], name) for name in expected}
    if values["continuation_state_total"] > values["state_total"]:
        raise ValueError("continuation states cannot exceed total states")
    return MaskStatistics(**values)


def _game_statistics_record(statistics: GameStatistics) -> dict[str, object]:
    return {
        "games_started": statistics.games_started,
        "current_policy_first_starts": statistics.current_policy_first_starts,
        "games_completed": statistics.games_completed,
        "steps": statistics.steps,
        "moves": statistics.moves,
        "draws": statistics.draws,
        "first_player_wins": statistics.first_player_wins,
        "captures": statistics.captures,
        "capture_sequences": list(statistics.capture_sequences),
        "promotions": statistics.promotions,
        "no_progress_draws": statistics.no_progress_draws,
        "ply_cap_draws": statistics.ply_cap_draws,
    }


def _game_statistics_from_record(value: object) -> GameStatistics:
    expected = set(_game_statistics_record(GameStatistics()))
    if not isinstance(value, dict):
        raise TypeError("game statistics record must be a mapping")
    record = cast(dict[str, object], value)
    if set(record) != expected:
        raise ValueError("game statistics record fields are invalid")
    sequences = _integer_list(record["capture_sequences"], "capture_sequences", positive=True)
    integer_fields = expected - {"capture_sequences"}
    values = {name: _nonnegative_integer(record[name], name) for name in integer_fields}
    if values["games_completed"] > values["games_started"]:
        raise ValueError("completed games cannot exceed started games")
    if values["current_policy_first_starts"] > values["games_started"]:
        raise ValueError("first-player starts cannot exceed game starts")
    if values["draws"] > values["games_completed"]:
        raise ValueError("draws cannot exceed completed games")
    return GameStatistics(
        games_started=values["games_started"],
        current_policy_first_starts=values["current_policy_first_starts"],
        games_completed=values["games_completed"],
        steps=values["steps"],
        moves=values["moves"],
        draws=values["draws"],
        first_player_wins=values["first_player_wins"],
        captures=values["captures"],
        capture_sequences=sequences,
        promotions=values["promotions"],
        no_progress_draws=values["no_progress_draws"],
        ply_cap_draws=values["ply_cap_draws"],
    )


class SelfPlayCollector:
    """Collect A0 current-policy self-play without dropping opponent chronology."""

    def __init__(
        self,
        *,
        config: RunConfig,
        network: CheckersNetwork,
        initial_states: tuple[State, ...] | None = None,
    ) -> None:
        """Create a collector with stable lanes, cumulative counters, and balanced roles.

        Args:
            config: Frozen validated run configuration.
            network: Shared policy/value network used for both colours in the A0 baseline.
            initial_states: Optional completed-move reset state for each focused-test lane.

        Raises:
            TypeError: If public inputs have invalid runtime types.
            NotImplementedError: If a Phase 8 historical-policy arm is requested.
            ValueError: If an initial lane is already terminal.
        """

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not isinstance(network, CheckersNetwork):
            raise TypeError("network must be a CheckersNetwork")
        if config.arm != "A0":
            raise NotImplementedError("SelfPlayCollector currently implements the Phase 7 A0 arm")
        self._config = config
        self._device = torch.device(config.device)
        self._network = network.to(self._device)
        self._vector_env = CheckersVectorEnv(
            config.num_envs,
            max_plies=config.max_plies,
            repetition_draws=config.repetition_draws,
            initial_states=initial_states,
        )
        if any(environment.terminated for environment in self._vector_env.envs):
            raise ValueError("initial self-play states must be non-terminal")

        self._episode_indices = [0 for _ in range(config.num_envs)]
        self._active_games = [_ActiveGame() for _ in range(config.num_envs)]
        self._mask_statistics = MaskStatistics()
        self._game_statistics = GameStatistics()
        self._calibration_predictions: list[float] = []
        self._calibration_outcomes: list[float] = []
        for lane in range(config.num_envs):
            self._record_game_start(lane)

    @property
    def vector_env(self) -> CheckersVectorEnv:
        """Return the live vector environment for audit and checkpoint integration."""

        return self._vector_env

    @property
    def episode_indices(self) -> tuple[int, ...]:
        """Return the stable per-lane zero-based episode ordinals."""

        return tuple(self._episode_indices)

    @property
    def active_episode_steps(self) -> tuple[int, ...]:
        """Return the number of retained transitions in each active game."""

        return tuple(game.steps for game in self._active_games)

    def to_record(self) -> dict[str, object]:
        """Return every collector-owned value needed for exact update-boundary resume."""

        return {
            "schema": COLLECTOR_SCHEMA,
            "vector_env": self._vector_env.serialize(),
            "episode_indices": list(self._episode_indices),
            "active_games": [game.to_record() for game in self._active_games],
            "mask_statistics": _mask_statistics_record(self._mask_statistics),
            "game_statistics": _game_statistics_record(self._game_statistics),
            "calibration_predictions": list(self._calibration_predictions),
            "calibration_outcomes": list(self._calibration_outcomes),
        }

    @classmethod
    def from_record(  # noqa: PLR0912, PLR0915
        cls,
        *,
        config: RunConfig,
        network: CheckersNetwork,
        record: object,
    ) -> SelfPlayCollector:
        """Validate and reconstruct a collector, including any active capture sequence."""

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not isinstance(network, CheckersNetwork):
            raise TypeError("network must be a CheckersNetwork")
        if config.arm != "A0":
            raise NotImplementedError("SelfPlayCollector currently implements the Phase 7 A0 arm")
        if not isinstance(record, dict):
            raise TypeError("collector record must be a mapping")
        values = cast(dict[str, object], record)
        if set(values) != COLLECTOR_RECORD_FIELDS:
            raise ValueError("collector record fields are invalid")
        if values["schema"] != COLLECTOR_SCHEMA:
            raise ValueError("collector record schema is unsupported")
        vector_text = values["vector_env"]
        if not isinstance(vector_text, str):
            raise TypeError("collector vector_env must be text")
        vector_env = CheckersVectorEnv.from_serialized(vector_text)
        if vector_env.num_envs != config.num_envs:
            raise ValueError("collector lane count does not match config")
        if any(
            environment.max_plies != config.max_plies
            or environment.repetition_draws != config.repetition_draws
            for environment in vector_env.envs
        ):
            raise ValueError("collector environment rules do not match config")
        episode_indices = _integer_list(values["episode_indices"], "episode_indices")
        raw_active_games = values["active_games"]
        if not isinstance(raw_active_games, list):
            raise TypeError("active_games must be a list")
        active_games = [_active_game_from_record(game) for game in raw_active_games]
        if len(episode_indices) != config.num_envs or len(active_games) != config.num_envs:
            raise ValueError("collector lane records must match num_envs")
        for lane, (environment, game) in enumerate(zip(vector_env.envs, active_games, strict=True)):
            if environment.terminated:
                raise ValueError(f"collector lane {lane} must not remain terminal")
            if environment.state.capture_in_progress != (game.active_capture_length > 0):
                raise ValueError(f"collector lane {lane} capture counters disagree with state")
        mask_statistics = _mask_statistics_from_record(values["mask_statistics"])
        game_statistics = _game_statistics_from_record(values["game_statistics"])
        if game_statistics.games_started != config.num_envs + sum(episode_indices):
            raise ValueError("collector game-start count disagrees with episode indices")
        calibration_predictions = _float_list(
            values["calibration_predictions"], "calibration_predictions"
        )
        calibration_outcomes = _float_list(values["calibration_outcomes"], "calibration_outcomes")
        if len(calibration_predictions) != len(calibration_outcomes):
            raise ValueError("collector calibration vectors must have equal length")

        instance = cls.__new__(cls)
        instance._config = config
        instance._device = torch.device(config.device)
        instance._network = network.to(instance._device)
        instance._vector_env = vector_env
        instance._episode_indices = episode_indices
        instance._active_games = active_games
        instance._mask_statistics = mask_statistics
        instance._game_statistics = game_statistics
        instance._calibration_predictions = calibration_predictions
        instance._calibration_outcomes = calibration_outcomes
        return instance

    def _record_game_start(self, lane: int) -> None:
        current_policy_as_red = (lane + self._episode_indices[lane]) % 2 == 0
        self._game_statistics.start_game(current_policy_as_red=current_policy_as_red)

    def _reset_lane(self, lane: int) -> None:
        self._episode_indices[lane] += 1
        stream_index = (
            ENV_STREAM_OFFSET + lane + self._episode_indices[lane] * self._config.num_envs
        )
        seed = derive_stream_seed(self._config.seed, stream_index)
        self._vector_env.envs[lane].reset(seed=seed)
        self._active_games[lane] = _ActiveGame()
        self._record_game_start(lane)

    @staticmethod
    def _oracle_disagreement(state: State, legal_mask: torch.Tensor) -> bool:
        production_steps = legal_steps(state)
        oracle_steps = oracle_legal_steps(state)
        if production_steps != oracle_steps:
            return True
        expected_actions = {step_to_action(state, step) for step in production_steps}
        observed_actions = set(torch.nonzero(legal_mask, as_tuple=False).flatten().tolist())
        return expected_actions != observed_actions

    def _record_terminal(
        self,
        *,
        lane: int,
        outcome: Outcome,
    ) -> GameSummary:
        game = self._active_games[lane]
        if game.active_capture_length:
            raise RuntimeError("terminal game retained an unfinished capture-sequence counter")
        summary = GameSummary(
            winner=outcome.winner,
            reason=outcome.reason,
            steps=game.steps,
            moves=game.moves,
            captures=game.captures,
            capture_sequences=tuple(game.capture_sequences),
            promotions=game.promotions,
        )
        self._game_statistics.record_game(summary)
        for actor, prediction in game.value_trace:
            self._calibration_predictions.append(prediction)
            self._calibration_outcomes.append(float(outcome.score_for(actor)))
        return summary

    def _update_active_game(  # noqa: PLR0913
        self,
        *,
        lane: int,
        actor: PlayerId,
        value: float,
        is_capture: bool,
        move_completed: bool,
        promoted: bool,
    ) -> None:
        game = self._active_games[lane]
        game.steps += 1
        game.value_trace.append((actor, value))
        if is_capture:
            game.captures += 1
            game.active_capture_length += 1
        if move_completed:
            game.moves += 1
            if game.active_capture_length:
                game.capture_sequences.append(game.active_capture_length)
                game.active_capture_length = 0
        game.promotions += int(promoted)

    def collect(self) -> CollectionResult:  # noqa: PLR0914, PLR0915
        """Collect and finalize exactly one configured time-major rollout."""

        buffer = RolloutBuffer(
            num_envs=self._config.num_envs,
            num_steps=self._config.num_steps,
            observation_shape=OBSERVATION_SHAPE,
            action_count=ACTION_COUNT,
            device=self._device,
        )
        rollout_logits: list[torch.Tensor] = []
        rollout_masks: list[torch.Tensor] = []
        completed_games: list[GameSummary] = []

        for _ in range(self._config.num_steps):
            states = tuple(environment.state for environment in self._vector_env.envs)
            observations = torch.as_tensor(
                self._vector_env.observations(),
                dtype=torch.float32,
                device=self._device,
            )
            legal_mask = torch.as_tensor(
                self._vector_env.legal_masks(),
                dtype=torch.bool,
                device=self._device,
            )
            legal_counts = legal_mask.sum(dim=-1)
            continuation_states = torch.tensor(
                [state.capture_in_progress for state in states],
                dtype=torch.bool,
                device=self._device,
            )
            oracle_disagreements = sum(
                self._oracle_disagreement(state, legal_mask[lane])
                for lane, state in enumerate(states)
            )
            empty_mask_count = int((legal_counts == 0).sum().item())
            if empty_mask_count:
                raise RuntimeError("non-terminal self-play lane produced an empty legal mask")

            with torch.no_grad():
                output = self._network(observations)
                distribution = MaskedCategorical(logits=output.logits, legal_mask=legal_mask)
                actions = distribution.sample()
                log_probabilities = distribution.log_prob(actions)
            selected_legal = legal_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
            sample_violations = int((~selected_legal).sum().item())
            self._mask_statistics.record(
                legal_counts=legal_counts,
                continuation_states=continuation_states,
                sample_legality_violations=sample_violations,
                oracle_disagreements=oracle_disagreements,
                empty_mask_count=empty_mask_count,
            )
            rollout_logits.append(output.logits.detach())
            rollout_masks.append(legal_mask.detach())

            actors = tuple(state.side_to_move for state in states)
            chosen_steps = tuple(
                action_to_step(state, int(action))
                for state, action in zip(states, actions.tolist(), strict=True)
            )
            actor_king_counts = tuple(
                state.kings[int(actor)].bit_count()
                for state, actor in zip(states, actors, strict=True)
            )
            _, rewards_array, terminated_array, truncated_array, infos = self._vector_env.step(
                np.asarray(actions.detach().cpu(), dtype=np.int64)
            )
            if bool(truncated_array.any()):
                raise RuntimeError("checkers self-play unexpectedly returned truncation")

            dones = torch.as_tensor(terminated_array, dtype=torch.bool, device=self._device)
            rewards = torch.as_tensor(rewards_array, dtype=output.value.dtype, device=self._device)
            move_completed = torch.tensor(
                [bool(info["move_completed"]) for info in infos],
                dtype=torch.bool,
                device=self._device,
            )
            next_actors = tuple(
                environment.state.side_to_move for environment in self._vector_env.envs
            )
            sigmas = torch.tensor(
                [
                    1 if after is before else -1
                    for before, after in zip(actors, next_actors, strict=True)
                ],
                dtype=torch.int8,
                device=self._device,
            )

            for lane, (actor, step, info) in enumerate(
                zip(actors, chosen_steps, infos, strict=True)
            ):
                after_state = self._vector_env.envs[lane].state
                promoted = after_state.kings[int(actor)].bit_count() > actor_king_counts[lane]
                self._update_active_game(
                    lane=lane,
                    actor=actor,
                    value=float(output.value[lane].item()),
                    is_capture=step.is_capture,
                    move_completed=bool(info["move_completed"]),
                    promoted=promoted,
                )
                if bool(terminated_array[lane]):
                    outcome = info["outcome"]
                    if not isinstance(outcome, Outcome):
                        raise RuntimeError(
                            "terminal self-play transition did not contain an Outcome"
                        )
                    completed_games.append(self._record_terminal(lane=lane, outcome=outcome))

            buffer.append(
                RolloutStep(
                    obs=observations,
                    legal_mask=legal_mask,
                    action=actions.to(torch.int64),
                    behaviour_logprob=log_probabilities,
                    value=output.value,
                    reward=rewards,
                    done=dones,
                    actor=torch.tensor(
                        [int(actor) for actor in actors],
                        dtype=torch.int64,
                        device=self._device,
                    ),
                    sigma=sigmas,
                    trainable=torch.ones(
                        self._config.num_envs,
                        dtype=torch.bool,
                        device=self._device,
                    ),
                    policy_id=(CURRENT_POLICY_ID,) * self._config.num_envs,
                    env_id=torch.arange(self._config.num_envs, device=self._device),
                    move_completed=move_completed,
                )
            )
            for lane, terminated in enumerate(terminated_array):
                if bool(terminated):
                    self._reset_lane(lane)

        bootstrap_observations = torch.as_tensor(
            self._vector_env.observations(),
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            bootstrap_value = self._network(bootstrap_observations).value
        rollout = buffer.finalize(
            bootstrap_value=bootstrap_value,
            gamma=self._config.gamma,
            gae_lambda=self._config.gae_lambda,
        )
        calibration_predictions = torch.tensor(
            self._calibration_predictions,
            dtype=rollout.value.dtype,
            device=self._device,
        )
        calibration_outcomes = torch.tensor(
            self._calibration_outcomes,
            dtype=rollout.value.dtype,
            device=self._device,
        )
        metrics = self._mask_statistics.metrics()
        metrics.update(
            policy_health(
                logits=torch.cat(rollout_logits, dim=0),
                legal_mask=torch.cat(rollout_masks, dim=0),
            )
        )
        metrics.update(
            value_health(
                values=rollout.value,
                targets=rollout.returns,
                calibration_predictions=calibration_predictions,
                realized_outcomes=calibration_outcomes,
            )
        )
        metrics.update(self._game_statistics.metrics())
        return CollectionResult(
            rollout=rollout,
            metrics=metrics,
            completed_games=tuple(completed_games),
        )
