"""Gymnasium environment for step-wise American Checkers play."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from checkers.env.encoding import (
    BOARD_SIZE,
    DEFAULT_MAX_PLIES,
    OBSERVATION_PLANES,
    encode_observation,
)
from checkers.env.masking import (
    ACTION_COUNT,
    ActionEncodingError,
    action_to_step,
    legal_action_mask,
)
from checkers.env.serialize import (
    EnvironmentSnapshot,
    parse_environment_snapshot,
    serialize_environment_snapshot,
)
from checkers.rules.board import bit, coord
from checkers.rules.moves import apply_step
from checkers.rules.notation import MovePath, format_move
from checkers.rules.state import PlayerId, State
from checkers.rules.terminal import Outcome, terminal_outcome
from checkers.rules.zobrist import incremental_state_key
from checkers.rules.zobrist import position_key as compute_position_key
from checkers.rules.zobrist import state_key as compute_state_key

Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
StepResult = tuple[Float32Array, float, bool, bool, dict[str, Any]]


class IllegalActionError(ValueError):
    """Raised when an action cannot legally be applied by the environment."""


def _validate_config(
    max_plies: int,
    repetition_draws: bool,
    initial_state: State,
    render_mode: str | None,
) -> None:
    if isinstance(max_plies, bool) or not isinstance(max_plies, int):
        raise TypeError("max_plies must be an integer")
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    if not isinstance(repetition_draws, bool):
        raise TypeError("repetition_draws must be bool")
    if not isinstance(initial_state, State):
        raise TypeError("initial_state must be a State")
    if initial_state.capture_in_progress:
        raise ValueError("initial_state must be a completed-move boundary")
    if render_mode not in (None, "ansi"):
        raise ValueError("render_mode must be None or 'ansi'")


class CheckersEnv(gym.Env[Float32Array, int]):
    """Deterministic canonical checkers environment with one jump per step."""

    metadata: dict[str, Any] = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        max_plies: int = DEFAULT_MAX_PLIES,
        repetition_draws: bool = False,
        initial_state: State | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize a deterministic environment and immutable rule configuration.

        Args:
            max_plies: R6.5 environment-step terminal boundary.
            repetition_draws: Enable optional arena-only threefold repetition.
            initial_state: Completed-move state restored on reset; defaults to WCDF initial play.
            render_mode: Optional Gymnasium render mode; only ``"ansi"`` is supported.

        Raises:
            TypeError: If configuration values have the wrong runtime types.
            ValueError: If values are out of range or an unsupported render mode is requested.
        """

        super().__init__()
        configured_initial = State.initial() if initial_state is None else initial_state
        _validate_config(max_plies, repetition_draws, configured_initial, render_mode)
        self.max_plies = max_plies
        self.repetition_draws = repetition_draws
        self.render_mode = render_mode
        self._initial_state = configured_initial
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        low = np.zeros((OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        high = np.ones((OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        high[6:].fill(np.finfo(np.float32).max)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._state = configured_initial
        self._state_key = compute_state_key(configured_initial)
        self._position_counts: dict[int, int] = {}
        self._active_move_squares: tuple[int, ...] = ()
        self._outcome: Outcome | None = None
        self._terminated = False
        self._reset_dynamic_state()

    @property
    def state(self) -> State:
        """Return the current immutable rules state."""

        return self._state

    @property
    def outcome(self) -> Outcome | None:
        """Return the terminal outcome, if the episode has ended."""

        return self._outcome

    @property
    def terminated(self) -> bool:
        """Return whether a game rule has ended the episode."""

        return self._terminated

    def _repetition_count(self, state: State) -> int:
        if state.capture_in_progress:
            return 0
        return self._position_counts.get(compute_position_key(state), 0)

    def _evaluate(self, state: State) -> Outcome | None:
        return terminal_outcome(
            state,
            max_plies=self.max_plies,
            repetition_draws=self.repetition_draws,
            repetition_count=self._repetition_count(state),
        )

    def _reset_dynamic_state(self) -> None:
        self._state = self._initial_state
        self._state_key = compute_state_key(self._state)
        initial_position = compute_position_key(self._state)
        self._position_counts = {initial_position: 1}
        self._active_move_squares = ()
        self._outcome = self._evaluate(self._state)
        self._terminated = self._outcome is not None

    def _info(
        self,
        *,
        actor: PlayerId,
        move_completed: bool,
        checkers_move_san: str | None,
    ) -> dict[str, Any]:
        return {
            "legal_mask": self.legal_mask(),
            "actor": actor,
            "move_completed": move_completed,
            "checkers_move_san": checkers_move_san,
            "outcome": self._outcome,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Float32Array, dict[str, Any]]:
        """Reset to the configured initial boundary state.

        Args:
            seed: Gymnasium-compatible RNG seed; transitions themselves are deterministic.
            options: Reserved for future schemas; only ``None`` or an empty dictionary is valid.

        Returns:
            The canonical observation and exact environment info mapping.

        Raises:
            TypeError: If ``options`` is not a dictionary or ``None``.
            ValueError: If any unsupported reset option is supplied.
        """

        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        if options:
            raise ValueError("reset options are not supported")
        self._reset_dynamic_state()
        return self.observe(), self._info(
            actor=self._state.side_to_move,
            move_completed=False,
            checkers_move_san=None,
        )

    def observe(self) -> Float32Array:
        """Return a fresh canonical observation of the current state.

        Returns:
            A float32 array with shape ``(8, 8, 8)``.
        """

        return encode_observation(self._state, max_plies=self.max_plies)

    def legal_mask(self) -> BoolArray:
        """Return a fresh fixed-width mask for actions accepted by ``step``.

        Returns:
            Boolean shape-``(128,)`` mask, all false after termination.
        """

        if self._terminated:
            return np.zeros(ACTION_COUNT, dtype=np.bool_)
        return legal_action_mask(self._state)

    def state_key(self) -> int:
        """Return the incrementally maintained complete-state Zobrist key.

        Returns:
            Unsigned frozen-schema 64-bit state key.
        """

        return self._state_key

    def position_key(self) -> int:
        """Return the official-position key at a completed-move boundary.

        Returns:
            Placement-plus-side Zobrist key.

        Raises:
            ValueError: If a capture sequence is in progress.
        """

        return compute_position_key(self._state)

    def _completed_move_notation(self, step_is_capture: bool) -> str:
        move = MovePath(squares=self._active_move_squares, is_capture=step_is_capture)
        notation = format_move(move)
        self._active_move_squares = ()
        return notation

    def step(self, action: int) -> StepResult:
        """Apply exactly one simple move or one jump of a capture sequence.

        Args:
            action: Canonical action ID in ``[0, 127]``.

        Returns:
            Gymnasium ``(observation, reward, terminated, truncated, info)`` tuple. Game rules
            never set ``truncated``.

        Raises:
            IllegalActionError: If the episode ended or the action is not legal in this state.
        """

        if self._terminated:
            raise IllegalActionError("cannot step a terminated environment")
        try:
            step = action_to_step(self._state, action)
        except (ActionEncodingError, TypeError, ValueError) as error:
            raise IllegalActionError(f"illegal action {action!r}") from error

        actor = self._state.side_to_move
        before = self._state
        if self._active_move_squares:
            self._active_move_squares += (step.destination,)
        else:
            self._active_move_squares = (step.origin, step.destination)
        transition = apply_step(before, step)
        self._state = transition.after
        self._state_key = incremental_state_key(self._state_key, before, self._state)

        checkers_move_san: str | None = None
        if transition.move_completed:
            checkers_move_san = self._completed_move_notation(step.is_capture)
            current_position = compute_position_key(self._state)
            self._position_counts[current_position] = (
                self._position_counts.get(current_position, 0) + 1
            )

        self._outcome = self._evaluate(self._state)
        self._terminated = self._outcome is not None
        reward = 0.0 if self._outcome is None else float(self._outcome.score_for(actor))
        return (
            self.observe(),
            reward,
            self._terminated,
            False,
            self._info(
                actor=actor,
                move_completed=transition.move_completed,
                checkers_move_san=checkers_move_san,
            ),
        )

    def serialize(self) -> str:
        """Serialize all state required to resume this environment exactly.

        Returns:
            Canonical versioned JSON snapshot.
        """

        snapshot = EnvironmentSnapshot(
            state=self._state,
            initial_state=self._initial_state,
            max_plies=self.max_plies,
            repetition_draws=self.repetition_draws,
            position_counts=tuple(sorted(self._position_counts.items())),
            active_move_squares=self._active_move_squares,
        )
        return serialize_environment_snapshot(snapshot)

    def _restore_snapshot(self, snapshot: EnvironmentSnapshot) -> None:
        if (
            snapshot.max_plies != self.max_plies
            or snapshot.repetition_draws != self.repetition_draws
        ):
            raise ValueError("snapshot configuration does not match environment configuration")

        position_counts = dict(snapshot.position_counts)
        repetition_count = (
            0
            if snapshot.state.capture_in_progress
            else position_counts[compute_position_key(snapshot.state)]
        )
        outcome = terminal_outcome(
            snapshot.state,
            max_plies=self.max_plies,
            repetition_draws=self.repetition_draws,
            repetition_count=repetition_count,
        )

        self._state = snapshot.state
        self._initial_state = snapshot.initial_state
        self._state_key = compute_state_key(snapshot.state)
        self._position_counts = position_counts
        self._active_move_squares = snapshot.active_move_squares
        self._outcome = outcome
        self._terminated = outcome is not None

    def restore(self, text: str) -> None:
        """Atomically restore a canonical snapshot into a compatible environment.

        Args:
            text: Snapshot returned by ``serialize``.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If parsing fails or immutable rule configuration differs.
        """

        snapshot = parse_environment_snapshot(text)
        self._restore_snapshot(snapshot)

    @classmethod
    def from_serialized(
        cls,
        text: str,
        *,
        render_mode: str | None = None,
    ) -> CheckersEnv:
        """Construct an environment using configuration and state from a snapshot.

        Args:
            text: Canonical environment snapshot.
            render_mode: Optional local presentation mode, which is not persisted.

        Returns:
            A new exactly resumed environment.

        Raises:
            TypeError: If snapshot input types are invalid.
            ValueError: If snapshot content or render configuration is invalid.
        """

        snapshot = parse_environment_snapshot(text)
        environment = cls(
            max_plies=snapshot.max_plies,
            repetition_draws=snapshot.repetition_draws,
            initial_state=snapshot.initial_state,
            render_mode=render_mode,
        )
        environment._restore_snapshot(snapshot)
        return environment

    def _piece_symbol(self, square: int) -> str:
        square_bit = bit(square)
        if self._state.men[PlayerId.RED] & square_bit:
            return "r"
        if self._state.kings[PlayerId.RED] & square_bit:
            return "R"
        if self._state.men[PlayerId.WHITE] & square_bit:
            return "w"
        if self._state.kings[PlayerId.WHITE] & square_bit:
            return "W"
        return "."

    def render(self, mode: str | None = None) -> str:
        """Render an ASCII board with ACF numbers and sequence-state marks.

        Args:
            mode: ``"ansi"`` or ``None``. ``None`` uses the configured/default ANSI mode.

        Returns:
            Human-readable ASCII board text.

        Raises:
            ValueError: If a mode other than ``"ansi"`` is requested.
        """

        requested_mode = self.render_mode if mode is None else mode
        if requested_mode not in (None, "ansi"):
            raise ValueError("only ansi rendering is supported")
        tokens: dict[tuple[int, int], str] = {}
        for square in range(32):
            symbol = self._piece_symbol(square)
            marker = " "
            if bit(square) & self._state.captured_pending:
                marker = "*"
            elif square == self._state.moving_square:
                marker = "@"
            tokens[coord(square)] = f"{square + 1:02d}:{symbol}{marker}"

        lines = [
            f"Checkers actor={self._state.side_to_move.name} ply={self._state.ply} "
            f"no_progress={self._state.no_progress[0]},{self._state.no_progress[1]}"
        ]
        for row in reversed(range(BOARD_SIZE)):
            cells = [tokens.get((row, column), "     ") for column in range(BOARD_SIZE)]
            lines.append(f"{row} |" + "|".join(cells) + "|")
        lines.append("Legend: r/R=Red man/king; w/W=White man/king; *=captured-pending; @=forced")
        return "\n".join(lines)
