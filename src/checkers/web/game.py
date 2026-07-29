"""Server-authoritative human-versus-policy checkers game sessions."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypedDict, cast

from checkers.agents.policy_agent import PolicyAgent, PolicyMode
from checkers.env.checkers_env import CheckersEnv
from checkers.env.masking import legal_action_map
from checkers.rules.board import (
    BOARD_SIZE,
    PLAYABLE_SQUARES,
    acf_number,
    bit,
    coord,
    is_playable_coord,
)
from checkers.rules.state import PlayerId
from checkers.web.policy_bundle import LoadedPolicy

ColorName = Literal["red", "white"]


class GameError(RuntimeError):
    """A stable client-facing game operation failure."""

    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class MoveRecord(TypedDict):
    ply: int
    actor: ColorName
    notation: str


@dataclass(frozen=True, slots=True)
class GameRetention:
    """Hard limits for the in-memory public game store."""

    max_active_games: int = 256
    idle_ttl_seconds: int = 6 * 60 * 60

    def __post_init__(self) -> None:
        for name, value in (
            ("max_active_games", self.max_active_games),
            ("idle_ttl_seconds", self.idle_ttl_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(slots=True)
class GameSession:
    """Mutable runtime state for one in-memory browser game."""

    game_id: str
    human: PlayerId
    policy_mode: PolicyMode
    seed: int
    environment: CheckersEnv
    agent: PolicyAgent
    last_accessed_at: float
    moves: list[MoveRecord] = field(default_factory=list)
    last_step: tuple[int, int] | None = None

    @property
    def model(self) -> PlayerId:
        return self.human.opponent


def _color(player: PlayerId) -> ColorName:
    return "red" if player is PlayerId.RED else "white"


def _parse_color(value: object) -> PlayerId:
    if value == "red":
        return PlayerId.RED
    if value == "white":
        return PlayerId.WHITE
    raise GameError("invalid_human_color", "humanColor must be red or white")


def _parse_mode(value: object) -> PolicyMode:
    match value:
        case "greedy":
            return "greedy"
        case "sampled":
            return "sampled"
        case _:
            raise GameError("invalid_policy_mode", "policyMode must be greedy or sampled")


def _parse_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 64):
        raise GameError("invalid_seed", "seed must be an unsigned 64-bit integer")
    return value


class GameService:
    """Own the loaded local policy and all ephemeral games."""

    def __init__(
        self,
        loaded_policy: LoadedPolicy,
        *,
        retention: GameRetention | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(loaded_policy, LoadedPolicy):
            raise TypeError("loaded_policy must be LoadedPolicy")
        configured_retention = GameRetention() if retention is None else retention
        if not isinstance(configured_retention, GameRetention):
            raise TypeError("retention must be GameRetention or None")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._policy = loaded_policy
        self._retention = configured_retention
        self._clock = clock
        self._games: dict[str, GameSession] = {}
        self._lock = threading.RLock()

    def model_snapshot(self) -> dict[str, object]:
        """Return non-secret, validated runtime model metadata."""

        metadata = self._policy.metadata
        return {
            "ready": True,
            "bundleId": metadata.bundle_id,
            "experimentId": metadata.experiment_id,
            "update": metadata.update_idx,
            "globalStep": metadata.global_step,
            "sourceCheckpoint": metadata.source_checkpoint,
            "sourceCheckpointSha256": metadata.source_checkpoint_sha256,
            "bundleSha256": self._policy.sha256,
            "bundleSizeBytes": self._policy.size_bytes,
            "gitSha": metadata.source_git_sha,
            "gitDirty": metadata.source_git_dirty,
            "device": "cpu",
            "actionCount": metadata.action_count,
            "maxPlies": metadata.max_plies,
            "repetitionDraws": metadata.repetition_draws,
            "parameterCount": sum(
                parameter.numel() for parameter in self._policy.network.parameters()
            ),
        }

    def create_game(
        self, *, human_color: object, policy_mode: object, seed: object
    ) -> dict[str, object]:
        """Create one game and play the opening policy turn when the human chose White."""

        human = _parse_color(human_color)
        mode = _parse_mode(policy_mode)
        checked_seed = _parse_seed(seed)
        with self._lock:
            now = self._clock()
            self._cleanup_expired(now)
            if len(self._games) >= self._retention.max_active_games:
                raise GameError(
                    "game_capacity_reached",
                    "the server is at its active-game limit; try again later",
                    status=503,
                )
            environment = CheckersEnv(
                max_plies=self._policy.metadata.max_plies,
                repetition_draws=self._policy.metadata.repetition_draws,
            )
            environment.reset(seed=checked_seed)
            session = GameSession(
                game_id=str(uuid.uuid4()),
                human=human,
                policy_mode=mode,
                seed=checked_seed,
                environment=environment,
                agent=PolicyAgent(
                    network=self._policy.network,
                    mode=mode,
                    seed=checked_seed,
                    name=f"web-{mode}",
                ),
                last_accessed_at=now,
            )
            self._games[session.game_id] = session
            self._play_model_turn(session)
            return self._snapshot(session)

    def get_game(self, game_id: str) -> dict[str, object]:
        """Return the current complete client snapshot for one game."""

        with self._lock:
            now = self._clock()
            self._cleanup_expired(now)
            session = self._require_game(game_id)
            session.last_accessed_at = now
            return self._snapshot(session)

    def apply_human_step(
        self, *, game_id: str, origin: object, destination: object
    ) -> dict[str, object]:
        """Validate and apply one human step, then complete the model response turn."""

        checked_origin = self._square(origin, "origin")
        checked_destination = self._square(destination, "destination")
        with self._lock:
            now = self._clock()
            self._cleanup_expired(now)
            session = self._require_game(game_id)
            environment = session.environment
            if environment.terminated:
                raise GameError("game_over", "the game has already ended", status=409)
            if environment.state.side_to_move is not session.human:
                raise GameError("not_human_turn", "wait for the model turn to finish", status=409)
            matching = [
                action
                for action, step in legal_action_map(environment.state).items()
                if step.origin == checked_origin and step.destination == checked_destination
            ]
            if len(matching) != 1:
                raise GameError(
                    "illegal_move",
                    "that origin and destination do not identify a current legal move",
                )
            self._apply_action(session, matching[0])
            self._play_model_turn(session)
            session.last_accessed_at = now
            return self._snapshot(session)

    def _cleanup_expired(self, now: float) -> None:
        expired = sorted(
            game_id
            for game_id, session in self._games.items()
            if now - session.last_accessed_at >= self._retention.idle_ttl_seconds
        )
        for game_id in expired:
            del self._games[game_id]

    @staticmethod
    def _square(value: object, name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < PLAYABLE_SQUARES
        ):
            raise GameError("invalid_square", f"{name} must be an integer from 0 through 31")
        return value

    def _require_game(self, game_id: str) -> GameSession:
        if not isinstance(game_id, str) or not game_id:
            raise GameError("invalid_game_id", "game id must be non-empty text")
        try:
            return self._games[game_id]
        except KeyError as error:
            raise GameError("game_not_found", "game does not exist", status=404) from error

    @staticmethod
    def _apply_action(session: GameSession, action: int) -> None:
        before = session.environment.state
        step = legal_action_map(before)[action]
        _observation, _reward, _terminated, _truncated, info = session.environment.step(action)
        session.last_step = (step.origin, step.destination)
        notation = info["checkers_move_san"]
        if notation is not None:
            session.moves.append(
                {
                    "ply": len(session.moves) + 1,
                    "actor": _color(cast(PlayerId, info["actor"])),
                    "notation": cast(str, notation),
                }
            )

    def _play_model_turn(self, session: GameSession) -> None:
        environment = session.environment
        while not environment.terminated and environment.state.side_to_move is session.model:
            self._apply_action(session, session.agent.select_action(environment.state))

    @staticmethod
    def _pieces(session: GameSession) -> list[dict[str, object]]:
        state = session.environment.state
        pieces: list[dict[str, object]] = []
        for square in range(32):
            square_bit = bit(square)
            for player in PlayerId:
                if square_bit & state.men[player]:
                    row, column = coord(square)
                    pieces.append(
                        {
                            "square": square,
                            "row": row,
                            "column": column,
                            "color": _color(player),
                            "kind": "man",
                        }
                    )
                elif square_bit & state.kings[player]:
                    row, column = coord(square)
                    pieces.append(
                        {
                            "square": square,
                            "row": row,
                            "column": column,
                            "color": _color(player),
                            "kind": "king",
                        }
                    )
        return pieces

    @staticmethod
    def _board() -> list[dict[str, object]]:
        cells: list[dict[str, object]] = []
        for row in range(BOARD_SIZE):
            for column in range(BOARD_SIZE):
                playable = is_playable_coord(row, column)
                cells.append(
                    {
                        "row": row,
                        "column": column,
                        "playable": playable,
                        "square": acf_number(row, column) - 1 if playable else None,
                    }
                )
        return cells

    def _snapshot(self, session: GameSession) -> dict[str, object]:
        environment = session.environment
        state = environment.state
        human_turn = not environment.terminated and state.side_to_move is session.human
        legal_moves: list[dict[str, object]] = []
        if human_turn:
            for action, step in legal_action_map(state).items():
                legal_moves.append(
                    {
                        "action": action,
                        "origin": step.origin,
                        "destination": step.destination,
                        "captured": step.captured,
                    }
                )
        outcome = environment.outcome
        outcome_snapshot = None
        if outcome is not None:
            outcome_snapshot = {
                "winner": None if outcome.winner is None else _color(outcome.winner),
                "reason": outcome.reason.value,
                "isDraw": outcome.is_draw,
            }
        return {
            "id": session.game_id,
            "humanColor": _color(session.human),
            "modelColor": _color(session.model),
            "policyMode": session.policy_mode,
            "seed": session.seed,
            "sideToMove": _color(state.side_to_move),
            "isHumanTurn": human_turn,
            "captureInProgress": state.capture_in_progress,
            "forcedSquare": state.moving_square,
            "ply": state.ply,
            "board": self._board(),
            "pieces": self._pieces(session),
            "legalMoves": legal_moves,
            "lastStep": None
            if session.last_step is None
            else {"origin": session.last_step[0], "destination": session.last_step[1]},
            "moves": list(session.moves),
            "outcome": outcome_snapshot,
        }
