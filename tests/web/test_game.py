"""Human-versus-model service behavior using the real rules and policy adapters."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, cast

import pytest

from checkers.env.masking import ACTION_COUNT
from checkers.web.game import GameError, GameRetention, GameService
from checkers.web.policy_bundle import LoadedPolicy

EXPECTED_COMPLETE_TURN_PLIES = 2
BOARD_CELL_COUNT = 64
INITIAL_PIECE_COUNT = 24
NETWORK_PARAMETER_COUNT = 470_410
REPLACEMENT_SEED = 3


class FakeClock:
    """Controllable monotonic clock for retention tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_red_human_move_triggers_one_complete_model_reply(loaded_policy: LoadedPolicy) -> None:
    """A legal human move must be followed by a legal complete model turn."""

    service = GameService(loaded_policy)
    created = service.create_game(human_color="red", policy_mode="greedy", seed=0)
    legal = created["legalMoves"]
    assert isinstance(legal, list) and legal
    move = legal[0]
    assert isinstance(move, dict)

    updated = service.apply_human_step(
        game_id=str(created["id"]),
        origin=move["origin"],
        destination=move["destination"],
    )

    assert updated["isHumanTurn"] is True
    assert updated["sideToMove"] == "red"
    assert updated["ply"] == EXPECTED_COMPLETE_TURN_PLIES
    moves = updated["moves"]
    board = updated["board"]
    pieces = updated["pieces"]
    assert isinstance(moves, list)
    assert all(isinstance(record, dict) for record in moves)
    assert [record["actor"] for record in moves] == ["red", "white"]
    assert isinstance(board, list) and len(board) == BOARD_CELL_COUNT
    assert isinstance(pieces, list) and len(pieces) == INITIAL_PIECE_COUNT


def test_white_human_waits_for_model_opening(loaded_policy: LoadedPolicy) -> None:
    """Choosing White must make the local policy move before the first snapshot returns."""

    service = GameService(loaded_policy)
    created = service.create_game(human_color="white", policy_mode="sampled", seed=41)

    assert created["sideToMove"] == "white"
    assert created["isHumanTurn"] is True
    assert created["ply"] == 1
    moves = created["moves"]
    assert isinstance(moves, list) and moves
    assert isinstance(moves[0], dict) and moves[0]["actor"] == "red"
    assert created["policyMode"] == "sampled"


def test_illegal_move_is_atomic_and_unknown_game_is_404(loaded_policy: LoadedPolicy) -> None:
    """Rejected browser input must not mutate an existing game."""

    service = GameService(loaded_policy)
    created = service.create_game(human_color="red", policy_mode="greedy", seed=0)
    before = service.get_game(str(created["id"]))

    with pytest.raises(GameError, match="do not identify") as error:
        service.apply_human_step(game_id=str(created["id"]), origin=0, destination=31)
    assert error.value.code == "illegal_move"
    assert service.get_game(str(created["id"])) == before

    with pytest.raises(GameError, match="does not exist") as missing:
        service.get_game("not-a-game")
    assert missing.value.status == HTTPStatus.NOT_FOUND


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("human_color", "black", "invalid_human_color"),
        ("policy_mode", "random", "invalid_policy_mode"),
        ("seed", -1, "invalid_seed"),
    ],
)
def test_game_creation_validates_client_options(
    loaded_policy: LoadedPolicy, field: str, value: object, code: str
) -> None:
    """Untrusted setup values must be rejected with stable error codes."""

    arguments: dict[str, object] = {
        "human_color": "red",
        "policy_mode": "greedy",
        "seed": 0,
    }
    arguments[field] = value
    with pytest.raises(GameError) as error:
        GameService(loaded_policy).create_game(**arguments)
    assert error.value.code == code


def test_model_snapshot_exposes_only_runtime_provenance(loaded_policy: LoadedPolicy) -> None:
    """The browser should receive useful provenance but no model tensors."""

    snapshot = GameService(loaded_policy).model_snapshot()
    assert snapshot["ready"] is True
    assert snapshot["device"] == "cpu"
    assert snapshot["parameterCount"] == NETWORK_PARAMETER_COUNT
    assert snapshot["actionCount"] == ACTION_COUNT
    assert "model_state" not in snapshot


def test_game_store_enforces_capacity_and_expires_idle_games(
    loaded_policy: LoadedPolicy,
) -> None:
    """The public session store must remain bounded and clean up deterministically."""

    clock = FakeClock()
    service = GameService(
        loaded_policy,
        retention=GameRetention(max_active_games=2, idle_ttl_seconds=10),
        clock=clock,
    )
    first = service.create_game(human_color="red", policy_mode="greedy", seed=1)
    service.create_game(human_color="red", policy_mode="greedy", seed=2)

    with pytest.raises(GameError, match="active-game limit") as capacity:
        service.create_game(human_color="red", policy_mode="greedy", seed=3)
    assert capacity.value.code == "game_capacity_reached"
    assert capacity.value.status == HTTPStatus.SERVICE_UNAVAILABLE

    clock.now = 9.0
    service.get_game(str(first["id"]))
    clock.now = 10.0
    replacement = service.create_game(
        human_color="red", policy_mode="greedy", seed=REPLACEMENT_SEED
    )
    assert replacement["seed"] == REPLACEMENT_SEED
    assert service.get_game(str(first["id"]))["seed"] == 1


def test_expired_game_returns_not_found(loaded_policy: LoadedPolicy) -> None:
    """An idle game must become inaccessible at the exact configured boundary."""

    clock = FakeClock()
    service = GameService(
        loaded_policy,
        retention=GameRetention(max_active_games=1, idle_ttl_seconds=5),
        clock=clock,
    )
    game = service.create_game(human_color="red", policy_mode="greedy", seed=7)
    clock.now = 5.0

    with pytest.raises(GameError, match="does not exist") as expired:
        service.get_game(str(game["id"]))
    assert expired.value.code == "game_not_found"


@pytest.mark.parametrize(
    ("field", "value", "error_type", "match"),
    [
        ("max_active_games", True, TypeError, "integer"),
        ("max_active_games", 0, ValueError, "positive"),
        ("idle_ttl_seconds", "six", TypeError, "integer"),
        ("idle_ttl_seconds", -1, ValueError, "positive"),
    ],
)
def test_retention_rejects_invalid_limits(
    field: str, value: object, error_type: type[Exception], match: str
) -> None:
    """Session limits must be positive integers rather than coercible values."""

    arguments: dict[str, object] = {"max_active_games": 2, "idle_ttl_seconds": 10}
    arguments[field] = value
    with pytest.raises(error_type, match=match):
        GameRetention(**cast(Any, arguments))


def test_service_and_move_boundaries_reject_invalid_objects(loaded_policy: LoadedPolicy) -> None:
    """Construction, game identity, and square boundaries must reject malformed caller input."""

    with pytest.raises(TypeError, match="loaded_policy"):
        GameService(cast(Any, object()))
    with pytest.raises(TypeError, match="retention"):
        GameService(loaded_policy, retention=cast(Any, object()))
    with pytest.raises(TypeError, match="clock"):
        GameService(loaded_policy, clock=cast(Any, 7))

    service = GameService(loaded_policy)
    game = service.create_game(human_color="red", policy_mode="greedy", seed=0)
    for value in (True, "1", -1, 32):
        with pytest.raises(GameError, match="integer from 0 through 31"):
            service.apply_human_step(game_id=str(game["id"]), origin=value, destination=0)
    with pytest.raises(GameError, match="non-empty text"):
        service.get_game("")
