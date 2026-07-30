"""Loopback HTTP contract tests for the web adapter."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import pytest

from checkers.web.game import GameService
from checkers.web.policy_bundle import LoadedPolicy
from checkers.web.server import WebServerConfig, create_server

LOOPBACK_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def _running_server(loaded_policy: LoadedPolicy) -> Iterator[str]:
    server = create_server(WebServerConfig(port=0), GameService(loaded_policy))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _json_request(url: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={} if data is None else {"Content-Type": "application/json"},
    )
    try:
        with LOOPBACK_OPENER.open(request, timeout=3) as response:
            value = json.loads(response.read())
            if not isinstance(value, dict):
                raise TypeError("JSON response must be an object")
            return response.status, cast(dict[str, Any], value)
    except urllib.error.HTTPError as error:
        value = json.loads(error.read())
        if not isinstance(value, dict):
            raise TypeError("JSON error response must be an object") from error
        return error.code, cast(dict[str, Any], value)


def _get(url: str) -> tuple[int, bytes, dict[str, str]]:
    with LOOPBACK_OPENER.open(url, timeout=3) as response:
        return response.status, response.read(), dict(response.headers.items())


def _raw_post(url: str, payload: bytes, content_type: str = "application/json") -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": content_type},
    )
    try:
        with LOOPBACK_OPENER.open(request, timeout=3) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_health_model_and_game_routes(loaded_policy: LoadedPolicy) -> None:
    """The loopback adapter must expose readiness and complete game snapshots."""

    with _running_server(loaded_policy) as root:
        status, health = _json_request(f"{root}/api/health")
        assert status == HTTPStatus.OK and health == {"status": "ok", "modelReady": True}

        status, model = _json_request(f"{root}/api/model")
        assert status == HTTPStatus.OK and model["ready"] is True

        status, game = _json_request(
            f"{root}/api/games",
            {"humanColor": "red", "policyMode": "greedy", "seed": 0},
        )
        assert status == HTTPStatus.CREATED and game["isHumanTurn"] is True
        move = game["legalMoves"][0]
        status, updated = _json_request(
            f"{root}/api/games/{game['id']}/moves",
            {"origin": move["origin"], "destination": move["destination"]},
        )
        assert status == HTTPStatus.OK and len(updated["moves"]) == 2  # noqa: PLR2004


def test_api_errors_are_structured(loaded_policy: LoadedPolicy) -> None:
    """Bad routes and content types must return the frozen error envelope."""

    with _running_server(loaded_policy) as root:
        status, payload = _json_request(f"{root}/api/unknown")
        assert status == HTTPStatus.NOT_FOUND
        assert payload["error"]["code"] == "route_not_found"

        request = urllib.request.Request(f"{root}/api/games", data=b"{}")
        try:
            LOOPBACK_OPENER.open(request, timeout=3)
        except urllib.error.HTTPError as error:
            assert error.code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
            payload = json.loads(error.read())
            assert payload["error"]["code"] == "invalid_content_type"


def test_security_headers_and_static_cache_policy(
    loaded_policy: LoadedPolicy, tmp_path: Path
) -> None:
    """Static and JSON responses must carry the frozen public security policy."""

    static_dir = tmp_path / "dist"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text("<h1>checkers</h1>", encoding="utf-8")
    (assets_dir / "index-abcdef12.js").write_text("export {};", encoding="utf-8")
    server = create_server(
        WebServerConfig(port=0, static_dir=static_dir), GameService(loaded_policy)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        status, _body, headers = _get(root)
        assert status == HTTPStatus.OK
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

        _status, _body, asset_headers = _get(f"{root}/assets/index-abcdef12.js")
        assert asset_headers["Cache-Control"] == "public, max-age=31536000, immutable"

        (assets_dir / "plain.js").write_text("export {};", encoding="utf-8")
        _status, _body, plain_headers = _get(f"{root}/assets/plain.js")
        assert plain_headers["Cache-Control"] == "public, max-age=3600"

        _status, _body, api_headers = _get(f"{root}/api/health")
        assert api_headers["Cache-Control"] == "no-store"
        assert api_headers["X-Content-Type-Options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    ("arguments", "error_type", "match"),
    [
        ({"host": "0.0.0.0"}, ValueError, "bind"),
        ({"port": True}, ValueError, "port"),
        ({"port": 65536}, ValueError, "port"),
        ({"static_dir": "dist"}, TypeError, "static_dir"),
        ({"max_concurrent_requests": False}, TypeError, "integer"),
        ({"max_concurrent_requests": 0}, ValueError, "positive"),
        ({"request_timeout_seconds": "15"}, TypeError, "integer"),
        ({"request_timeout_seconds": -1}, ValueError, "positive"),
    ],
)
def test_server_config_rejects_unsafe_values(
    arguments: dict[str, object], error_type: type[Exception], match: str
) -> None:
    """Public bind and resource limits must reject unsafe or coercible values."""

    with pytest.raises(error_type, match=match):
        WebServerConfig(**cast(Any, arguments))


def test_server_constructor_rejects_wrong_dependency_types(loaded_policy: LoadedPolicy) -> None:
    """The adapter must not accept duck-typed configuration or service objects."""

    service = GameService(loaded_policy)
    with pytest.raises(TypeError, match="config"):
        create_server(cast(Any, object()), service)
    with pytest.raises(TypeError, match="service"):
        create_server(WebServerConfig(port=0), cast(Any, object()))


def test_malformed_requests_and_missing_frontend_are_structured(
    loaded_policy: LoadedPolicy,
) -> None:
    """Malformed JSON, oversized bodies, unknown POSTs, and absent assets must be bounded 4xxs."""

    with _running_server(loaded_policy) as root:
        status, payload = _raw_post(f"{root}/api/games", b"not-json")
        assert status == HTTPStatus.BAD_REQUEST
        assert payload["error"]["code"] == "invalid_json"

        status, payload = _raw_post(f"{root}/api/games", b"[]")
        assert status == HTTPStatus.BAD_REQUEST
        assert payload["error"]["code"] == "invalid_json"

        status, payload = _raw_post(f"{root}/api/games", b"x" * (16 * 1024 + 1))
        assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert payload["error"]["code"] == "invalid_content_length"

        status, payload = _raw_post(f"{root}/api/unknown", b"{}")
        assert status == HTTPStatus.NOT_FOUND
        assert payload["error"]["code"] == "route_not_found"

        status, payload = _json_request(f"{root}/api/games/")
        assert status == HTTPStatus.BAD_REQUEST
        assert payload["error"]["code"] == "invalid_game_id"

        try:
            LOOPBACK_OPENER.open(root, timeout=3)
        except urllib.error.HTTPError as error:
            assert error.code == HTTPStatus.NOT_FOUND
            payload = json.loads(error.read())
            assert payload["error"]["code"] == "frontend_not_built"
