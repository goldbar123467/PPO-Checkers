"""Minimal loopback JSON/static server for the local checkers harness."""

from __future__ import annotations

import json
import mimetypes
import re
import socket
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from checkers.web.game import GameError, GameService

MAX_REQUEST_BYTES = 16 * 1024
MAX_PORT_EXCLUSIVE = 1 << 16
GAME_PATH_SLASH_COUNT = 3
MOVE_PATH_PART_COUNT = 5
ServerRequest = socket.socket | tuple[bytes, socket.socket]
HASHED_ASSET_PATTERN = re.compile(r"[.-][A-Za-z0-9_-]{8,}\.")


@dataclass(frozen=True, slots=True)
class WebServerConfig:
    """Immutable bind and static-file settings."""

    host: str = "127.0.0.1"
    port: int = 8765
    static_dir: Path | None = None
    max_concurrent_requests: int = 8
    request_timeout_seconds: int = 15

    def __post_init__(self) -> None:
        if self.host != "127.0.0.1":
            raise ValueError("the checkers web harness must bind to 127.0.0.1")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 0 <= self.port < MAX_PORT_EXCLUSIVE
        ):
            raise ValueError("port must be an integer from 0 through 65535")
        if self.static_dir is not None and not isinstance(self.static_dir, Path):
            raise TypeError("static_dir must be a Path or None")
        for name, value in (
            ("max_concurrent_requests", self.max_concurrent_requests),
            ("request_timeout_seconds", self.request_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


class CheckersWebServer(ThreadingHTTPServer):
    """Threaded local server carrying its application dependencies."""

    daemon_threads = True
    request_queue_size = 32

    def __init__(self, config: WebServerConfig, service: GameService) -> None:
        self.service = service
        self.static_dir = None if config.static_dir is None else config.static_dir.resolve()
        self.request_timeout_seconds = config.request_timeout_seconds
        self._request_slots = threading.BoundedSemaphore(config.max_concurrent_requests)
        super().__init__((config.host, config.port), CheckersRequestHandler)

    def process_request(self, request: ServerRequest, client_address: tuple[Any, ...]) -> None:
        """Apply backpressure before spawning a bounded request thread."""

        self._request_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(
        self, request: ServerRequest, client_address: tuple[Any, ...]
    ) -> None:
        """Release one bounded request slot after the handler exits."""

        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class CheckersRequestHandler(BaseHTTPRequestHandler):
    """Strict same-origin API routes plus optional built frontend assets."""

    server: CheckersWebServer
    server_version = "checkers-web"
    sys_version = ""

    def setup(self) -> None:
        """Apply a finite socket timeout to every accepted connection."""

        super().setup()
        self.connection.settimeout(self.server.request_timeout_seconds)
        self._request_started_at = time.monotonic()
        self._error_code: str | None = None

    def log_message(self, format_: str, *args: object) -> None:
        """Suppress the unsafe free-form base-class access log."""

        _ = format_, args

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Emit one stable JSON record without query text or game identifiers."""

        path = urlsplit(self.path).path
        if path.startswith("/api/games/") and path.endswith("/moves"):
            route = "/api/games/:id/moves"
        elif path.startswith("/api/games/"):
            route = "/api/games/:id"
        elif path.startswith("/assets/"):
            route = "/assets/:asset"
        else:
            route = path
        record: dict[str, object] = {
            "event": "http_request",
            "method": self.command,
            "route": route,
            "status": code,
            "responseBytes": size,
            "durationMs": round((time.monotonic() - self._request_started_at) * 1000, 3),
            "client": self.client_address[0],
        }
        if self._error_code is not None:
            record["errorCode"] = self._error_code
        print(json.dumps(record, separators=(",", ":"), sort_keys=True), flush=True)

    def do_GET(self) -> None:  # noqa: N802
        """Serve health/model/game JSON or a safe static file."""

        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "modelReady": True})
            elif path == "/api/model":
                self._json(HTTPStatus.OK, self.server.service.model_snapshot())
            elif path.startswith("/api/games/") and path.count("/") == GAME_PATH_SLASH_COUNT:
                game_id = unquote(path.rsplit("/", 1)[1])
                self._json(HTTPStatus.OK, self.server.service.get_game(game_id))
            elif path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "route_not_found", "API route does not exist")
            else:
                self._static(path)
        except GameError as error:
            self._error(error.status, error.code, str(error))

    def do_HEAD(self) -> None:  # noqa: N802
        """Return GET-equivalent headers without transferring a response body."""

        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        """Create a game or apply a human step."""

        path = urlsplit(self.path).path
        try:
            body = self._request_json()
            if path == "/api/games":
                response = self.server.service.create_game(
                    human_color=body.get("humanColor"),
                    policy_mode=body.get("policyMode"),
                    seed=body.get("seed"),
                )
                self._json(HTTPStatus.CREATED, response)
                return
            if path.startswith("/api/games/") and path.endswith("/moves"):
                parts = path.split("/")
                if len(parts) == MOVE_PATH_PART_COUNT and parts[3]:
                    response = self.server.service.apply_human_step(
                        game_id=unquote(parts[3]),
                        origin=body.get("origin"),
                        destination=body.get("destination"),
                    )
                    self._json(HTTPStatus.OK, response)
                    return
            self._error(HTTPStatus.NOT_FOUND, "route_not_found", "API route does not exist")
        except GameError as error:
            self._error(error.status, error.code, str(error))

    def _request_json(self) -> dict[str, object]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise GameError(
                "invalid_content_type",
                "Content-Type must be application/json",
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError as error:
            raise GameError("invalid_content_length", "Content-Length is invalid") from error
        if not 0 <= length <= MAX_REQUEST_BYTES:
            raise GameError("invalid_content_length", "request body size is invalid", status=413)
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GameError("invalid_json", "request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise GameError("invalid_json", "request JSON must be an object")
        return cast(dict[str, object], payload)

    def _static(self, request_path: str) -> None:
        static_dir = self.server.static_dir
        if static_dir is None or not static_dir.is_dir():
            self._error(HTTPStatus.NOT_FOUND, "frontend_not_built", "frontend build is unavailable")
            return
        relative = unquote(request_path).lstrip("/") or "index.html"
        candidate = (static_dir / relative).resolve()
        if static_dir not in candidate.parents and candidate != static_dir:
            self._error(HTTPStatus.NOT_FOUND, "asset_not_found", "asset does not exist")
            return
        if not candidate.is_file() and "." not in Path(relative).name:
            candidate = static_dir / "index.html"
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "asset_not_found", "asset does not exist")
            return
        content_type, _encoding = mimetypes.guess_type(candidate.name)
        payload = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        if candidate.name == "index.html":
            # Keep intermediaries from injecting scripts into the student interface.
            cache_control = "no-store, no-transform"
        elif HASHED_ASSET_PATTERN.search(candidate.name):
            cache_control = "public, max-age=31536000, immutable"
        else:
            cache_control = "public, max-age=3600"
        self.send_header("Cache-Control", cache_control)
        self._security_headers()
        self.end_headers()
        self._write_payload(payload)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self._write_payload(encoded)

    def _write_payload(self, payload: bytes) -> None:
        """Write a body unless this is HEAD, tolerating a browser that has navigated away."""

        if self.command == "HEAD":
            return
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _error(self, status: int, code: str, message: str) -> None:
        self._error_code = code
        self._json(status, {"error": {"code": code, "message": message}})

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), geolocation=(), microphone=()")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "font-src 'self' data:; style-src 'self'; "
            "script-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )


def create_server(config: WebServerConfig, service: GameService) -> CheckersWebServer:
    """Construct but do not start a configured checkers web server."""

    if not isinstance(config, WebServerConfig):
        raise TypeError("config must be WebServerConfig")
    if not isinstance(service, GameService):
        raise TypeError("service must be GameService")
    return CheckersWebServer(config, service)
