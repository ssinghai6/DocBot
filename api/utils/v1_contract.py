"""Public ``/v1`` API contract plumbing (DOCBOT-3901).

* Error envelope: every ``/v1`` error is
  ``{"error": {"type", "code", "message", "request_id"}}``. ``HTTPException``
  with a dict ``detail={"code": ..., "message": ...}`` keeps its code; a string
  detail gets a default code for its status. ``/api/*`` keeps ``{"detail": ...}``.
* ``X-Request-Id`` on every ``/v1`` response (also echoed in the envelope),
  plus the per-key ``X-RateLimit-*`` headers set by ``require_api_key``.
* 400 ``key_in_url`` when an API key arrives in the query string.
* ``build_v1_openapi``: the app's spec filtered to ``/v1/*`` paths.
"""

from __future__ import annotations

import secrets
from http import HTTPStatus
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

V1_PREFIX = "/v1/"
REQUEST_ID_HEADER = "X-Request-Id"

# Query-string names (and the dk_ value prefix) that mean "a key in the URL".
_KEY_PARAM_NAMES = frozenset({"api_key", "apikey", "key", "token", "access_token", "authorization"})

_TYPE_BY_STATUS = {401: "authentication", 403: "permission", 404: "not_found", 429: "rate_limit"}
_DEFAULT_CODE_BY_STATUS = {
    400: "invalid_request",
    401: "invalid_key",
    402: "insufficient_credits",
    403: "missing_scope",
    404: "resource_not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_failed",
    429: "rate_limited",
    500: "internal_error",
    502: "pipeline_failed",
    503: "service_unavailable",
}


def is_v1_path(path: str) -> bool:
    return path.startswith(V1_PREFIX)


def error_type(status: int) -> str:
    """Envelope ``type`` for an HTTP status (shared contract: six types only)."""
    if status >= 500:
        return "server"
    return _TYPE_BY_STATUS.get(status, "invalid_request")


def _default_message(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def _code_and_message(status: int, detail: Any) -> tuple[str, str]:
    default_code = _DEFAULT_CODE_BY_STATUS.get(status, "server_error" if status >= 500 else "invalid_request")
    if isinstance(detail, Mapping):
        # {"code", "message"} is the /v1 convention; QuotaErrorPayload uses "error".
        code = detail.get("code") or detail.get("error") or default_code
        message = detail.get("message") or detail.get("detail") or _default_message(status)
        return str(code), str(message)
    message = str(detail) if detail else _default_message(status)
    if status == 401 and message.startswith(("Missing", "Empty")):
        return "missing_key", message
    return default_code, message


def new_request_id() -> str:
    return f"req_{secrets.token_hex(8)}"


def request_id_for(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if not rid:
        rid = new_request_id()
        request.state.request_id = rid
    return rid


def envelope_response(
    status: int, code: str, message: str, request_id: str, headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    body = {"error": {"type": error_type(status), "code": code, "message": message, "request_id": request_id}}
    return JSONResponse(
        status_code=status, content=body, headers={**(headers or {}), REQUEST_ID_HEADER: request_id},
    )


def v1_error_response(
    request: Request, status: int, detail: Any, headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    code, message = _code_and_message(status, detail)
    return envelope_response(status, code, message, request_id_for(request), headers)


def guard_response(
    request: Request, status: int, detail: Any, headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    """Middleware short-circuit response: envelope on /v1, legacy ``{"detail"}`` elsewhere."""
    if is_v1_path(request.url.path):
        return v1_error_response(request, status, detail, headers)
    return JSONResponse(status_code=status, content={"detail": detail}, headers=dict(headers or {}))


# ---------------------------------------------------------------------------
# Exception handlers (delegate to FastAPI's defaults outside /v1)
# ---------------------------------------------------------------------------


async def _http_exception(request: Request, exc: StarletteHTTPException) -> Response:
    if not is_v1_path(request.url.path):
        return await http_exception_handler(request, exc)
    return v1_error_response(request, exc.status_code, exc.detail, exc.headers)


async def _validation_exception(request: Request, exc: RequestValidationError) -> Response:
    if not is_v1_path(request.url.path):
        return await request_validation_exception_handler(request, exc)
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = f"{loc}: {first.get('msg', 'invalid value')}" if loc else str(first.get("msg", "Invalid request."))
    return v1_error_response(request, 422, {"code": "validation_failed", "message": message})


async def _unhandled_exception(request: Request, exc: Exception) -> Response:
    # Starlette's ServerErrorMiddleware still re-raises after this, so it is logged.
    if not is_v1_path(request.url.path):
        return PlainTextResponse("Internal Server Error", status_code=500)
    return v1_error_response(request, 500, {"code": "internal_error", "message": "An internal error occurred."})


# ---------------------------------------------------------------------------
# ASGI middleware: request id, rate-limit headers, key_in_url
# ---------------------------------------------------------------------------


def key_in_query(query_string: bytes) -> bool:
    for name, value in parse_qsl(query_string.decode("latin-1"), keep_blank_values=True):
        if name.lower() in _KEY_PARAM_NAMES or value.startswith("dk_"):
            return True
    return False


class V1ContractMiddleware:
    """Pure ASGI (never buffers SSE). Only touches ``/v1/*`` requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_v1_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        rid = state.get("request_id") or new_request_id()
        state["request_id"] = rid

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if REQUEST_ID_HEADER.lower() not in headers:
                    headers[REQUEST_ID_HEADER] = rid
                for name, value in (state.get("rate_limit_headers") or {}).items():
                    if name.lower() not in headers:
                        headers[name] = value
            await send(message)

        if key_in_query(scope.get("query_string", b"")):
            response = envelope_response(
                400, "key_in_url",
                "API keys must be sent in the Authorization header, never in the URL. Rotate this key.",
                rid,
            )
            await response(scope, receive, send_with_headers)
            return

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Filtered OpenAPI
# ---------------------------------------------------------------------------

BEARER_SCHEME = {
    "type": "http",
    "scheme": "bearer",
    "description": (
        "Send `Authorization: Bearer <key>`. Test keys start with `dk_test_` (sandbox data, "
        "free daily allowance); live keys start with `dk_live_` (your org's data, billed in credits). "
        "Never put a key in the URL."
    ),
}


def _schema_refs(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            out.add(ref.rsplit("/", 1)[1])
        for value in node.values():
            _schema_refs(value, out)
    elif isinstance(node, list):
        for value in node:
            _schema_refs(value, out)


def build_v1_openapi(app: FastAPI) -> dict[str, Any]:
    """The app's OpenAPI spec reduced to ``/v1/*`` paths and the schemas they reference.

    Generic on path prefix, so routes from any router module are included.
    """
    full = get_openapi(
        title="DocBot API",
        version="1.0.0",
        description="DocBot public developer API (v1). Errors use the `{\"error\": {...}}` envelope.",
        routes=app.routes,
    )
    paths: dict[str, Any] = {}
    for path, item in full.get("paths", {}).items():
        if not is_v1_path(path):
            continue
        for op in item.values():
            # The Authorization header is described by bearerAuth, not as a parameter.
            params = [p for p in op.get("parameters", []) if not (
                p.get("in") == "header" and p.get("name", "").lower() == "authorization"
            )]
            if params:
                op["parameters"] = params
            else:
                op.pop("parameters", None)
        paths[path] = item

    all_schemas = full.get("components", {}).get("schemas", {})
    keep: set[str] = set()
    _schema_refs(paths, keep)
    pending = list(keep)
    while pending:
        found: set[str] = set()
        _schema_refs(all_schemas.get(pending.pop(), {}), found)
        pending.extend(found - keep)
        keep |= found

    return {
        "openapi": full["openapi"],
        "info": full["info"],
        "servers": [{"url": "https://docbot.space"}],
        "paths": paths,
        "components": {
            "schemas": {name: all_schemas[name] for name in sorted(keep) if name in all_schemas},
            "securitySchemes": {"bearerAuth": BEARER_SCHEME},
        },
        "security": [{"bearerAuth": []}],
    }


def install_v1_contract(app: FastAPI) -> None:
    """Register the /v1 middleware and exception handlers on ``app``."""
    app.add_middleware(V1ContractMiddleware)
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_exception)
    app.add_exception_handler(Exception, _unhandled_exception)
