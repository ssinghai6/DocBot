"""Security response headers for the FastAPI backend (DOCBOT-1619, SEC-11).

The backend only serves JSON, SSE streams and 302 redirects, so it gets the
strictest possible CSP: nothing may load, nothing may frame it. The only HTML
it serves is FastAPI's interactive docs (``/docs``, ``/redoc``), which pull
Swagger/ReDoc assets from a CDN; those paths get the non-CSP headers only.
DOCBOT-2596: those docs (and the exemption) exist only when ``ENV=dev``.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) so it never
buffers streaming SSE responses: it only rewrites the ``http.response.start``
message's header list.
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, MutableMapping, Optional

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# JSON API: no subresources, no framing, no form posts, no <base>.
API_CSP = (
    "default-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

HSTS_VALUE = "max-age=31536000; includeSubDomains"

PERMISSIONS_POLICY = (
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
    "browsing-topics=()"
)

# DOCBOT-2596: FastAPI's /docs, /redoc and full /openapi.json list every
# internal route, so they exist only when ENV=dev. Read once at import.
DOCS_ENABLED: bool = os.getenv("ENV", "").lower() == "dev"

# Paths serving FastAPI's CDN-backed HTML docs skip CSP so they render —
# honoured only when DOCS_ENABLED (in prod those paths 404 and keep CSP).
CSP_EXEMPT_PREFIXES: tuple[str, ...] = ("/docs", "/redoc")


def docs_urls() -> dict[str, Optional[str]]:
    """``FastAPI(**docs_urls())`` kwargs: FastAPI defaults in dev, all None (404) otherwise."""
    if DOCS_ENABLED:
        return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}
    return {"docs_url": None, "redoc_url": None, "openapi_url": None}

BASE_SECURITY_HEADERS: dict[str, str] = {
    "Strict-Transport-Security": HSTS_VALUE,
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": PERMISSIONS_POLICY,
}


def security_headers_for_path(path: str) -> dict[str, str]:
    """Return the header set to apply for a request path."""
    headers = dict(BASE_SECURITY_HEADERS)
    if not (DOCS_ENABLED and path.startswith(CSP_EXEMPT_PREFIXES)):
        headers["Content-Security-Policy"] = API_CSP
    return headers


class SecurityHeadersMiddleware:
    """Add security headers to every HTTP response unless already present.

    Headers a route sets explicitly win (we never overwrite), so a handler can
    opt into a different policy if it ever needs one.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        extra = security_headers_for_path(str(scope.get("path", "")))

        async def send_with_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                raw: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                present = {k.decode("latin-1").lower() for k, _ in raw}
                for name, value in extra.items():
                    if name.lower() not in present:
                        raw.append((name.encode("latin-1"), value.encode("latin-1")))
                message["headers"] = raw
            await send(message)

        await self.app(scope, receive, send_with_headers)
