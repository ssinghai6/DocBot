"""DOCBOT-1619 (SEC-11): security headers on the FastAPI backend and Next.js frontend.

Backend: exercises ``SecurityHeadersMiddleware`` on a minimal app (and checks it
is wired into ``api.index``). Frontend: evaluates ``security-headers.mjs`` with a
local ``node`` binary (no network) and asserts the CSP shape. Also pins the
MarkdownRenderer media policy from DOCBOT-2006.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404 - runs the local node binary on a repo file, fixed args
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from api.utils.security_headers import (
    API_CSP,
    HSTS_VALUE,
    SecurityHeadersMiddleware,
    security_headers_for_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Backend middleware ──────────────────────────────────────────────────────


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            yield b"data: one\n\n"
            yield b"data: two\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/custom")
    async def custom() -> JSONResponse:
        return JSONResponse({"ok": True}, headers={"X-Frame-Options": "SAMEORIGIN"})

    app.add_middleware(SecurityHeadersMiddleware)
    return app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(_app())


def test_api_response_has_all_security_headers(client: TestClient) -> None:
    r = client.get("/api/ping")
    assert r.status_code == 200
    assert r.headers["content-security-policy"] == API_CSP
    assert r.headers["strict-transport-security"] == HSTS_VALUE
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in r.headers["permissions-policy"]


def test_api_csp_is_locked_down() -> None:
    assert "default-src 'none'" in API_CSP
    assert "frame-ancestors 'none'" in API_CSP
    assert "unsafe" not in API_CSP


def test_hsts_one_year_with_subdomains() -> None:
    assert HSTS_VALUE == "max-age=31536000; includeSubDomains"


def test_streaming_response_keeps_body_and_gets_headers(client: TestClient) -> None:
    r = client.get("/api/stream")
    assert r.text == "data: one\n\ndata: two\n\n"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == API_CSP


def test_route_set_header_is_not_overwritten(client: TestClient) -> None:
    r = client.get("/api/custom")
    assert r.headers["x-frame-options"] == "SAMEORIGIN"
    # Only one value — no duplicate appended.
    assert r.headers.get_list("x-frame-options") == ["SAMEORIGIN"]


def test_404_also_gets_headers(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize("path", ["/docs", "/docs/oauth2-redirect", "/redoc"])
def test_docs_paths_skip_csp_but_keep_other_headers(path: str) -> None:
    headers = security_headers_for_path(path)
    assert "Content-Security-Policy" not in headers
    assert headers["X-Frame-Options"] == "DENY"


def test_middleware_is_wired_into_main_app() -> None:
    from api.index import app

    classes = [m.cls for m in app.user_middleware]
    assert SecurityHeadersMiddleware in classes
    # Outermost (index 0) so guard short-circuits and CORS preflights are covered.
    assert classes[0] is SecurityHeadersMiddleware


def test_main_app_health_route_has_headers() -> None:
    from api.index import app

    # No `with` block: skip lifespan (DB init) — middleware runs regardless.
    r = TestClient(app).get("/api/does-not-exist-1619")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == API_CSP


# ── Frontend (security-headers.mjs via node) ────────────────────────────────

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node binary not available")


def _frontend_headers(opts: dict[str, object]) -> dict[str, str]:
    assert NODE is not None
    module = (REPO_ROOT / "security-headers.mjs").as_uri()
    script = (
        f"import {{ securityHeaders }} from {json.dumps(module)};"
        f"const h = securityHeaders({json.dumps(opts)});"
        "process.stdout.write(JSON.stringify(Object.fromEntries(h.map(x => [x.key, x.value]))));"
    )
    out = subprocess.run(  # nosec B603 - fixed argv, local node, no shell
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(out.stdout)


def _directives(csp: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for part in csp.split(";"):
        tokens = part.strip().split()
        if tokens:
            result[tokens[0]] = tokens[1:]
    return result


@needs_node
def test_frontend_prod_csp() -> None:
    h = _frontend_headers({"isDev": False, "backendUrl": "docbot-api.up.railway.app"})
    d = _directives(h["Content-Security-Policy"])
    assert "'unsafe-eval'" not in d["script-src"]
    assert d["frame-ancestors"] == ["'none'"]
    assert d["object-src"] == ["'none'"]
    assert "data:" in d["img-src"] and "blob:" in d["img-src"]
    assert "https://docbot-api.up.railway.app" in d["connect-src"]
    assert "https://va.vercel-scripts.com" in d["script-src"]
    assert "data:" in d["font-src"]
    # No preview-only origins in production.
    assert "https://vercel.live" not in h["Content-Security-Policy"]
    assert h["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert h["X-Frame-Options"] == "DENY"
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"


@needs_node
def test_frontend_dev_csp_allows_eval_and_hmr() -> None:
    d = _directives(_frontend_headers({"isDev": True})["Content-Security-Policy"])
    assert "'unsafe-eval'" in d["script-src"]
    assert "ws:" in d["connect-src"]


@needs_node
def test_frontend_preview_allows_vercel_live() -> None:
    d = _directives(
        _frontend_headers({"isDev": False, "isPreview": True})["Content-Security-Policy"]
    )
    assert "https://vercel.live" in d["script-src"]
    assert "https://vercel.live" in d["frame-src"]
    assert "'unsafe-eval'" not in d["script-src"]


@needs_node
def test_frontend_bad_backend_url_is_dropped() -> None:
    d = _directives(
        _frontend_headers({"isDev": False, "backendUrl": "http://["})["Content-Security-Policy"]
    )
    assert d["connect-src"][0] == "'self'"
    assert not any(src.startswith("http://[") for src in d["connect-src"])


def test_next_config_wires_security_headers() -> None:
    cfg = (REPO_ROOT / "next.config.ts").read_text()
    assert 'from "./security-headers.mjs"' in cfg
    assert "async headers()" in cfg


# ── MarkdownRenderer media policy (DOCBOT-2006) ─────────────────────────────


def _markdown_source() -> str:
    return (REPO_ROOT / "src" / "components" / "MarkdownRenderer.tsx").read_text()


def _safe_image_regex() -> re.Pattern[str]:
    m = re.search(r"const SAFE_IMAGE_SRC = /(.+)/i\n", _markdown_source())
    assert m, "SAFE_IMAGE_SRC regex literal not found in MarkdownRenderer.tsx"
    return re.compile(m.group(1), re.IGNORECASE)


@pytest.mark.parametrize(
    "src",
    ["data:image/png;base64,AAAA", "DATA:IMAGE/svg+xml,<svg/>", "blob:https://x/uuid"],
)
def test_markdown_allows_inline_images(src: str) -> None:
    assert _safe_image_regex().match(src)


@pytest.mark.parametrize(
    "src",
    [
        "https://evil.example/pixel.png",
        "http://evil.example/a.png",
        "//evil.example/a.png",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "/local.png",
    ],
)
def test_markdown_blocks_remote_and_script_images(src: str) -> None:
    assert not _safe_image_regex().match(src)


def test_markdown_skips_raw_html_and_uses_url_transform() -> None:
    src = _markdown_source()
    assert re.search(r"<ReactMarkdown[^>]*\bskipHtml\b", src, re.S)
    assert "urlTransform={urlTransform}" in src
    assert "rehype-raw" not in src
