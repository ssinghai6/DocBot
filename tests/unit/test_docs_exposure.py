"""DOCBOT-2596: FastAPI's /docs, /redoc, /openapi.json are closed unless ENV=dev."""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
import api.utils.security_headers as sh  # noqa: E402

DOCS_PATHS = ["/docs", "/redoc", "/openapi.json"]


@pytest.mark.parametrize("path", DOCS_PATHS)
def test_prod_app_404s_docs(path: str) -> None:
    if sh.DOCS_ENABLED:
        pytest.skip("ENV=dev at import time; docs are enabled")
    resp = TestClient(idx.app, raise_server_exceptions=False).get(path)
    assert resp.status_code == 404


def test_prod_docs_urls_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sh, "DOCS_ENABLED", False)
    assert sh.docs_urls() == {"docs_url": None, "redoc_url": None, "openapi_url": None}


def test_dev_serves_docs_without_csp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sh, "DOCS_ENABLED", True)
    app = FastAPI(**sh.docs_urls())
    app.add_middleware(sh.SecurityHeadersMiddleware)
    client = TestClient(app)
    for path in DOCS_PATHS:
        assert client.get(path).status_code == 200, path
    assert "content-security-policy" not in client.get("/docs").headers


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_csp_exemption_is_dev_only(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sh, "DOCS_ENABLED", False)
    assert sh.security_headers_for_path(path)["Content-Security-Policy"] == sh.API_CSP
    monkeypatch.setattr(sh, "DOCS_ENABLED", True)
    assert "Content-Security-Policy" not in sh.security_headers_for_path(path)
