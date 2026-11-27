"""DOCBOT-4019: the SAML ACS session cookie is Secure behind Railway's TLS proxy.

Uvicorn sees http there (no forwarded-allow-ips), so Secure must follow APP_BASE_URL,
not request.url.scheme. No network: SAML processing and provisioning are mocked.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api import audit_service, auth_service, identity_service  # noqa: E402


@pytest.fixture
def saml(monkeypatch: pytest.MonkeyPatch) -> None:
    attrs = {"email": "v@example.com", "name": "V", "provider": "okta", "provider_id": "1"}
    monkeypatch.setattr(auth_service, "is_saml_configured", lambda: True)
    monkeypatch.setattr(auth_service, "build_saml_auth", MagicMock())
    monkeypatch.setattr(auth_service, "process_acs", MagicMock(return_value=attrs))
    monkeypatch.setattr(auth_service, "create_user_session", AsyncMock(return_value="tok"))
    monkeypatch.setattr(identity_service, "identity_from_saml_attrs", lambda a: a)
    monkeypatch.setattr(identity_service, "resolve_provider_login", AsyncMock(return_value="u1"))
    monkeypatch.setattr(idx, "_claim_anon_work", AsyncMock(return_value=None))
    monkeypatch.setattr(audit_service, "log_event", MagicMock())


def _session_cookie() -> str:
    # TestClient talks plain http, just like uvicorn behind the TLS-terminating proxy.
    resp = TestClient(idx.app).post("/api/auth/saml/acs", data={"SAMLResponse": "x"}, follow_redirects=False)
    assert resp.status_code == 302
    return next(c for c in resp.headers.get_list("set-cookie") if c.startswith("docbot_session=tok"))


def test_acs_cookie_secure_when_app_base_url_is_https(saml: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    cookie = _session_cookie()
    assert "secure" in cookie.lower() and "HttpOnly" in cookie


def test_acs_cookie_not_secure_for_http_base_url(saml: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_BASE_URL", "http://localhost:3000")
    assert "secure" not in _session_cookie().lower()
