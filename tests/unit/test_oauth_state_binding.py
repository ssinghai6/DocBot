"""DOCBOT-2331: OAuth state is bound to the browser via an HttpOnly cookie; Google uses PKCE.

No network: httpx and the provisioning helpers are mocked.
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api import audit_service, auth_service, identity_service, oauth_service  # noqa: E402

COOKIE = oauth_service.OAUTH_STATE_COOKIE
ATTRS = {"email": "v@example.com", "name": "V", "provider_id": "1", "provider": "google"}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for k in ("GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"):
        monkeypatch.setenv(k, "x")
    mocks = {
        "github": AsyncMock(return_value=ATTRS),
        "google": AsyncMock(return_value=ATTRS),
        "claim": AsyncMock(return_value=None),
    }
    monkeypatch.setattr(oauth_service, "github_exchange_code", mocks["github"])
    monkeypatch.setattr(oauth_service, "google_exchange_code", mocks["google"])
    monkeypatch.setattr(idx, "_claim_anon_work", mocks["claim"])
    monkeypatch.setattr(identity_service, "identity_from_oauth_attrs", lambda a: a)
    monkeypatch.setattr(identity_service, "resolve_provider_login", AsyncMock(return_value="u1"))
    monkeypatch.setattr(auth_service, "create_user_session", AsyncMock(return_value="tok"))
    monkeypatch.setattr(audit_service, "log_event", MagicMock())
    return mocks


def _start(client: TestClient, provider: str) -> tuple[str, str]:
    resp = client.get(f"/api/auth/{provider}")
    assert resp.status_code == 200
    set_cookie = resp.headers["set-cookie"]
    assert "HttpOnly" in set_cookie and "Path=/api/auth" in set_cookie and "Max-Age=600" in set_cookie
    assert "samesite=lax" in set_cookie.lower()
    state = parse_qs(urlparse(resp.json()["url"]).query)["state"][0]
    return state, resp.cookies[COOKIE]


@pytest.mark.parametrize("provider", ["github", "google"])
def test_callback_without_cookie_is_rejected(env: dict[str, Any], provider: str) -> None:
    starter = TestClient(idx.app)
    state, _ = _start(starter, provider)
    victim = TestClient(idx.app)  # different browser: no oauth_state cookie
    resp = victim.get(f"/api/auth/{provider}/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert resp.status_code in (302, 307) and "auth_error=invalid_state" in resp.headers["location"]
    env[provider].assert_not_called()
    env["claim"].assert_not_called()
    assert "docbot_session" not in resp.headers.get("set-cookie", "")


@pytest.mark.parametrize("provider", ["github", "google"])
def test_callback_with_mismatched_cookie_is_rejected(env: dict[str, Any], provider: str) -> None:
    attacker = TestClient(idx.app)
    attacker_state, _ = _start(attacker, provider)
    victim = TestClient(idx.app)
    _start(victim, provider)  # victim has their own, different state cookie
    resp = victim.get(
        f"/api/auth/{provider}/callback", params={"code": "c", "state": attacker_state}, follow_redirects=False,
    )
    assert "auth_error=invalid_state" in resp.headers["location"]
    assert f'{COOKIE}=""' in resp.headers["set-cookie"]  # deleted on failure too
    env[provider].assert_not_called()
    env["claim"].assert_not_called()


@pytest.mark.parametrize("provider", ["github", "google"])
def test_matching_cookie_logs_in_and_clears_cookie(env: dict[str, Any], provider: str) -> None:
    client = TestClient(idx.app)
    state, cookie = _start(client, provider)
    resp = client.get(f"/api/auth/{provider}/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert "auth_success=1" in resp.headers["location"]
    set_cookie = resp.headers.get_list("set-cookie")
    assert any(c.startswith("docbot_session=tok") for c in set_cookie)
    assert any(c.startswith(f'{COOKIE}=""') for c in set_cookie)
    env[provider].assert_awaited_once()
    env["claim"].assert_awaited_once()

    # single use: replaying the same state with a matching cookie still fails
    replay = TestClient(idx.app)
    replay.cookies.set(COOKIE, cookie, path="/api/auth")
    resp2 = replay = client.get(f"/api/auth/{provider}/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert "auth_error=invalid_state" in resp2.headers["location"]
    assert env[provider].await_count == 1


def test_google_pkce_challenge_matches_verifier_sent_on_exchange(env: dict[str, Any]) -> None:
    client = TestClient(idx.app)
    resp = client.get("/api/auth/google")
    qs = parse_qs(urlparse(resp.json()["url"]).query)
    assert qs["code_challenge_method"] == ["S256"]
    assert "code_verifier" not in qs
    state = qs["state"][0]
    verifier = client.cookies[COOKIE].partition(".")[2]
    assert verifier and verifier not in resp.json()["url"]
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert qs["code_challenge"] == [expected]

    client.get("/api/auth/google/callback", params={"code": "c", "state": state}, follow_redirects=False)
    env["google"].assert_awaited_once_with("c", verifier)


def test_google_exchange_posts_code_verifier() -> None:
    token_resp = MagicMock(json=MagicMock(return_value={"access_token": "t"}), raise_for_status=MagicMock())
    info_resp = MagicMock(json=MagicMock(return_value={"email": "a@b.co", "sub": "1", "email_verified": True}),
                          raise_for_status=MagicMock())
    client = MagicMock(post=AsyncMock(return_value=token_resp), get=AsyncMock(return_value=info_resp))
    ctx = MagicMock(__aenter__=AsyncMock(return_value=client), __aexit__=AsyncMock(return_value=False))
    import asyncio
    with patch.object(oauth_service.httpx, "AsyncClient", return_value=ctx):
        asyncio.run(oauth_service.google_exchange_code("c", "the-verifier"))
    assert client.post.await_args.kwargs["data"]["code_verifier"] == "the-verifier"


def test_check_oauth_state_cookie() -> None:
    assert oauth_service.check_oauth_state_cookie(None, "s") is None
    assert oauth_service.check_oauth_state_cookie("s", "") is None
    assert oauth_service.check_oauth_state_cookie("other", "s") is None
    assert oauth_service.check_oauth_state_cookie("s", "s") == ""
    assert oauth_service.check_oauth_state_cookie("s.v", "s") == "v"


@pytest.mark.parametrize("provider", ["github", "google"])
def test_callback_on_another_replica_succeeds(env: dict[str, Any], provider: str) -> None:
    """DOCBOT-4017: the callback may land on a process that never issued the state."""
    client = TestClient(idx.app)
    state, _ = _start(client, provider)
    getattr(oauth_service, "_pending_states", {}).clear()  # other replica / post-redeploy process
    resp = client.get(f"/api/auth/{provider}/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert "auth_success=1" in resp.headers["location"]
    env[provider].assert_awaited_once()
