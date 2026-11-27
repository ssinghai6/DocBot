"""DOCBOT-2598: self-serve test API keys (api/routes/developer_routes.py).

Real SQL against in-memory SQLite (api_keys table), a minimal FastAPI app
mounting the real router, and the session-cookie guard overridden with a
fake user row.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import api_key_service as keys
from api.routes import developer_routes


@pytest.fixture
async def store():
    metadata = MetaData()
    table = keys.register_api_keys_table(metadata)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    keys.wire_api_key_store(table, factory)
    yield table, factory
    await engine.dispose()


@pytest.fixture
def make_client(store):
    def _make(user_id: str = "u1", verified: bool = True) -> TestClient:
        app = FastAPI()
        app.include_router(developer_routes.router)
        user = SimpleNamespace(id=user_id, org_id="org-1", role="viewer", email_verified=verified)
        app.dependency_overrides[developer_routes._signed_in.dependency] = lambda: user
        return TestClient(app)
    return _make


def _fake_request():
    from unittest.mock import MagicMock
    request = MagicMock()
    request.method = "POST"
    request.url.path = "/v1/hybrid/chat"
    request.headers = {}
    request.client = None
    request.state = MagicMock()
    return request


@pytest.mark.integration
class TestDeveloperTestKeys:
    def test_mint_returns_test_key_once_with_preset_scopes_and_30_day_expiry(self, make_client, store):
        client = make_client()
        resp = client.post("/api/developer/test-key")
        assert resp.status_code == 201
        body = resp.json()
        assert body["key"].startswith("dk_test_")
        assert body["mode"] == "test"
        assert body["scopes"] == keys.TEST_KEY_SCOPES
        assert body["prefix"] == body["key"][: len(body["prefix"])]
        expires = datetime.fromisoformat(body["expires_at"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        assert abs(expires - (datetime.now(timezone.utc) + timedelta(days=30))) < timedelta(minutes=1)

        listing = client.get("/api/developer/test-keys").json()["keys"]
        assert [k["id"] for k in listing] == [body["id"]]
        assert "key" not in listing[0]
        assert set(listing[0]) == {"id", "prefix", "scopes", "created_at", "expires_at", "last_used_at"}

    async def test_raw_key_never_stored(self, make_client, store):
        table, factory = store
        raw = make_client().post("/api/developer/test-key").json()["key"]
        async with factory() as session:
            row = (await session.execute(select(table))).first()
        assert raw not in row
        assert row.mode == "test" and row.created_by == "u1"

    def test_unverified_email_is_403(self, make_client):
        resp = make_client(verified=False).post("/api/developer/test-key")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "email_not_verified"

    def test_third_active_key_is_409_until_one_is_revoked(self, make_client):
        client = make_client()
        first = client.post("/api/developer/test-key").json()
        assert client.post("/api/developer/test-key").status_code == 201
        third = client.post("/api/developer/test-key")
        assert third.status_code == 409
        assert third.json()["detail"]["code"] == "test_key_limit"

        assert client.delete(f"/api/developer/test-key/{first['id']}").status_code == 200
        assert client.post("/api/developer/test-key").status_code == 201

    def test_limit_is_per_user(self, make_client):
        a, b = make_client("ua"), make_client("ub")
        for _ in range(2):
            assert a.post("/api/developer/test-key").status_code == 201
        assert b.post("/api/developer/test-key").status_code == 201

    async def test_revoked_key_is_401_on_next_use(self, make_client):
        client = make_client()
        minted = client.post("/api/developer/test-key").json()
        dep = keys.require_api_key("query:run")
        assert (await dep(request=_fake_request(), authorization=f"Bearer {minted['key']}")).id == minted["id"]

        assert client.delete(f"/api/developer/test-key/{minted['id']}").json() == {
            "status": "revoked", "id": minted["id"],
        }
        with pytest.raises(HTTPException) as exc_info:
            await dep(request=_fake_request(), authorization=f"Bearer {minted['key']}")
        assert exc_info.value.status_code == 401
        assert client.get("/api/developer/test-keys").json()["keys"] == []

    def test_cannot_revoke_another_users_key(self, make_client):
        minted = make_client("owner").post("/api/developer/test-key").json()
        resp = make_client("intruder").delete(f"/api/developer/test-key/{minted['id']}")
        assert resp.status_code == 404
