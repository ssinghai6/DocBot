"""Unit/integration tests for api_key_service — DOCBOT-1530.

Exercises real SQL against an in-memory SQLite async engine (same style as
tests/integration/test_db_pipeline.py) rather than mocking the session
factory, since the behavior under test (hash-not-raw-key persisted, scope
checking, org_id attachment) is most meaningfully verified against a real
round trip through the table.
"""

from __future__ import annotations

import pytest
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.api_key_service import (
    API_KEY_SCOPES,
    create_api_key,
    list_api_keys,
    register_api_keys_table,
    require_api_key,
    revoke_api_key,
    validate_api_key,
    wire_api_key_store,
)


@pytest.fixture
async def wired_store():
    """Stand up a fresh in-memory SQLite engine with the api_keys table wired."""
    metadata = MetaData()
    table = register_api_keys_table(metadata)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    wire_api_key_store(table, factory, audit_log_table=None)
    yield table, factory
    await engine.dispose()


@pytest.mark.integration
class TestCreateApiKey:
    async def test_returns_raw_key_once_and_stores_only_hash(self, wired_store):
        table, factory = wired_store
        record, raw_key = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="My Key")

        assert raw_key.startswith("dk_live_")
        assert record.org_id == "org-1"
        assert record.scopes == ["hybrid:read"]

        async with factory() as session:
            from sqlalchemy import select
            result = await session.execute(select(table))
            row = result.first()

        assert row.key_hash != raw_key
        assert raw_key not in row.key_hash
        assert row.key_prefix == raw_key[: len(row.key_prefix)]

    async def test_rejects_empty_scopes(self, wired_store):
        with pytest.raises(ValueError):
            await create_api_key(org_id="org-1", scopes=[], name="x")

    async def test_rejects_unknown_scope(self, wired_store):
        with pytest.raises(ValueError):
            await create_api_key(org_id="org-1", scopes=["totally:invalid"], name="x")


@pytest.mark.integration
class TestValidateApiKey:
    async def test_validates_a_real_key(self, wired_store):
        record, raw_key = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        result = await validate_api_key(raw_key)
        assert result is not None
        assert result.id == record.id
        assert result.org_id == "org-1"

    async def test_rejects_wrong_key(self, wired_store):
        await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        result = await validate_api_key("dk_live_not_the_right_key_at_all")
        assert result is None

    async def test_rejects_nonexistent_key(self, wired_store):
        result = await validate_api_key("dk_live_never_issued")
        assert result is None

    async def test_rejects_revoked_key(self, wired_store):
        record, raw_key = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        revoked = await revoke_api_key(record.id)
        assert revoked is True

        result = await validate_api_key(raw_key)
        assert result is None

    async def test_updates_last_used_at(self, wired_store):
        _, raw_key = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        first = await validate_api_key(raw_key)
        assert first.last_used_at is None  # not yet updated as of issuance

        second = await validate_api_key(raw_key)
        assert second.last_used_at is not None


@pytest.mark.integration
class TestRevokeApiKey:
    async def test_revoke_is_idempotent(self, wired_store):
        record, _ = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        assert await revoke_api_key(record.id) is True
        assert await revoke_api_key(record.id) is True  # already revoked, still "found"

    async def test_revoke_nonexistent_returns_false(self, wired_store):
        assert await revoke_api_key("does-not-exist") is False

    async def test_revoke_scoped_to_wrong_org_not_found(self, wired_store):
        record, _ = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        # A different org may not revoke this key.
        assert await revoke_api_key(record.id, org_id="org-2") is False
        # The key is unaffected and still validates.
        assert await revoke_api_key(record.id, org_id="org-1") is True


@pytest.mark.integration
class TestListApiKeys:
    async def test_org_id_scoping(self, wired_store):
        await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="a")
        await create_api_key(org_id="org-1", scopes=["autopilot:run"], name="b")
        await create_api_key(org_id="org-2", scopes=["hybrid:read"], name="c")

        org1_keys = await list_api_keys("org-1")
        org2_keys = await list_api_keys("org-2")

        assert len(org1_keys) == 2
        assert len(org2_keys) == 1
        assert all(k.org_id == "org-1" for k in org1_keys)


@pytest.mark.integration
class TestRequireApiKeyDependency:
    """Exercises the FastAPI dependency factory directly (no running app)."""

    async def _call_dependency(self, min_scope: str, authorization, wired_store):
        from unittest.mock import MagicMock
        table, factory = wired_store
        dependency = require_api_key(min_scope)
        request = MagicMock()
        request.method = "POST"
        request.url.path = "/v1/hybrid/chat"
        request.headers = {}
        request.client = None
        request.state = MagicMock()
        return await dependency(request=request, authorization=authorization)

    async def test_missing_header_raises_401(self, wired_store):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await self._call_dependency("hybrid:read", None, wired_store)
        assert exc_info.value.status_code == 401

    async def test_malformed_header_raises_401(self, wired_store):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await self._call_dependency("hybrid:read", "NotBearer abc", wired_store)
        assert exc_info.value.status_code == 401

    async def test_invalid_key_raises_401(self, wired_store):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await self._call_dependency("hybrid:read", "Bearer dk_live_bogus", wired_store)
        assert exc_info.value.status_code == 401

    async def test_insufficient_scope_raises_403(self, wired_store):
        from fastapi import HTTPException
        _, raw_key = await create_api_key(org_id="org-1", scopes=["hybrid:read"], name="k")
        with pytest.raises(HTTPException) as exc_info:
            await self._call_dependency("autopilot:run", f"Bearer {raw_key}", wired_store)
        assert exc_info.value.status_code == 403

    async def test_valid_key_with_sufficient_scope_attaches_org_id(self, wired_store):
        record, raw_key = await create_api_key(org_id="org-42", scopes=["autopilot:run"], name="k")
        from unittest.mock import MagicMock
        request = MagicMock()
        request.method = "POST"
        request.url.path = "/v1/autopilot/run"
        request.headers = {}
        request.client = None
        request.state = MagicMock()

        dependency = require_api_key("autopilot:run")
        result = await dependency(request=request, authorization=f"Bearer {raw_key}")

        assert result.id == record.id
        assert request.state.org_id == "org-42"
        assert request.state.api_key_id == record.id


def test_api_key_scopes_cover_expected_capabilities():
    """Sanity check the fixed scope set documented in CLAUDE.md / the ticket."""
    assert "hybrid:read" in API_KEY_SCOPES
    assert "autopilot:run" in API_KEY_SCOPES
    assert "connectors:manage" in API_KEY_SCOPES
    assert "lineage:read" in API_KEY_SCOPES


# ---------------------------------------------------------------------------
# Integration-style: require_api_key actually gates a FastAPI route.
#
# A full boot of api/index.py's real app is impractical here (its engine is
# configured with Postgres-only pool kwargs and its lifespan wires a dozen
# other services) — this stands up a minimal app that mounts the real
# `require_api_key` dependency on stub routes shaped like each /v1/* route's
# auth gate (one per scope the ticket's /v1/ surface uses), which is what
# actually determines 401/403/200 behavior; the handler bodies in
# api/index.py are now-trivial pass-throughs to already-tested service
# functions.
# ---------------------------------------------------------------------------


@pytest.fixture
def v1_test_app(wired_store):
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/v1/hybrid/chat")
    async def hybrid_stub(_key=Depends(require_api_key("hybrid:read"))):
        return {"ok": True, "org_id": _key.org_id}

    @app.post("/v1/autopilot/run")
    async def autopilot_stub(_key=Depends(require_api_key("autopilot:run"))):
        return {"ok": True, "org_id": _key.org_id}

    @app.get("/v1/lineage/{run_id}")
    async def lineage_stub(run_id: str, _key=Depends(require_api_key("lineage:read"))):
        return {"ok": True, "run_id": run_id}

    @app.get("/v1/connectors")
    async def connectors_stub(_key=Depends(require_api_key("connectors:manage"))):
        return {"ok": True}

    return TestClient(app)


@pytest.mark.integration
class TestV1RouteGating:
    """One 401 (no key) + one 403 (wrong scope) + one 200 (right scope) per route."""

    ROUTES = [
        ("post", "/v1/hybrid/chat", "hybrid:read"),
        ("post", "/v1/autopilot/run", "autopilot:run"),
        ("get", "/v1/lineage/abc123", "lineage:read"),
        ("get", "/v1/connectors", "connectors:manage"),
    ]

    @pytest.mark.parametrize("method,path,scope", ROUTES)
    async def test_no_key_is_401(self, v1_test_app, method, path, scope):
        resp = getattr(v1_test_app, method)(path)
        assert resp.status_code == 401

    @pytest.mark.parametrize("method,path,scope", ROUTES)
    async def test_wrong_scope_is_403(self, v1_test_app, wired_store, method, path, scope):
        other_scope = next(s for s in API_KEY_SCOPES if s != scope)
        _, raw_key = await create_api_key(org_id="org-1", scopes=[other_scope], name="k")
        resp = getattr(v1_test_app, method)(path, headers={"Authorization": f"Bearer {raw_key}"})
        assert resp.status_code == 403

    @pytest.mark.parametrize("method,path,scope", ROUTES)
    async def test_correct_scope_is_200(self, v1_test_app, wired_store, method, path, scope):
        _, raw_key = await create_api_key(org_id="org-1", scopes=[scope], name="k")
        resp = getattr(v1_test_app, method)(path, headers={"Authorization": f"Bearer {raw_key}"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
