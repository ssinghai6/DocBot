"""DOCBOT-2597: sandbox sessions for test API keys.

Real SQL against in-memory SQLite using the app's own sessions /
db_connections / schema_cache / answer_lineage tables; ``init_demo_session``
is replaced by a fake that writes the same rows (no embeddings), and the
rate-limit counters use the in-memory store.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import MetaData, insert, select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

import api.index as idx  # noqa: E402
from api.routes import v1_routes  # noqa: E402
from api import api_key_service as keys  # noqa: E402
from api import credit_service  # noqa: E402
from api import sandbox_session_service as sandbox  # noqa: E402
from api.utils import request_guard  # noqa: E402

TEST_KEY = SimpleNamespace(id="k-test", org_id="org-1", created_by="u1", mode="test")
OTHER_TEST_KEY = SimpleNamespace(id="k-other", org_id="org-1", created_by="u2", mode="test")
LIVE_KEY = SimpleNamespace(id="k-live", org_id="org-1", created_by=None, mode="live")


@pytest.fixture
async def db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    tables = [idx.sessions_table, idx.db_connections_table, idx.schema_cache_table, idx.answer_lineage_table]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    vector_stores: dict = {}
    sandbox.wire_sandbox_store(
        idx.sessions_table, idx.db_connections_table, idx.schema_cache_table,
        factory, vector_stores, lambda: None,
    )
    request_guard.use_counter_store(request_guard.MemoryCounterStore())

    async def fake_init_demo_session(**kw):
        n = len(vector_stores)
        sid, cid = f"demo-s{n}-{kw['owner']['user_id']}", f"demo-c{n}-{kw['owner']['user_id']}"
        vector_stores[sid] = object()
        async with factory() as s, s.begin():
            await s.execute(insert(idx.sessions_table).values(session_id=sid, source="demo", **kw["owner"]))
            await s.execute(insert(idx.db_connections_table).values(
                id=cid, session_id=sid, dialect="sqlite", host="__local_file__", port=0,
                dbname="/nonexistent.db", credentials_blob="x",
                user_id=kw["owner"]["user_id"], org_id=kw["owner"]["org_id"],
            ))
        return {"session_id": sid, "connection_id": cid}

    import api.demo_service as demo_service
    init_mock = AsyncMock(side_effect=fake_init_demo_session)
    monkeypatch.setattr(demo_service, "init_demo_session", init_mock)
    yield SimpleNamespace(factory=factory, vector_stores=vector_stores, init=init_mock)
    request_guard.use_counter_store(request_guard.MemoryCounterStore())
    await engine.dispose()


async def _add_session(db, sid, *, user_id="u1", sandbox_flag=False, created_at=None):
    values = dict(session_id=sid, user_id=user_id, org_id="org-1", is_sandbox=sandbox_flag)
    if created_at is not None:
        values["created_at"] = created_at
    async with db.factory() as s, s.begin():
        await s.execute(insert(idx.sessions_table).values(**values))


async def _add_connection(db, cid, sid, dbname="/nonexistent.db"):
    async with db.factory() as s, s.begin():
        await s.execute(insert(idx.db_connections_table).values(
            id=cid, session_id=sid, dialect="sqlite", host="__local_file__", port=0,
            dbname=dbname, credentials_blob="x", user_id="u1", org_id="org-1",
        ))


def _assert_404(exc_info):
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# AC1: create a sandbox session it can query
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCreate:
    async def test_create_returns_ids_the_test_key_can_query(self, db):
        out = await sandbox.create_sandbox_session(TEST_KEY, "quickbite")
        assert set(out) == {"session_id", "connection_id", "dataset", "expires_at"}
        assert out["dataset"] == "quickbite"
        assert out["expires_at"] - datetime.now(timezone.utc) > timedelta(hours=23)
        owner = db.init.call_args.kwargs["owner"]
        assert owner == {"user_id": "u1", "org_id": "org-1", "is_sandbox": True}
        assert db.init.call_args.kwargs["dataset"] == "quickbite"
        # The isolation check and the existing org check both admit it.
        await sandbox.enforce_test_key_scope(TEST_KEY, out["session_id"], out["connection_id"])
        from api.resource_scope_service import session_in_org
        assert await session_in_org(out["session_id"], "org-1", idx.sessions_table, db.factory)

    async def test_unsupported_dataset_is_400(self, db):
        with pytest.raises(HTTPException) as exc_info:
            await sandbox.create_sandbox_session(TEST_KEY, "payroll")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "unsupported_dataset"

    async def test_sixth_create_in_an_hour_is_429(self, db):
        for _ in range(5):
            await sandbox.create_sandbox_session(TEST_KEY, "quickbite")
        with pytest.raises(HTTPException) as exc_info:
            await sandbox.create_sandbox_session(TEST_KEY, "quickbite")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["code"] == "sandbox_rate_limited"
        assert "Retry-After" in exc_info.value.headers
        assert db.init.await_count == 5
        # Another owner has its own window.
        await sandbox.create_sandbox_session(OTHER_TEST_KEY, "quickbite")

    async def test_route_returns_201_for_a_test_key(self, db):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        md = MetaData()
        table = keys.register_api_keys_table(md)
        async with engine.begin() as conn:
            await conn.run_sync(md.create_all)
        keys.wire_api_key_store(table, async_sessionmaker(engine, expire_on_commit=False))
        _, raw = await keys.create_api_key(
            org_id="org-1", scopes=list(keys.TEST_KEY_SCOPES), name="t", mode="test", created_by="u1",
        )
        from api.routes.v1_sandbox_routes import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.post("/v1/sandbox/sessions", json={"dataset": "quickbite"},
                           headers={"Authorization": f"Bearer {raw}"})
        assert resp.status_code == 201, resp.text
        assert resp.json()["session_id"].startswith("demo-")
        assert client.post("/v1/sandbox/sessions", json={}).status_code == 401
        await engine.dispose()


# ---------------------------------------------------------------------------
# AC2: test keys get 404 on anything that is not their own sandbox
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIsolation:
    async def test_non_sandbox_session_is_404(self, db):
        await _add_session(db, "live-s1", sandbox_flag=False)
        with pytest.raises(HTTPException) as exc_info:
            await sandbox.enforce_test_key_scope(TEST_KEY, "live-s1", None)
        _assert_404(exc_info)

    async def test_other_users_sandbox_is_404(self, db):
        await _add_session(db, "sb-u2", user_id="u2", sandbox_flag=True)
        with pytest.raises(HTTPException) as exc_info:
            await sandbox.enforce_test_key_scope(TEST_KEY, "sb-u2", None)
        _assert_404(exc_info)

    async def test_non_sandbox_connection_is_404_even_with_own_sandbox_session(self, db):
        await _add_session(db, "sb-u1", sandbox_flag=True)
        await _add_session(db, "live-s1", sandbox_flag=False)
        await _add_connection(db, "live-c1", "live-s1")
        with pytest.raises(HTTPException) as exc_info:
            await sandbox.enforce_test_key_scope(TEST_KEY, "sb-u1", "live-c1")
        _assert_404(exc_info)
        with pytest.raises(HTTPException):
            await sandbox.enforce_test_key_scope(TEST_KEY, "sb-u1", "missing-conn")

    async def test_live_key_is_unaffected(self, db):
        await _add_session(db, "live-s1", sandbox_flag=False)
        await sandbox.enforce_test_key_scope(LIVE_KEY, "live-s1", None)

    async def test_lineage_of_non_sandbox_run_is_404(self, db):
        await _add_session(db, "live-s1", sandbox_flag=False)
        await _add_session(db, "sb-u1", sandbox_flag=True)
        async with db.factory() as s, s.begin():
            for run_id, sid in (("run-live", "live-s1"), ("run-sb", "sb-u1")):
                await s.execute(insert(idx.answer_lineage_table).values(
                    run_id=run_id, session_id=sid, mode="hybrid", data_json="{}",
                ))
        with pytest.raises(HTTPException) as exc_info:
            await sandbox.enforce_test_key_run_scope(TEST_KEY, idx.answer_lineage_table, "run-live")
        _assert_404(exc_info)
        await sandbox.enforce_test_key_run_scope(TEST_KEY, idx.answer_lineage_table, "run-sb")

    async def test_index_v1_resource_hook_calls_enforcement(self, db):
        await _add_session(db, "live-s1", sandbox_flag=False)
        with pytest.raises(HTTPException) as exc_info:
            await v1_routes._require_v1_resources(TEST_KEY, "live-s1", None)
        _assert_404(exc_info)


# ---------------------------------------------------------------------------
# AC3: daily test allowance, org balance untouched
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAllowance:
    async def test_exhausted_allowance_is_429_and_org_balance_untouched(self, db, monkeypatch):
        monkeypatch.setenv("TEST_KEY_DAILY_CREDITS", "10")
        reserve, settle = AsyncMock(), AsyncMock()
        monkeypatch.setattr(credit_service, "reserve", reserve)
        monkeypatch.setattr(credit_service, "settle", settle)
        begin_billing = AsyncMock()
        monkeypatch.setattr(credit_service, "begin_billing", begin_billing)

        run = await v1_routes._begin_v1_billing(None, "hybrid", TEST_KEY, "demo-x", None)
        assert isinstance(run, sandbox.TestAllowanceRun)
        assert run.org_id is None and run.exempt
        async with run:
            run.meter.record_sandbox(1.0, "0.05")  # 50_000 micro-USD -> > 10 credits
        with pytest.raises(HTTPException) as exc_info:
            await v1_routes._begin_v1_billing(None, "hybrid", TEST_KEY, "demo-x", None)
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["code"] == "test_allowance_exhausted"

        begin_billing.assert_not_awaited()
        reserve.assert_not_awaited()
        settle.assert_not_awaited()
        # Allowance is per owner.
        await sandbox.begin_test_key_billing("hybrid", OTHER_TEST_KEY)

    async def test_live_key_still_uses_org_billing(self, db, monkeypatch):
        begin_billing = AsyncMock(return_value="org-run")
        monkeypatch.setattr(credit_service, "begin_billing", begin_billing)
        assert await v1_routes._begin_v1_billing(None, "hybrid", LIVE_KEY, "s", None) == "org-run"
        begin_billing.assert_awaited_once()


# ---------------------------------------------------------------------------
# AC5: 24h purge
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_purge_removes_old_sandbox_session_store_and_sqlite_file(db, tmp_path, monkeypatch):
    import api.utils.vector_store as vector_store
    deleted: list[str] = []
    monkeypatch.setattr(vector_store, "delete_store", deleted.append)

    old = datetime.now(timezone.utc) - timedelta(hours=25)
    sqlite_file = tmp_path / "sandbox.db"
    sqlite_file.write_bytes(b"")
    await _add_session(db, "sb-old", sandbox_flag=True, created_at=old)
    await _add_connection(db, "sb-old-c", "sb-old", dbname=str(sqlite_file))
    async with db.factory() as s, s.begin():
        await s.execute(insert(idx.schema_cache_table).values(
            connection_id="sb-old-c", schema_json="[]", expires_at=datetime.now(timezone.utc),
        ))
    await _add_session(db, "sb-new", sandbox_flag=True)
    await _add_session(db, "live-old", sandbox_flag=False, created_at=old)
    db.vector_stores.update({"sb-old": object(), "sb-new": object()})

    assert await sandbox.purge_expired_sandbox_sessions() == 1

    assert not sqlite_file.exists()
    assert deleted == ["sb-old"]
    assert "sb-old" not in db.vector_stores and "sb-new" in db.vector_stores
    async with db.factory() as s:
        sessions = {r.session_id for r in (await s.execute(select(idx.sessions_table.c.session_id))).all()}
        conns = (await s.execute(select(idx.db_connections_table))).all()
        cache = (await s.execute(select(idx.schema_cache_table))).all()
    assert sessions == {"sb-new", "live-old"}
    assert conns == [] and cache == []


def test_purge_job_registered_hourly():
    from unittest.mock import MagicMock
    scheduler = MagicMock()
    sandbox.register_sandbox_purge_job(scheduler)
    args, kwargs = scheduler.add_job.call_args
    assert args[1] == "interval" and kwargs["hours"] == 1 and kwargs["id"] == "sandbox_session_purge"


# ---------------------------------------------------------------------------
# /v1 error envelope (DOCBOT-3901 contract) for this PR's error codes
# ---------------------------------------------------------------------------


def _envelope_client(monkeypatch):
    monkeypatch.setattr(request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    request_guard.use_counter_store(request_guard.MemoryCounterStore())
    return TestClient(idx.app, raise_server_exceptions=False)


def _assert_envelope(resp, status, type_, code):
    assert resp.status_code == status, resp.text
    err = resp.json()["error"]
    assert set(err) == {"type", "code", "message", "request_id"}
    assert (err["type"], err["code"]) == (type_, code)
    assert err["request_id"] == resp.headers["X-Request-Id"]


def test_expired_key_uses_v1_envelope(monkeypatch):
    client = _envelope_client(monkeypatch)
    expired = keys.APIKeyRecord(id="k", org_id="org-1", key_prefix="dk_test_x", name="t", mode="test")
    monkeypatch.setattr(keys, "validate_api_key", AsyncMock(side_effect=keys.ExpiredAPIKey(expired)))
    monkeypatch.setattr(keys, "_audit", lambda *a: None)
    resp = client.post("/v1/sandbox/sessions", json={}, headers={"Authorization": "Bearer dk_test_x"})
    _assert_envelope(resp, 401, "authentication", "expired_key")


@pytest.mark.parametrize("code", ["sandbox_rate_limited", "test_allowance_exhausted"])
def test_429_codes_use_v1_envelope(monkeypatch, code):
    from api.routes import v1_sandbox_routes
    client = _envelope_client(monkeypatch)
    record = keys.APIKeyRecord(
        id="k", org_id="org-1", key_prefix="dk_test_x", name="t",
        scopes=["sessions:write"], mode="test", created_by="u1",
    )
    monkeypatch.setattr(keys, "validate_api_key", AsyncMock(return_value=record))
    monkeypatch.setattr(v1_sandbox_routes, "create_sandbox_session", AsyncMock(side_effect=HTTPException(
        status_code=429, detail={"code": code, "message": "slow down"}, headers={"Retry-After": "60"},
    )))
    resp = client.post("/v1/sandbox/sessions", json={}, headers={"Authorization": "Bearer dk_test_x"})
    _assert_envelope(resp, 429, "rate_limit", code)
    assert resp.headers["Retry-After"] == "60"
