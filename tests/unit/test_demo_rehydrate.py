"""DOCBOT-2632: a demo session survives a restart (lazy rehydrate).

No network: the vector store is mocked and the app DB is in-memory SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.index as idx
from api import demo_service as d
from api.utils.caller_scope import CallerScope

OWNER = {"user_id": None, "org_id": None, "anon_owner": "anon-digest-x"}
SCOPE = CallerScope(user_id=None, org_id=None, anon_token="anon-digest-x", is_admin=False)


@pytest.fixture
async def app_db():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [idx.sessions_table, idx.db_connections_table, idx.schema_cache_table]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def chroma():
    """Fake Chroma: ``disk`` stands in for CHROMA_DIR, survives a 'restart'."""
    disk: dict = {}

    def create(sid, chunks, _emb):
        disk[sid] = ("store", len(chunks))
        return disk[sid]

    def load(sid, cache, _fn=None):
        if sid in cache:
            return cache[sid]
        if sid in disk:
            cache[sid] = disk[sid]
        return disk.get(sid)

    create_mock = MagicMock(side_effect=create)
    with patch("api.demo_service.create_store", new=create_mock), \
         patch("api.demo_service.get_or_load_store", new=load):
        yield disk, create_mock


async def _init(app_db, stores, dataset="tieout"):
    return await d.init_demo_session(
        vector_stores=stores,
        get_embeddings_fn=MagicMock(),
        sessions_table=idx.sessions_table,
        db_connections_table=idx.db_connections_table,
        schema_cache_table=idx.schema_cache_table,
        async_session_factory=app_db,
        dataset=dataset,
        owner=OWNER,
    )


async def _ensure(app_db, stores, session_id, connection_id=None):
    return await d.ensure_demo_session(
        session_id, connection_id,
        vector_stores=stores,
        get_embeddings_fn=MagicMock(),
        sessions_table=idx.sessions_table,
        db_connections_table=idx.db_connections_table,
        async_session_factory=app_db,
    )


async def _db_path(app_db, connection_id) -> Path:
    async with app_db() as s:
        row = (await s.execute(select(idx.db_connections_table.c.dbname).where(
            idx.db_connections_table.c.id == connection_id))).one()
    return Path(row.dbname)


async def test_nothing_missing_is_a_noop(app_db, chroma):
    stores: dict = {}
    r = await _init(app_db, stores)
    assert await _ensure(app_db, stores, r["session_id"], r["connection_id"]) is False
    (await _db_path(app_db, r["connection_id"])).unlink()


@pytest.mark.parametrize("dataset", ["tieout", "quickbite", "fuel"])
async def test_restart_rebuilds_store_and_sqlite_with_same_ids(app_db, chroma, dataset):
    disk, create_mock = chroma
    r = await _init(app_db, {}, dataset)
    sid, cid = r["session_id"], r["connection_id"]
    path = await _db_path(app_db, cid)

    # Restart: /tmp wiped, Chroma collection gone, process cache empty.
    path.unlink()
    disk.clear()
    stores: dict = {}
    create_mock.reset_mock()

    assert await _ensure(app_db, stores, sid, cid) is True
    assert stores[sid] == ("store", len(d._DATASETS[dataset]["chunks"]))
    create_mock.assert_called_once()
    assert create_mock.call_args.args[0] == sid
    assert path.exists()  # same path the row and creds blob point at
    with sqlite3.connect(str(path)) as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] > 0
    path.unlink()


async def test_missing_sqlite_only(app_db, chroma):
    _, create_mock = chroma
    stores: dict = {}
    r = await _init(app_db, stores)
    path = await _db_path(app_db, r["connection_id"])
    path.unlink()
    create_mock.reset_mock()

    # Connection-only call (DB chat with the "anonymous" placeholder session).
    assert await _ensure(app_db, stores, "anonymous", r["connection_id"]) is True
    assert path.exists()
    create_mock.assert_not_called()
    path.unlink()


async def test_missing_vector_store_only(app_db, chroma):
    disk, create_mock = chroma
    r = await _init(app_db, {})
    disk.clear()
    create_mock.reset_mock()
    stores: dict = {}
    assert await _ensure(app_db, stores, r["session_id"]) is True
    assert r["session_id"] in stores
    (await _db_path(app_db, r["connection_id"])).unlink()


async def test_missing_session_row_is_not_recreated(app_db, chroma):
    disk, create_mock = chroma
    r = await _init(app_db, {})
    path = await _db_path(app_db, r["connection_id"])
    path.unlink()
    disk.clear()
    async with app_db() as s, s.begin():
        await s.execute(delete(idx.sessions_table).where(idx.sessions_table.c.session_id == r["session_id"]))
    create_mock.reset_mock()

    assert await _ensure(app_db, {}, r["session_id"], r["connection_id"]) is False
    create_mock.assert_not_called()
    assert not path.exists()
    async with app_db() as s:
        assert (await s.execute(select(idx.sessions_table))).fetchall() == []


async def test_refuses_non_demo_and_unknown_ids(app_db, chroma):
    _, create_mock = chroma
    broken_factory = MagicMock(side_effect=AssertionError("no DB access for non-demo ids"))
    assert await d.ensure_demo_session(
        "0b8f-not-demo", "also-not-demo",
        vector_stores={}, get_embeddings_fn=MagicMock(),
        sessions_table=idx.sessions_table, db_connections_table=idx.db_connections_table,
        async_session_factory=broken_factory,
    ) is False
    # A caller-chosen demo-looking id with no server row: nothing is minted.
    assert await _ensure(app_db, {}, "demo-00000000-0000-0000-0000-000000000000",
                         "demo-11111111-1111-1111-1111-111111111111") is False
    create_mock.assert_not_called()


async def test_non_demo_source_row_with_demo_prefix_is_refused(app_db, chroma):
    _, create_mock = chroma
    async with app_db() as s, s.begin():
        await s.execute(idx.sessions_table.insert().values(
            session_id="demo-22222222-2222-2222-2222-222222222222", source="upload",
            files_info='[{"filename": "QuickBite-10K-2025.pdf"}]', **OWNER,
        ))
    assert await _ensure(app_db, {}, "demo-22222222-2222-2222-2222-222222222222") is False
    create_mock.assert_not_called()


async def test_require_session_rehydrates_then_404s_when_row_gone(app_db, chroma):
    disk, _ = chroma
    r = await _init(app_db, {})
    path = await _db_path(app_db, r["connection_id"])
    path.unlink()
    disk.clear()
    stores: dict = {}
    with patch.object(idx, "async_session_factory", app_db), \
         patch.object(idx, "VECTOR_STORES", stores), \
         patch.object(idx, "get_embeddings", MagicMock()):
        await idx._require_session(r["session_id"], SCOPE)
        assert r["session_id"] in stores and path.exists()

        # Another caller's scope still gets 404 and triggers no rebuild.
        path.unlink()
        other = CallerScope(user_id=None, org_id=None, anon_token="someone-else", is_admin=False)
        with pytest.raises(HTTPException) as exc:
            await idx._require_session(r["session_id"], other)
        assert exc.value.status_code == 404 and not path.exists()

        # Session row gone (e.g. deleted): 404, the frontend starts a fresh demo.
        async with app_db() as s, s.begin():
            await s.execute(delete(idx.sessions_table))
        with pytest.raises(HTTPException) as exc:
            await idx._require_session(r["session_id"], SCOPE)
        assert exc.value.status_code == 404
