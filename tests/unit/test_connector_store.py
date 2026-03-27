"""Tests for api/connector_store.py — DOCBOT-706.

Verifies connector persistence: table registration, credential encryption
round-trip, save/load/delete lifecycle, and last_sync_at tracking.
"""

import asyncio
import os

import pytest
from sqlalchemy import MetaData, create_engine, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.connector_store import (
    delete_connector,
    get_connector_info,
    list_active_connectors,
    load_all_active_connectors,
    register_connector_tables,
    save_connector,
    update_last_sync,
    wire_connector_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _metadata():
    return MetaData()


@pytest.fixture
def connections_table(_metadata):
    return register_connector_tables(_metadata)


@pytest.fixture
async def wired_store(tmp_path, connections_table, _metadata):
    """Set up an async SQLite engine, create tables, and wire the store."""
    db_path = tmp_path / "test_connectors.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async with engine.begin() as conn:
        await conn.run_sync(_metadata.create_all)

    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    wire_connector_store(connections_table, session_factory)

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "table": connections_table,
    }

    await engine.dispose()
    # Reset module-level refs
    wire_connector_store(None, None)


# ---------------------------------------------------------------------------
# Table Registration
# ---------------------------------------------------------------------------

class TestTableRegistration:
    def test_table_name(self, connections_table):
        assert connections_table.name == "marketplace_connections"

    def test_columns_exist(self, connections_table):
        col_names = {c.name for c in connections_table.columns}
        expected = {
            "id", "connector_type", "encrypted_credentials",
            "is_active", "last_sync_at", "created_at", "updated_at",
        }
        assert expected.issubset(col_names)

    def test_primary_key(self, connections_table):
        pk_cols = [c.name for c in connections_table.primary_key.columns]
        assert pk_cols == ["id"]


# ---------------------------------------------------------------------------
# Credential Encryption Round-Trip
# ---------------------------------------------------------------------------

class TestCredentialEncryption:
    @pytest.mark.asyncio
    async def test_save_encrypts_credentials(self, wired_store):
        """Raw DB row must NOT contain plaintext credentials."""
        ctx = wired_store
        creds = {"client_id": "test123", "client_secret": "supersecret"}
        await save_connector("enc-1", "amazon", creds)

        async with ctx["session_factory"]() as session:
            result = await session.execute(
                select(ctx["table"].c.encrypted_credentials)
                .where(ctx["table"].c.id == "enc-1")
            )
            encrypted = result.scalar_one()

        assert "supersecret" not in encrypted
        assert "test123" not in encrypted

    @pytest.mark.asyncio
    async def test_load_decrypts_credentials(self, wired_store):
        """load_all_active_connectors should return original plaintext creds."""
        creds = {"client_id": "abc", "refresh_token": "tok-456"}
        await save_connector("enc-2", "amazon", creds)

        loaded = await load_all_active_connectors()
        match = [c for c in loaded if c["connector_id"] == "enc-2"]
        assert len(match) == 1
        assert match[0]["credentials"] == creds

    @pytest.mark.asyncio
    async def test_info_excludes_credentials(self, wired_store):
        """get_connector_info must never return credentials."""
        await save_connector("enc-3", "amazon", {"secret": "hidden"})
        info = await get_connector_info("enc-3")
        assert info is not None
        assert "credentials" not in info
        assert "encrypted_credentials" not in info
        assert "secret" not in str(info)


# ---------------------------------------------------------------------------
# Save / Load / Delete Lifecycle
# ---------------------------------------------------------------------------

class TestSaveLoadDelete:
    @pytest.mark.asyncio
    async def test_save_and_load(self, wired_store):
        await save_connector("sld-1", "amazon", {"key": "val"})
        loaded = await load_all_active_connectors()
        ids = [c["connector_id"] for c in loaded]
        assert "sld-1" in ids

    @pytest.mark.asyncio
    async def test_delete_soft_deletes(self, wired_store):
        """Deleted connector should not appear in load or list."""
        await save_connector("sld-2", "amazon", {"key": "val"})
        result = await delete_connector("sld-2")
        assert result is True

        loaded = await load_all_active_connectors()
        ids = [c["connector_id"] for c in loaded]
        assert "sld-2" not in ids

        listed = await list_active_connectors()
        listed_ids = [c["connector_id"] for c in listed]
        assert "sld-2" not in listed_ids

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, wired_store):
        result = await delete_connector("does-not-exist")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_excludes_deleted(self, wired_store):
        await save_connector("sld-3a", "amazon", {"k": "1"})
        await save_connector("sld-3b", "amazon", {"k": "2"})
        await delete_connector("sld-3a")

        listed = await list_active_connectors()
        listed_ids = [c["connector_id"] for c in listed]
        assert "sld-3a" not in listed_ids
        assert "sld-3b" in listed_ids

    @pytest.mark.asyncio
    async def test_save_upsert_updates(self, wired_store):
        """Saving same ID twice should update, not duplicate."""
        await save_connector("sld-4", "amazon", {"version": "1"})
        await save_connector("sld-4", "amazon", {"version": "2"})

        loaded = await load_all_active_connectors()
        match = [c for c in loaded if c["connector_id"] == "sld-4"]
        assert len(match) == 1
        assert match[0]["credentials"] == {"version": "2"}

    @pytest.mark.asyncio
    async def test_reactivate_after_delete(self, wired_store):
        """Re-saving a soft-deleted connector should reactivate it."""
        await save_connector("sld-5", "amazon", {"k": "v"})
        await delete_connector("sld-5")
        await save_connector("sld-5", "amazon", {"k": "v2"})

        loaded = await load_all_active_connectors()
        match = [c for c in loaded if c["connector_id"] == "sld-5"]
        assert len(match) == 1
        assert match[0]["credentials"] == {"k": "v2"}


# ---------------------------------------------------------------------------
# last_sync_at Tracking
# ---------------------------------------------------------------------------

class TestUpdateLastSync:
    @pytest.mark.asyncio
    async def test_updates_timestamp(self, wired_store):
        await save_connector("sync-1", "amazon", {"k": "v"})

        info_before = await get_connector_info("sync-1")
        assert info_before is not None
        assert info_before["last_sync_at"] is None

        await update_last_sync("sync-1")

        info_after = await get_connector_info("sync-1")
        assert info_after is not None
        assert info_after["last_sync_at"] is not None


# ---------------------------------------------------------------------------
# List Active Connectors Metadata
# ---------------------------------------------------------------------------

class TestListActiveConnectors:
    @pytest.mark.asyncio
    async def test_returns_metadata_fields(self, wired_store):
        await save_connector("meta-1", "amazon", {"k": "v"})
        listed = await list_active_connectors()
        match = [c for c in listed if c["connector_id"] == "meta-1"]
        assert len(match) == 1
        entry = match[0]
        assert "connector_type" in entry
        assert "is_active" in entry
        assert "created_at" in entry
        assert "updated_at" in entry
        assert "last_sync_at" in entry
        # Must NOT contain credentials
        assert "credentials" not in entry
        assert "encrypted_credentials" not in entry
