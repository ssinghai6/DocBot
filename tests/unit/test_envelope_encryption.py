"""Unit tests for per-org envelope encryption — DOCBOT-1529.

Covers: DEK generation/wrapping, new-scheme round trips, legacy-scheme
detection + decryption, the lazy read-triggered re-encryption path, and
tamper detection for both schemes.
"""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.utils.encryption import encrypt_credentials as legacy_encrypt_credentials
from api.utils.envelope_encryption import (
    register_org_encryption_keys_table,
    get_or_create_org_dek,
    encrypt_credentials_for_org,
    decrypt_credentials_with_migration,
    is_envelope_blob,
    write_back_blob,
    _wrap_dek,
    _unwrap_dek,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def org_keys_table_and_factory(tmp_path):
    """A real (file-backed) async SQLite engine with the org_encryption_keys
    table created, wired up the same way production wires it."""
    metadata = MetaData()
    table = register_org_encryption_keys_table(metadata)

    db_path = tmp_path / "test_envelope.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield table, session_factory, engine, metadata

    await engine.dispose()


# ---------------------------------------------------------------------------
# DEK wrap/unwrap (KEK-level)
# ---------------------------------------------------------------------------

class TestDekWrapUnwrap:
    def test_wrap_unwrap_round_trip(self):
        dek = Fernet.generate_key()
        wrapped = _wrap_dek(dek)
        assert isinstance(wrapped, str)
        assert dek not in wrapped.encode()  # raw key bytes not visible in the wrapped blob
        recovered = _unwrap_dek(wrapped)
        assert recovered == dek

    def test_unwrap_tampered_wrapped_dek_raises(self):
        dek = Fernet.generate_key()
        wrapped = _wrap_dek(dek)
        tampered = wrapped[:-5] + "XXXXX"
        with pytest.raises(InvalidToken):
            _unwrap_dek(tampered)


# ---------------------------------------------------------------------------
# Per-org DEK lifecycle
# ---------------------------------------------------------------------------

class TestOrgDekLifecycle:
    @pytest.mark.asyncio
    async def test_new_org_gets_fresh_dek_generated_and_wrapped(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory

        dek = await get_or_create_org_dek("org-alpha", factory, table)
        assert isinstance(dek, bytes)
        # A valid Fernet key is usable to encrypt/decrypt — proves it's real.
        Fernet(dek).decrypt(Fernet(dek).encrypt(b"probe"))

        # The persisted row holds the WRAPPED dek, never the raw key.
        async with factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(table.c.wrapped_dek).where(table.c.org_id == "org-alpha")
            )
            wrapped = result.scalar_one()
        assert dek not in wrapped.encode()
        assert _unwrap_dek(wrapped) == dek

    @pytest.mark.asyncio
    async def test_same_org_reuses_same_dek_on_second_call(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory

        dek1 = await get_or_create_org_dek("org-beta", factory, table)
        dek2 = await get_or_create_org_dek("org-beta", factory, table)
        assert dek1 == dek2

    @pytest.mark.asyncio
    async def test_different_orgs_get_different_deks(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory

        dek_a = await get_or_create_org_dek("org-a", factory, table)
        dek_b = await get_or_create_org_dek("org-b", factory, table)
        assert dek_a != dek_b


# ---------------------------------------------------------------------------
# New-scheme round trip
# ---------------------------------------------------------------------------

class TestEnvelopeSchemeRoundTrip:
    @pytest.mark.asyncio
    async def test_encrypt_for_org_round_trips(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        data = {"dialect": "postgresql", "host": "db.example.com", "password": "s3cr3t"}

        blob = await encrypt_credentials_for_org(data, "org-gamma", factory, table)
        assert is_envelope_blob(blob)
        assert blob.startswith("ENV1:org-gamma:")
        assert "s3cr3t" not in blob

        recovered, new_blob = await decrypt_credentials_with_migration(blob, "org-gamma", factory, table)
        assert recovered == data
        # Already on the new scheme — no rewrite needed.
        assert new_blob is None

    @pytest.mark.asyncio
    async def test_isolation_one_orgs_dek_does_not_decrypt_another_orgs_blob(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        data = {"password": "org-x-secret"}
        blob = await encrypt_credentials_for_org(data, "org-x", factory, table)

        # Swap the embedded org_id to a different (also-real) org whose DEK
        # differs — the ciphertext was produced with org-x's DEK, so
        # decrypting it as org-y must fail, not silently succeed.
        await get_or_create_org_dek("org-y", factory, table)  # ensure org-y has its own DEK
        _, _, token = blob.split(":", 2)
        forged_blob = f"ENV1:org-y:{token}"

        with pytest.raises(InvalidToken):
            await decrypt_credentials_with_migration(forged_blob, "org-y", factory, table)


# ---------------------------------------------------------------------------
# Legacy-scheme detection + decryption
# ---------------------------------------------------------------------------

class TestLegacySchemeDetection:
    def test_legacy_blob_is_not_detected_as_envelope(self):
        legacy_blob = legacy_encrypt_credentials({"password": "legacy-secret"})
        assert not is_envelope_blob(legacy_blob)

    def test_envelope_blob_is_detected(self):
        assert is_envelope_blob("ENV1:some-org:gAAAAA...")
        assert not is_envelope_blob("gAAAAABplainFernetToken")
        assert not is_envelope_blob(None)
        assert not is_envelope_blob(12345)

    @pytest.mark.asyncio
    async def test_legacy_blob_decrypts_via_legacy_path_when_org_unknown(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        data = {"dialect": "mysql", "password": "old-scheme-secret"}
        legacy_blob = legacy_encrypt_credentials(data)

        recovered, new_blob = await decrypt_credentials_with_migration(legacy_blob, None, factory, table)

        assert recovered == data
        # No org_id known yet — must NOT attempt migration.
        assert new_blob is None


# ---------------------------------------------------------------------------
# Lazy, read-triggered re-encryption (the core migration requirement)
# ---------------------------------------------------------------------------

class TestLazyMigration:
    @pytest.mark.asyncio
    async def test_legacy_blob_migrates_to_envelope_scheme_on_read(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        data = {"dialect": "postgresql", "host": "h", "password": "migrate-me"}
        legacy_blob = legacy_encrypt_credentials(data)
        assert not is_envelope_blob(legacy_blob)

        # Simulate a read where the row's org_id IS known (e.g. db_connections.org_id).
        recovered, new_blob = await decrypt_credentials_with_migration(
            legacy_blob, "org-delta", factory, table
        )

        assert recovered == data
        assert new_blob is not None
        assert is_envelope_blob(new_blob)
        assert new_blob.startswith("ENV1:org-delta:")

    @pytest.mark.asyncio
    async def test_migrated_blob_round_trips_correctly_on_second_read(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        data = {"dialect": "postgresql", "host": "h", "password": "migrate-me-twice"}
        legacy_blob = legacy_encrypt_credentials(data)

        # First read: migrates.
        recovered_1, new_blob = await decrypt_credentials_with_migration(
            legacy_blob, "org-epsilon", factory, table
        )
        assert recovered_1 == data
        assert new_blob is not None

        # Second read, against the migrated blob returned from the first —
        # proves the caller's "write new_blob back, then it reads correctly
        # forever after" contract actually holds.
        recovered_2, new_blob_2 = await decrypt_credentials_with_migration(
            new_blob, "org-epsilon", factory, table
        )
        assert recovered_2 == data
        assert new_blob_2 is None  # already on the new scheme, no further rewrite

    @pytest.mark.asyncio
    async def test_write_back_blob_persists_and_is_read_back_correctly(self, org_keys_table_and_factory):
        """End-to-end proof of the lazy migration contract: decrypt a legacy
        blob, write the migrated blob back over a real row, re-read that row
        from the DB, and confirm it now decrypts via the envelope path with
        no further migration offered."""
        table, factory, engine, _metadata = org_keys_table_and_factory

        # A minimal stand-in "credentials row" table to migrate in place,
        # mirroring db_connections/marketplace_connections shape. Created on
        # its own fresh MetaData (keeps it independent of the org_keys
        # table's metadata) but the same engine, so both live in one DB file.
        from sqlalchemy import Column, MetaData, String, Table, insert, select
        creds_metadata = MetaData()
        creds_table = Table(
            "fake_creds_row", creds_metadata,
            Column("id", String, primary_key=True),
            Column("blob", String, nullable=False),
        )
        async with engine.begin() as conn:
            await conn.run_sync(creds_metadata.create_all)

        data = {"password": "end-to-end-secret"}
        legacy_blob = legacy_encrypt_credentials(data)

        async with factory() as session:
            async with session.begin():
                await session.execute(insert(creds_table).values(id="row-1", blob=legacy_blob))

        # Read 1: decrypt (legacy) + migrate + write back.
        async with factory() as session:
            result = await session.execute(select(creds_table).where(creds_table.c.id == "row-1"))
            row = result.fetchone()
        recovered, new_blob = await decrypt_credentials_with_migration(row.blob, "org-zeta", factory, table)
        assert recovered == data
        assert new_blob is not None
        await write_back_blob(creds_table, "id", "row-1", "blob", new_blob, factory)

        # Read 2: the stored row is now envelope-encrypted and needs no further migration.
        async with factory() as session:
            result = await session.execute(select(creds_table).where(creds_table.c.id == "row-1"))
            row2 = result.fetchone()
        assert is_envelope_blob(row2.blob)
        recovered_2, new_blob_2 = await decrypt_credentials_with_migration(row2.blob, "org-zeta", factory, table)
        assert recovered_2 == data
        assert new_blob_2 is None


# ---------------------------------------------------------------------------
# Tamper detection — both schemes fail cleanly, never silently wrong
# ---------------------------------------------------------------------------

class TestTamperDetection:
    @pytest.mark.asyncio
    async def test_tampered_legacy_blob_raises_invalid_token(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        legacy_blob = legacy_encrypt_credentials({"password": "x"})
        tampered = legacy_blob[:-5] + "XXXXX"

        with pytest.raises(InvalidToken):
            await decrypt_credentials_with_migration(tampered, "org-eta", factory, table)

    @pytest.mark.asyncio
    async def test_tampered_envelope_blob_raises_invalid_token(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        blob = await encrypt_credentials_for_org({"password": "y"}, "org-theta", factory, table)
        prefix, org_id, token = blob.split(":", 2)
        tampered_token = token[:-5] + "XXXXX"
        tampered_blob = f"{prefix}:{org_id}:{tampered_token}"

        with pytest.raises(InvalidToken):
            await decrypt_credentials_with_migration(tampered_blob, "org-theta", factory, table)

    @pytest.mark.asyncio
    async def test_malformed_envelope_blob_raises_invalid_token(self, org_keys_table_and_factory):
        table, factory, _engine, _metadata = org_keys_table_and_factory
        # Missing the third (token) segment entirely.
        malformed = "ENV1:org-iota"

        with pytest.raises(InvalidToken):
            await decrypt_credentials_with_migration(malformed, "org-iota", factory, table)
