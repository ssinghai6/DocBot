"""Per-org envelope encryption for stored credentials — DOCBOT-1529.

Background
----------
Before this ticket, every stored credential blob (DB connection strings,
marketplace connector OAuth tokens) was encrypted with ONE flat Fernet key
derived from the `DB_ENCRYPTION_KEY` env var, shared across every tenant.
That is a real enterprise-questionnaire red flag: one leaked/compromised
key exposes every customer's credentials at once (single point of
compromise, blast radius = 100% of tenants).

Architecture — standard envelope encryption
--------------------------------------------
This module implements the same shape AWS KMS / GCP KMS / HashiCorp Vault
use for "envelope encryption":

    KEK (key-encryption-key)  = Fernet(DB_ENCRYPTION_KEY)   — the existing
                                 root secret. Unchanged env var, unchanged
                                 meaning, just a new role: it no longer
                                 encrypts credentials directly.
    DEK (data-encryption-key) = a fresh, random Fernet key — ONE per
                                 organization, generated on first use.
    wrapped_dek               = KEK.encrypt(DEK's raw key bytes)
                                 — persisted in `org_encryption_keys`,
                                 never in plaintext.
    credential blob           = "ENV1:" + org_id + ":" + DEK.encrypt(json(data))

Compromising one org's DEK exposes only that org's credentials — the
root KEK is never used to decrypt credential data directly in the new
scheme, only to unwrap DEKs. This is the real per-tenant blast-radius
isolation the flat-key design was missing.

Legacy-scheme detection (no breaking change, no batch migration)
------------------------------------------------------------------
Every blob written before this ticket is a bare Fernet token (no prefix —
Fernet tokens never started with a colon-delimited tag). Every blob written
by this module starts with the literal ``"ENV1:"`` prefix, which a Fernet
token can never collide with (Fernet's URL-safe base64 alphabet does not
begin with the literal ASCII text "ENV1:"). So:

    blob.startswith("ENV1:")  → new envelope scheme → unwrap that org's
                                 DEK, decrypt the remainder with it.
    otherwise                 → legacy scheme → decrypt directly with the
                                 KEK, exactly as `encrypt_credentials` /
                                 `decrypt_credentials` in
                                 `api.utils.encryption` always have.

`decrypt_credentials_with_migration()` is the ONLY place that performs
migration, and it only ever does so by re-encrypting a blob it just
*successfully decrypted* in this process and handing the caller a new
blob to write back on their next read-triggered save. Nothing in this
module ever batch-reads, batch-decrypts, or batch-rewrites existing rows —
that is a deliberate, hard constraint (a wrong batch decrypt/re-encrypt
has no recovery path; there is no backup of the plaintext).

Future KMS upgrade path
------------------------
`_wrap_dek()` and `_unwrap_dek()` are the ONLY two functions in this
module that touch the KEK. Upgrading from "KEK = Fernet(DB_ENCRYPTION_KEY)"
to "KEK = a real AWS/GCP KMS key" is a small, isolated change to the
bodies of exactly those two functions (swap the Fernet encrypt/decrypt
calls for a KMS `Encrypt`/`Decrypt` or `GenerateDataKey` API call) — every
other function in this module (DEK generation, per-credential encryption,
the `org_encryption_keys` table, the legacy-detection logic, the lazy
migration path) is completely unaffected. That is the whole point of the
envelope shape: it is KMS-shaped software today, so it is a drop-in swap
to real KMS later, not a rearchitecture.

What this DOES protect against
--------------------------------
- A leaked/compromised DEK for one org no longer exposes any other org's
  credentials (per-tenant blast-radius isolation — the gap this ticket
  closes).
- Legacy (pre-migration) blobs continue to decrypt correctly with zero
  downtime and zero batch migration risk; they self-migrate to the new,
  isolated scheme the next time they are naturally read.

What this does NOT protect against (honest limitation)
----------------------------------------------------------
- A compromised `DB_ENCRYPTION_KEY` (the KEK) itself. Anyone holding the
  KEK can unwrap every org's DEK and therefore decrypt every org's
  credentials — the KEK is still one flat secret held by this process.
  Closing THAT gap requires the KEK itself to live in a real external
  KMS/HSM that this process calls into rather than holds in an env var —
  exactly the upgrade path documented above, and exactly why a real KMS
  account (which this environment cannot provision) is the next step,
  not something this ticket claims to deliver.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Column, DateTime, String, Table, Text, func, select, update
from sqlalchemy import insert as sa_insert

from api.utils.encryption import _get_fernet as _kek

logger = logging.getLogger(__name__)

# Envelope-scheme blobs are tagged "ENV1:<org_id>:<fernet-token>". A bare
# legacy Fernet token can never start with this literal prefix.
ENVELOPE_PREFIX = "ENV1"
_SEP = ":"


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

def register_org_encryption_keys_table(metadata: Any) -> Table:
    """Define the `org_encryption_keys` table on shared metadata.

    One row per organization: its DEK, wrapped (encrypted) by the KEK.
    Never stores a DEK in plaintext. Called once at import time from
    `api/index.py`, mirroring every other `register_*_table` helper.
    """
    return Table(
        "org_encryption_keys",
        metadata,
        Column("org_id", String, primary_key=True),
        Column("wrapped_dek", Text, nullable=False),
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )


# ---------------------------------------------------------------------------
# KEK-touching functions — the ONLY two a real KMS swap would change
# ---------------------------------------------------------------------------

def _wrap_dek(dek: bytes) -> str:
    """Encrypt a DEK's raw key bytes with the KEK.

    Future KMS upgrade: replace this body with a KMS `Encrypt` (or
    `GenerateDataKey`) API call and return its ciphertext blob instead.
    Every caller of `_wrap_dek` is unaffected by that change.
    """
    return _kek().encrypt(dek).decode()


def _unwrap_dek(wrapped: str) -> bytes:
    """Decrypt a wrapped DEK with the KEK.

    Future KMS upgrade: replace this body with a KMS `Decrypt` API call.
    Every caller of `_unwrap_dek` is unaffected by that change.
    """
    try:
        return _kek().decrypt(wrapped.encode() if isinstance(wrapped, str) else wrapped)
    except InvalidToken as exc:
        raise InvalidToken("DEK unwrap failed — invalid token or KEK mismatch.") from exc


# ---------------------------------------------------------------------------
# Per-org DEK lifecycle
# ---------------------------------------------------------------------------

async def get_or_create_org_dek(
    org_id: str,
    async_session_factory: Any,
    org_keys_table: Table,
) -> bytes:
    """Return *org_id*'s DEK (raw Fernet key bytes), creating one on first use.

    Safe under concurrent callers: if two requests race to create the same
    org's DEK, the loser's insert is a no-op (or is caught) and a re-select
    returns the winner's row — every caller ends up using the same DEK.
    """
    async with async_session_factory() as session:
        result = await session.execute(
            select(org_keys_table.c.wrapped_dek).where(org_keys_table.c.org_id == org_id)
        )
        row = result.first()

    if row is not None:
        return _unwrap_dek(row[0])

    dek = Fernet.generate_key()
    wrapped = _wrap_dek(dek)
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        async with session.begin():
            try:
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                stmt = (
                    pg_insert(org_keys_table)
                    .values(org_id=org_id, wrapped_dek=wrapped, created_at=now)
                    .on_conflict_do_nothing(index_elements=["org_id"])
                )
                await session.execute(stmt)
            except Exception:
                # Fallback for SQLite (tests) or any backend without a
                # native upsert — a PK collision here just means another
                # caller already created this org's DEK; swallow it and
                # fall through to the re-select below.
                try:
                    await session.execute(
                        sa_insert(org_keys_table).values(
                            org_id=org_id, wrapped_dek=wrapped, created_at=now
                        )
                    )
                except Exception:
                    pass

    async with async_session_factory() as session:
        result = await session.execute(
            select(org_keys_table.c.wrapped_dek).where(org_keys_table.c.org_id == org_id)
        )
        row = result.first()

    if row is None:
        # Should be unreachable — fail loudly rather than silently handing
        # back an un-persisted DEK that the next read could never recover.
        raise RuntimeError(f"Failed to persist a DEK for org {org_id!r}")

    return _unwrap_dek(row[0])


# ---------------------------------------------------------------------------
# DEK-level encrypt/decrypt
# ---------------------------------------------------------------------------

def _encrypt_with_dek(data: dict, dek: bytes) -> str:
    return Fernet(dek).encrypt(json.dumps(data).encode()).decode()


def _decrypt_with_dek(token: str, dek: bytes) -> dict:
    try:
        plaintext = Fernet(dek).decrypt(token.encode() if isinstance(token, str) else token)
    except InvalidToken as exc:
        raise InvalidToken(
            "Credential decryption failed (envelope scheme) — invalid token or DEK mismatch."
        ) from exc
    return json.loads(plaintext.decode())


def is_envelope_blob(blob: Any) -> bool:
    """True if *blob* was encrypted with the new per-org envelope scheme."""
    return isinstance(blob, str) and blob.startswith(ENVELOPE_PREFIX + _SEP)


async def encrypt_credentials_for_org(
    data: dict,
    org_id: str,
    async_session_factory: Any,
    org_keys_table: Table,
) -> str:
    """Encrypt *data* with *org_id*'s DEK and return a tagged envelope blob."""
    dek = await get_or_create_org_dek(org_id, async_session_factory, org_keys_table)
    token = _encrypt_with_dek(data, dek)
    return f"{ENVELOPE_PREFIX}{_SEP}{org_id}{_SEP}{token}"


async def decrypt_credentials_with_migration(
    blob: str,
    org_id: Optional[str],
    async_session_factory: Any,
    org_keys_table: Table,
) -> tuple[dict, Optional[str]]:
    """Decrypt *blob*, auto-detecting legacy vs. envelope scheme.

    Returns ``(data, new_blob)``. ``new_blob`` is non-``None`` only when
    *blob* was on the legacy scheme, *org_id* is known, and migration to
    the per-org envelope scheme succeeded — the caller should write
    ``new_blob`` back over the stored blob (the lazy, read-triggered
    re-encryption path). ``new_blob`` is ``None`` whenever no rewrite is
    needed (already on the envelope scheme) or possible (no org_id known
    yet for this row — stays on the legacy scheme until it is).

    Never mutates any stored row itself — callers own the write-back.
    """
    if is_envelope_blob(blob):
        try:
            _, stored_org_id, token = blob.split(_SEP, 2)
        except ValueError as exc:
            raise InvalidToken("Malformed envelope-encrypted blob.") from exc
        dek = await get_or_create_org_dek(stored_org_id, async_session_factory, org_keys_table)
        data = _decrypt_with_dek(token, dek)
        return data, None

    # Legacy scheme: direct-KEK Fernet, exactly as it always worked before
    # this ticket (api.utils.encryption.decrypt_credentials).
    from api.utils.encryption import decrypt_credentials as _legacy_decrypt

    data = _legacy_decrypt(blob)

    if not org_id:
        return data, None

    try:
        new_blob = await encrypt_credentials_for_org(data, org_id, async_session_factory, org_keys_table)
    except Exception:
        # Migration is a best-effort side effect of a successful read — a
        # failure here must never turn a successful decrypt into an error.
        logger.warning("Lazy envelope migration failed for org %s", org_id, exc_info=True)
        return data, None

    return data, new_blob


async def write_back_blob(
    table: Table,
    id_column: str,
    id_value: Any,
    blob_column: str,
    new_blob: str,
    async_session_factory: Any,
) -> None:
    """Persist a migrated credential blob back over the row it came from.

    Generic across tables (`db_connections`, `marketplace_connections`, …)
    so every lazy-migration call site shares one write path. Never raises —
    a failed write-back just means the row stays on the legacy scheme and
    is retried on its next read; it must never fail the read that triggered it.
    """
    try:
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(table)
                    .where(table.c[id_column] == id_value)
                    .values(**{blob_column: new_blob})
                )
        logger.info(
            "Lazily migrated credentials for %s=%s on %s to per-org envelope scheme",
            id_column, id_value, table.name,
        )
    except Exception:
        logger.warning(
            "Failed to write back migrated credentials for %s=%s on %s",
            id_column, id_value, table.name, exc_info=True,
        )
