"""Server-side Autopilot data-source resolution.

Background
----------
The ``/api/autopilot/run`` route used to pass the client-asserted
``has_docs`` / ``has_db`` / ``has_csv`` booleans straight into
``autopilot_service.run_autopilot()``. When the frontend failed to set a
flag (e.g. ``has_csv`` for an uploaded CSV), Autopilot believed the session
had no data source and collapsed to document-only retrieval — even though
an encrypted CSV row was persisted for the session — and the final answer
apologized that "no numeric data" existed.

Resolution is now server-side from persisted state:

  - ``db_connections`` rows scoped to the chat session (plus the explicit
    ``connection_id`` when the client sent one) are loaded and their
    encrypted creds blobs decrypted (legacy Fernet scheme or the per-org
    envelope scheme, DOCBOT-1529). ``dialect == "csv"`` → ``has_csv`` (with
    the cached DataProfile surfaced when present); any other non-empty
    dialect → ``has_db`` — an unknown dialect must never zero out data.
  - ``has_docs`` requires an actual vector store for the chat session: the
    in-process ``VECTOR_STORES`` cache first, then the on-disk Chroma
    collection (the same lazy recovery ``/api/chat`` performs). A recovered
    store is injected back into the cache dict so downstream ``doc_search``
    (which reads ``vector_stores[session_id]``) can use it.

Client-sent flags keep their places in the request models for backwards
compatibility but are hints only: they are consulted solely as a fallback
when their own resolution leg fails (row lookup error / vector store
error), so a resolution failure can never zero out data source knowledge
the client held. Classification never logs credentials, blobs, or
exception payloads — only exception types and row ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional

from sqlalchemy import or_, select

from api.utils.encryption import decrypt_credentials
from api.utils.envelope_encryption import (
    decrypt_credentials_with_migration,
    is_envelope_blob,
)

logger = logging.getLogger(__name__)

CSV_DIALECT = "csv"


@dataclass(frozen=True)
class SourceFlags:
    """Resolved Autopilot data-source flags (+ optional cached CSV profile).

    The three booleans mirror ``autopilot_service.AutopilotState`` keys
    (``has_docs`` / ``has_db`` / ``has_csv``) exactly — ``as_kwargs()`` is
    the contract that keeps that alignment explicit.
    """

    has_docs: bool = False
    has_db: bool = False
    has_csv: bool = False
    data_profile: Optional[dict] = None

    def as_kwargs(self) -> dict[str, bool]:
        return {
            "has_docs": self.has_docs,
            "has_db": self.has_db,
            "has_csv": self.has_csv,
        }


def classify_connection(
    dialect: Optional[str], creds: Optional[dict] = None
) -> SourceFlags:
    """Pure classification of one decrypted connection row.

    ``dialect == "csv"`` → ``has_csv`` (DataProfile pulled from the decoded
    creds when present); an empty/missing dialect yields all-False flags;
    every other non-empty dialect counts as a SQL data source — including
    dialects this build doesn't list, so a future connector can never make
    data invisible through unknown-dialect classification.
    """
    normalized = dialect.strip().lower() if dialect else ""
    if not normalized:
        return SourceFlags()
    if normalized == CSV_DIALECT:
        profile: Optional[dict] = None
        if isinstance(creds, dict):
            raw_profile = creds.get("data_profile")
            if isinstance(raw_profile, dict):
                profile = raw_profile
        return SourceFlags(has_csv=True, data_profile=profile)
    return SourceFlags(has_db=True)


def docs_available(vector_stores: Optional[dict], session_id: str) -> bool:
    """Pure check: a vector store is registered for the session."""
    return bool(vector_stores) and vector_stores.get(session_id) is not None


async def _decrypt_creds_tolerant(
    blob: str,
    org_id: Optional[str],
    async_session_factory: Any,
    org_encryption_keys_table: Optional[Any],
) -> dict:
    """Decrypt a creds blob across both storage schemes (DOCBOT-1529).

    Legacy blobs decrypt with the flat KEK exactly as they always have;
    ``ENV1:``-tagged envelope blobs require the org DEK, which is only
    available when envelope encryption has been wired. Raises on failure —
    the caller owns the non-fatal handling.
    """
    if is_envelope_blob(blob):
        if org_encryption_keys_table is None:
            raise RuntimeError(
                "Envelope-encrypted blob encountered but envelope "
                "encryption is not wired."
            )
        creds, _new_blob = await decrypt_credentials_with_migration(
            blob, org_id, async_session_factory, org_encryption_keys_table
        )
        return creds
    return decrypt_credentials(blob)


async def resolve_autopilot_sources(
    session_id: str,
    connection_id: Optional[str] = None,
    *,
    db_connections_table: Any,
    async_session_factory: Any,
    vector_stores: Optional[dict] = None,
    org_encryption_keys_table: Optional[Any] = None,
    fallback: Optional[SourceFlags] = None,
) -> SourceFlags:
    """Resolve Autopilot data-source flags from persisted state.

    Never raises: each resolution leg (connection-row lookup, per-blob
    decryption, vector store check) is individually guarded — a failed leg
    logs a warning (exception type only, never credentials/blobs) and
    falls back to the client-sent flags passed in *fallback*, when given.
    """
    resolved = SourceFlags()

    # ── 1. db_connections rows (session-scoped + explicit connection) ──
    rows: Optional[list] = None
    try:
        conditions = [db_connections_table.c.session_id == session_id]
        if connection_id:
            conditions.append(db_connections_table.c.id == connection_id)
        async with async_session_factory() as db:
            result = await db.execute(
                select(db_connections_table).where(or_(*conditions))
            )
            rows = result.fetchall()
    except Exception as exc:
        logger.warning(
            "autopilot source resolution: db_connections lookup failed for "
            "session %s — data-source flags fall back to client hints (%s)",
            session_id,
            type(exc).__name__,
        )
        rows = None

    if rows:
        seen_ids: set = set()
        csv_profile: Optional[dict] = None
        for row in rows:
            row_id = getattr(row, "id", None)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)

            creds: Optional[dict] = None
            blob = getattr(row, "credentials_blob", None)
            if blob:
                try:
                    creds = await _decrypt_creds_tolerant(
                        blob,
                        getattr(row, "org_id", None),
                        async_session_factory,
                        org_encryption_keys_table,
                    )
                except Exception as exc:
                    logger.warning(
                        "autopilot source resolution: could not decrypt "
                        "credentials for connection %s (session %s) — "
                        "profile unavailable, dialect still trusted (%s)",
                        row_id,
                        session_id,
                        type(exc).__name__,
                    )

            row_flags = classify_connection(getattr(row, "dialect", None), creds)
            if row_flags.has_csv and csv_profile is None:
                csv_profile = row_flags.data_profile
            resolved = replace(
                resolved,
                has_db=resolved.has_db or row_flags.has_db,
                has_csv=resolved.has_csv or row_flags.has_csv,
            )
        if csv_profile is not None:
            resolved = replace(resolved, data_profile=csv_profile)
    elif rows is None and fallback is not None:
        # Row lookup itself failed — trust the client's word for the data
        # sources it believed the session had. Never raises data to False.
        resolved = replace(
            resolved,
            has_db=resolved.has_db or fallback.has_db,
            has_csv=resolved.has_csv or fallback.has_csv,
        )

    # ── 2. documents — in-process cache, then on-disk Chroma recovery ──
    if docs_available(vector_stores, session_id):
        resolved = replace(resolved, has_docs=True)
    else:
        try:
            from api.utils.embeddings_provider import get_embeddings
            from api.utils.vector_store import load_store

            store = load_store(session_id, get_embeddings())
            if store is not None:
                if vector_stores is not None:
                    vector_stores[session_id] = store
                resolved = replace(resolved, has_docs=True)
        except Exception as exc:
            logger.warning(
                "autopilot source resolution: vector store recovery failed "
                "for session %s — has_docs falls back to client hint (%s)",
                session_id,
                type(exc).__name__,
            )
            if fallback is not None and fallback.has_docs:
                resolved = replace(resolved, has_docs=True)

    return resolved
