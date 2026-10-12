"""
DocBot Vector Store — DOCBOT-1001, retention/deletion lifecycle DOCBOT-1525

Wraps ChromaDB persistent store so uploaded documents survive Railway container
restarts and redeploys (when CHROMA_PERSIST_DIR is mounted to a persistent volume).

Collection naming: "session_<uuid_with_underscores>" — Chroma requires alphanumeric
+ underscores only; UUID hyphens are replaced.

Environment variables
---------------------
CHROMA_PERSIST_DIR          Path where Chroma stores its SQLite + embedding files.
                            Default: /tmp/docbot_chroma
                            Set to a Railway persistent-volume mount path (e.g.
                            /data/chroma) for cross-redeploy persistence.
VECTOR_STORE_RETENTION_DAYS Inactivity window (days) after which a session's
                            vector store is eligible for automatic deletion.
                            Default: 30. Keyed off sessions.updated_at (the
                            same "last touched" column the chat API bumps on
                            every message), not created_at, so a long-running
                            but still-active session is never deleted
                            mid-conversation. See cleanup_expired_vector_stores()
                            below.

Encryption-at-rest status (DOCBOT-1525 finding — OPEN GAP, flagged, not fixed
in this module)
----------------------------------------------------------------------------
This module persists full document chunk text + embeddings to disk at
CHROMA_PERSIST_DIR in plaintext SQLite files. Chroma itself has no built-in
at-rest encryption option. Whether that data is encrypted at rest depends
entirely on the underlying storage:
  - Default (/tmp): ephemeral container filesystem. Railway does not publicly
    document at-rest encryption for ephemeral container storage; treat this
    as UNENCRYPTED unless Railway states otherwise.
  - Railway persistent volume (the recommended path, e.g. /data/chroma): as
    of this writing, this has NOT been independently verified against
    Railway's current infrastructure/compliance docs. Whether Railway
    volumes are encrypted at rest by default is an OPEN QUESTION — raise it
    with the user and confirm with Railway support or their SOC 2 report
    before treating customer document contents as encrypted at rest on
    disk. Do not assume this is handled.
This note is documentation/flagging only; no code change in this module can
fix storage-layer encryption, which is a Railway infrastructure property.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "/tmp/docbot_chroma"))

# DOCBOT-1525: default inactivity window before a session's vector store
# becomes eligible for automatic deletion. 30 days mirrors common SaaS
# "cold session" retention defaults — long enough that no legitimate
# multi-week research thread gets reaped, short enough that we are not
# holding customer document text indefinitely (the original complaint this
# ticket addresses). Configurable via VECTOR_STORE_RETENTION_DAYS.
_DEFAULT_RETENTION_DAYS = 30
VECTOR_STORE_RETENTION_DAYS = int(
    os.getenv("VECTOR_STORE_RETENTION_DAYS", str(_DEFAULT_RETENTION_DAYS))
)


def _collection_name(session_id: str) -> str:
    """Convert a UUID session_id to a valid Chroma collection name."""
    return f"session_{session_id.replace('-', '_')}"


def get_persist_dir() -> Path:
    return _PERSIST_DIR


def create_store(session_id: str, documents: list, embeddings: Any) -> Any:
    """
    Create a new persistent Chroma collection for session_id, index documents,
    and return the LangChain Chroma wrapper.

    Replaces InMemoryVectorStore.from_documents().
    """
    from langchain_chroma import Chroma

    _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    collection = _collection_name(session_id)

    # If a stale collection from a previous upload exists, delete it first
    # so we start fresh (re-upload scenario).
    _delete_collection_if_exists(collection)

    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=str(_PERSIST_DIR),
    )
    logger.info("vector_store: created collection %s (%d docs)", collection, len(documents))
    return store


def load_store(session_id: str, embeddings: Any) -> Any | None:
    """
    Load an existing persistent Chroma collection from disk.
    Returns None if the collection does not exist on disk.
    """
    from langchain_chroma import Chroma
    import chromadb

    collection = _collection_name(session_id)
    if not _collection_exists(collection):
        return None

    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(_PERSIST_DIR),
    )
    logger.info("vector_store: loaded collection %s from disk", collection)
    return store


def delete_store(session_id: str) -> None:
    """
    Delete the Chroma collection for session_id from disk.
    No-op if the collection does not exist.
    """
    collection = _collection_name(session_id)
    _delete_collection_if_exists(collection)
    logger.info("vector_store: deleted collection %s", collection)


def list_stored_sessions() -> list[str]:
    """
    Return session_ids for all collections persisted on disk.
    Used at startup to warm the in-process VECTOR_STORES dict.
    """
    import chromadb

    if not _PERSIST_DIR.exists():
        return []

    try:
        client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
        sessions = []
        for col in client.list_collections():
            name = col.name
            if name.startswith("session_"):
                # Reverse the underscore substitution: session_<uuid_underscores>
                # UUIDs are 32 hex chars + 4 hyphens = 36 chars
                # After replacement: 32 hex + 4 underscores = 36 chars
                raw = name[len("session_"):]
                # Re-insert hyphens at positions 8, 12, 16, 20
                parts = [raw[:8], raw[9:13], raw[14:18], raw[19:23], raw[24:]]
                sid = "-".join(parts)
                sessions.append(sid)
        return sessions
    except Exception as exc:
        logger.warning("vector_store: could not list stored sessions: %s", exc)
        return []


# ---------------------------------------------------------------------------
# DOCBOT-1525: retention / deletion lifecycle
# ---------------------------------------------------------------------------


async def cleanup_expired_vector_stores(
    sessions_table: Any,
    async_session_factory: Any,
    retention_days: int | None = None,
) -> int:
    """
    Delete the Chroma vector store for every session whose ``updated_at``
    is older than the retention window (default VECTOR_STORE_RETENTION_DAYS,
    30 days).

    Mirrors api.file_upload_service.cleanup_expired_uploads()'s shape:
    find expired rows, best-effort delete each, never let one failure stop
    the rest of the pass, log a summary count.

    Only sessions that currently have a persisted Chroma collection on disk
    are considered (list_stored_sessions()), so this never touches sessions
    that never uploaded a document (CSV/SQL-only sessions) or whose store
    was already deleted.

    Called on startup and intended to be scheduled periodically (e.g. daily
    via APScheduler) — a long-running Railway deployment may not restart
    often enough for startup-only cleanup to matter.

    Returns the number of vector stores deleted.
    """
    from sqlalchemy import select

    window_days = retention_days if retention_days is not None else VECTOR_STORE_RETENTION_DAYS
    if window_days <= 0:
        return 0  # 0 or negative disables the check, same convention as SESSION_COST_CEILING_USD

    stored_session_ids = list_stored_sessions()
    if not stored_session_ids:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    removed = 0

    async with async_session_factory() as session:
        result = await session.execute(
            select(sessions_table.c.session_id, sessions_table.c.updated_at).where(
                sessions_table.c.session_id.in_(stored_session_ids)
            )
        )
        rows = result.fetchall()

    # Sessions with a vector store on disk but no row in `sessions` (e.g. the
    # session record was deleted directly) are orphaned stores — clean them
    # up too rather than leaking them forever.
    known_ids = {row.session_id for row in rows}
    orphaned_ids = [sid for sid in stored_session_ids if sid not in known_ids]

    for row in rows:
        updated_at = row.updated_at
        if updated_at is None:
            continue
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        if updated_at >= cutoff:
            continue  # Still within the active window

        try:
            delete_store(row.session_id)
            logger.info(
                "vector_store cleanup: removed expired store for session %s "
                "(last active %s, retention %d day(s))",
                row.session_id, updated_at.isoformat(), window_days,
            )
            removed += 1
        except Exception as exc:
            logger.warning(
                "vector_store cleanup: failed to delete store for session %s: %s",
                row.session_id, exc,
            )
            continue  # Never let one failed deletion stop the rest of the pass

    for sid in orphaned_ids:
        try:
            delete_store(sid)
            logger.info("vector_store cleanup: removed orphaned store for session %s", sid)
            removed += 1
        except Exception as exc:
            logger.warning(
                "vector_store cleanup: failed to delete orphaned store for session %s: %s",
                sid, exc,
            )
            continue

    if removed:
        logger.info("vector_store cleanup: removed %d expired vector store(s).", removed)
    return removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_chroma_client():
    import chromadb
    _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_PERSIST_DIR))


def _collection_exists(collection_name: str) -> bool:
    try:
        client = _get_chroma_client()
        existing = [c.name for c in client.list_collections()]
        return collection_name in existing
    except Exception:
        return False


def _delete_collection_if_exists(collection_name: str) -> None:
    try:
        client = _get_chroma_client()
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
    except Exception as exc:
        logger.warning("vector_store: could not delete collection %s: %s", collection_name, exc)
