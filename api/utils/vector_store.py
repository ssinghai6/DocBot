"""
DocBot Vector Store — DOCBOT-1001, retention/deletion lifecycle DOCBOT-1525

Wraps ChromaDB persistent store so uploaded documents survive Railway container
restarts and redeploys (when CHROMA_PERSIST_DIR is mounted to a persistent volume).

Collection naming: "session_<uuid_with_underscores>" — Chroma requires alphanumeric
+ underscores only; UUID hyphens are replaced.

Environment variables
---------------------
CHROMA_DIR                  Path where Chroma stores its SQLite + embedding files
                            (DOCBOT-2311). Takes precedence over the legacy
                            CHROMA_PERSIST_DIR name, which is still honoured.
                            Default: <tempdir>/docbot_chroma (ephemeral, wiped
                            on every Railway redeploy). Set to a Railway
                            persistent-volume mount path (e.g. /data/chroma) for
                            cross-redeploy persistence.
VECTOR_STORE_CACHE_MAX      Max session stores held in the in-process LRU cache
                            (DOCBOT-2311). Default: 64. Evicted sessions stay
                            on disk and are lazy-loaded on the next request.
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

import json
import logging
import os
import tempfile
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

def _resolve_persist_dir() -> Path:
    """DOCBOT-2311: CHROMA_DIR > CHROMA_PERSIST_DIR (legacy) > tempdir default.

    DOCBOT-1633: the default is resolved via tempfile (honours TMPDIR; /tmp on
    Linux) rather than a hardcoded /tmp literal. Point CHROMA_DIR at a mounted
    volume for durable storage.
    """
    configured = os.getenv("CHROMA_DIR") or os.getenv("CHROMA_PERSIST_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "docbot_chroma"


_PERSIST_DIR = _resolve_persist_dir()

# DOCBOT-2311: bounded in-process cache of loaded session stores.
_DEFAULT_CACHE_MAX = 64
VECTOR_STORE_CACHE_MAX = max(
    1, int(os.getenv("VECTOR_STORE_CACHE_MAX", str(_DEFAULT_CACHE_MAX)))
)

# DOCBOT-2311: user-facing copy when a session's vectors are gone from disk.
DOCUMENTS_EXPIRED_MESSAGE = (
    "Your uploaded documents have expired and are no longer available "
    "(this can happen after a server update). Please re-upload your documents "
    "to include them in answers."
)

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


class SessionStoreCache(OrderedDict):
    """DOCBOT-2311: LRU-capped ``{session_id: store}`` map.

    Drop-in replacement for the plain ``VECTOR_STORES`` dict. Reads and writes
    mark an entry most-recently-used; inserting past ``maxsize`` evicts the
    least-recently-used entry from memory only. Its Chroma collection stays on
    disk and :func:`get_or_load_store` lazy-loads it on the next request.
    """

    def __init__(self, maxsize: int | None = None) -> None:
        super().__init__()
        self.maxsize = maxsize if maxsize is not None else VECTOR_STORE_CACHE_MAX

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.maxsize:
            evicted, _ = self.popitem(last=False)
            logger.info("vector_store: evicted session %s from in-memory LRU cache", evicted)


def get_or_load_store(
    session_id: str,
    vector_stores: dict,
    embeddings_fn: Any = None,
) -> Any | None:
    """DOCBOT-2311: return the session's store, lazy-loading from disk on miss.

    ``vector_stores`` is the process cache (``VECTOR_STORES``). On a miss the
    Chroma collection for ``session_id`` is loaded from ``CHROMA_DIR`` and put
    back in the cache. Returns ``None`` when no vectors exist on disk either;
    callers must then surface :func:`documents_expired_event` rather than
    answering silently without documents.

    Ownership: collections are keyed purely by session id. Callers pass a
    session id that has already been owner-checked, so this never crosses
    owners.
    """
    if not session_id:
        return None
    if session_id in vector_stores:
        return vector_stores[session_id]
    if embeddings_fn is None:
        from api.utils.embeddings_provider import get_embeddings as embeddings_fn
    try:
        store = load_store(session_id, embeddings_fn())
    except Exception as exc:  # chromadb raises assorted error types on a corrupt dir
        logger.warning(
            "vector_store: lazy-load failed for session %s (%s)",
            session_id, type(exc).__name__,
        )
        return None
    if store is None:
        return None
    vector_stores[session_id] = store
    logger.info("vector_store: lazy-loaded session %s from disk", session_id)
    return store


def documents_expired_event(session_id: str) -> dict:
    """DOCBOT-2311: payload for the ``documents_expired`` SSE event."""
    return {
        "type": "documents_expired",
        "session_id": session_id,
        "message": DOCUMENTS_EXPIRED_MESSAGE,
    }


def documents_expired_sse(session_id: str) -> str:
    """SSE-framed :func:`documents_expired_event`."""
    return f"data: {json.dumps(documents_expired_event(session_id))}\n\n"


async def documents_expired_stream(session_id: str) -> AsyncIterator[str]:
    """Complete doc-chat SSE reply for a session whose vectors are gone.

    Emits the ``documents_expired`` event, the user-facing message as a token
    (so the chat bubble is never blank), empty citations, and ``done``.
    """
    yield documents_expired_sse(session_id)
    yield f"data: {json.dumps({'type': 'token', 'content': DOCUMENTS_EXPIRED_MESSAGE})}\n\n"
    yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


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
