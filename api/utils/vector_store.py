"""
DocBot Vector Store — DOCBOT-1001

Wraps ChromaDB persistent store so uploaded documents survive Railway container
restarts and redeploys (when CHROMA_PERSIST_DIR is mounted to a persistent volume).

Collection naming: "session_<uuid_with_underscores>" — Chroma requires alphanumeric
+ underscores only; UUID hyphens are replaced.

Environment variables
---------------------
CHROMA_PERSIST_DIR  Path where Chroma stores its SQLite + embedding files.
                    Default: /tmp/docbot_chroma
                    Set to a Railway persistent-volume mount path (e.g. /data/chroma)
                    for cross-redeploy persistence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "/tmp/docbot_chroma"))


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
