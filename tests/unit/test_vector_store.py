"""Unit tests for api/utils/vector_store.py — DOCBOT-1001.

These tests use a real tmp-dir-backed Chroma instance (no external APIs).
Embeddings are replaced with a deterministic stub to keep tests fast and
offline.
"""

import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Deterministic embedding stub — returns a fixed-length float list
# ---------------------------------------------------------------------------

class _FixedEmbeddings:
    """Returns a deterministic 4-dim vector based on text hash."""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        h = hash(text) % 1000
        return [float(h), float(h + 1), float(h + 2), float(h + 3)]


EMBEDDINGS = _FixedEmbeddings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(n: int = 3) -> list[Document]:
    return [
        Document(page_content=f"content chunk {i}", metadata={"source": "test.pdf", "page": i})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateStore:

    def test_creates_retrievable_store(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store
            session_id = str(uuid.uuid4())
            store = create_store(session_id, _make_docs(), EMBEDDINGS)
            assert store is not None
            retriever = store.as_retriever(search_kwargs={"k": 2})
            results = retriever.invoke("content chunk")
            assert len(results) >= 1

    def test_collection_persists_to_disk(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store, _collection_exists, _collection_name
            session_id = str(uuid.uuid4())
            create_store(session_id, _make_docs(), EMBEDDINGS)
            assert _collection_exists(_collection_name(session_id))

    def test_re_upload_replaces_existing_collection(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store
            session_id = str(uuid.uuid4())
            create_store(session_id, _make_docs(2), EMBEDDINGS)
            # Re-upload with more docs — should not raise
            store2 = create_store(session_id, _make_docs(5), EMBEDDINGS)
            assert store2 is not None


class TestLoadStore:

    def test_loads_existing_collection(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store, load_store
            session_id = str(uuid.uuid4())
            create_store(session_id, _make_docs(), EMBEDDINGS)
            loaded = load_store(session_id, EMBEDDINGS)
            assert loaded is not None

    def test_returns_none_for_missing_session(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import load_store
            loaded = load_store(str(uuid.uuid4()), EMBEDDINGS)
            assert loaded is None

    def test_loaded_store_is_searchable(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store, load_store
            session_id = str(uuid.uuid4())
            create_store(session_id, _make_docs(4), EMBEDDINGS)
            loaded = load_store(session_id, EMBEDDINGS)
            results = loaded.as_retriever(search_kwargs={"k": 2}).invoke("content chunk 1")
            assert len(results) >= 1


class TestDeleteStore:

    def test_deletes_collection_from_disk(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store, delete_store, _collection_exists, _collection_name
            session_id = str(uuid.uuid4())
            create_store(session_id, _make_docs(), EMBEDDINGS)
            assert _collection_exists(_collection_name(session_id))
            delete_store(session_id)
            assert not _collection_exists(_collection_name(session_id))

    def test_delete_nonexistent_is_noop(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import delete_store
            # Should not raise
            delete_store(str(uuid.uuid4()))


class TestListStoredSessions:

    def test_returns_stored_session_ids(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store, list_stored_sessions
            s1 = str(uuid.uuid4())
            s2 = str(uuid.uuid4())
            create_store(s1, _make_docs(), EMBEDDINGS)
            create_store(s2, _make_docs(), EMBEDDINGS)
            stored = list_stored_sessions()
            assert s1 in stored
            assert s2 in stored

    def test_empty_when_no_persist_dir(self, tmp_path):
        non_existent = tmp_path / "no_such_dir"
        with patch("api.utils.vector_store._PERSIST_DIR", non_existent):
            from api.utils.vector_store import list_stored_sessions
            assert list_stored_sessions() == []

    def test_deleted_session_not_listed(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import create_store, delete_store, list_stored_sessions
            session_id = str(uuid.uuid4())
            create_store(session_id, _make_docs(), EMBEDDINGS)
            delete_store(session_id)
            assert session_id not in list_stored_sessions()
