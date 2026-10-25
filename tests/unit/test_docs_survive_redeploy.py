"""DOCBOT-2311: uploaded document vectors survive redeploys.

Covers:
  - CHROMA_DIR env resolution (with legacy CHROMA_PERSIST_DIR fallback)
  - SessionStoreCache LRU cap
  - persist -> "new process" (cleared in-memory cache) -> lazy-load works
  - missing vectors -> explicit ``documents_expired`` SSE event in doc chat,
    hybrid chat and autopilot (never a silent no-docs answer)

Real tmp-dir Chroma, deterministic embedding stub, no network.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class _FixedEmbeddings:
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        h = hash(text) % 1000
        return [float(h), float(h + 1), float(h + 2), float(h + 3)]


EMBEDDINGS = _FixedEmbeddings()


def _docs(n: int = 3) -> list[Document]:
    return [
        Document(page_content=f"revenue chunk {i}", metadata={"source": "10k.pdf", "page": i})
        for i in range(n)
    ]


def _events(chunks: list[str]) -> list[dict]:
    out = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


async def _collect(agen) -> list[str]:
    return [c async for c in agen]


# ---------------------------------------------------------------------------
# CHROMA_DIR resolution
# ---------------------------------------------------------------------------


class TestPersistDirResolution:
    def test_chroma_dir_takes_precedence(self, monkeypatch, tmp_path):
        from api.utils.vector_store import _resolve_persist_dir

        monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "vol"))
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "legacy"))
        assert _resolve_persist_dir() == tmp_path / "vol"

    def test_legacy_env_still_honoured(self, monkeypatch, tmp_path):
        from api.utils.vector_store import _resolve_persist_dir

        monkeypatch.delenv("CHROMA_DIR", raising=False)
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "legacy"))
        assert _resolve_persist_dir() == tmp_path / "legacy"

    def test_default_is_tempdir(self, monkeypatch):
        import tempfile

        from api.utils.vector_store import _resolve_persist_dir

        monkeypatch.delenv("CHROMA_DIR", raising=False)
        monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
        assert _resolve_persist_dir() == Path(tempfile.gettempdir()) / "docbot_chroma"


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class TestSessionStoreCache:
    def test_evicts_least_recently_used(self):
        from api.utils.vector_store import SessionStoreCache

        cache = SessionStoreCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        _ = cache["a"]  # touch a -> b is now LRU
        cache["c"] = 3
        assert "b" not in cache
        assert set(cache) == {"a", "c"}

    def test_dict_semantics_preserved(self):
        from api.utils.vector_store import SessionStoreCache

        cache = SessionStoreCache(maxsize=3)
        cache["x"] = "store"
        assert cache.get("x") == "store"
        assert cache.pop("x") == "store"
        assert cache.pop("x", None) is None


# ---------------------------------------------------------------------------
# persist -> new process -> lazy-load
# ---------------------------------------------------------------------------


class TestLazyLoad:
    def test_lazy_loads_after_cache_cleared(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import SessionStoreCache, create_store, get_or_load_store

            sid = str(uuid.uuid4())
            cache = SessionStoreCache(maxsize=4)
            cache[sid] = create_store(sid, _docs(), EMBEDDINGS)

            # Simulate a redeploy onto the same volume: in-memory cache is gone.
            cache.clear()
            assert sid not in cache

            store = get_or_load_store(sid, cache, lambda: EMBEDDINGS)
            assert store is not None
            assert sid in cache  # re-cached
            hits = store.as_retriever(search_kwargs={"k": 2}).invoke("revenue chunk 1")
            assert len(hits) >= 1

    def test_evicted_session_reloads_from_disk(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import SessionStoreCache, create_store, get_or_load_store

            cache = SessionStoreCache(maxsize=1)
            s1, s2 = str(uuid.uuid4()), str(uuid.uuid4())
            cache[s1] = create_store(s1, _docs(), EMBEDDINGS)
            cache[s2] = create_store(s2, _docs(), EMBEDDINGS)
            assert s1 not in cache  # evicted from memory only
            assert get_or_load_store(s1, cache, lambda: EMBEDDINGS) is not None

    def test_missing_returns_none(self, tmp_path):
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path):
            from api.utils.vector_store import get_or_load_store

            cache: dict = {}
            assert get_or_load_store(str(uuid.uuid4()), cache, lambda: EMBEDDINGS) is None
            assert cache == {}

    def test_empty_session_id_returns_none(self):
        from api.utils.vector_store import get_or_load_store

        assert get_or_load_store("", {}, lambda: EMBEDDINGS) is None


# ---------------------------------------------------------------------------
# documents_expired events
# ---------------------------------------------------------------------------


class TestDocumentsExpired:
    def test_doc_chat_stream(self):
        from api.utils.vector_store import DOCUMENTS_EXPIRED_MESSAGE, documents_expired_stream

        events = _events(asyncio.run(_collect(documents_expired_stream("sid-1"))))
        assert events[0] == {
            "type": "documents_expired",
            "session_id": "sid-1",
            "message": DOCUMENTS_EXPIRED_MESSAGE,
        }
        assert {"type": "token", "content": DOCUMENTS_EXPIRED_MESSAGE} in events
        assert events[-1]["type"] == "done"

    def test_hybrid_docs_only_emits_expired_and_stops(self, tmp_path):
        from api.hybrid_service import hybrid_chat

        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path), patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=EMBEDDINGS
        ):
            gen = hybrid_chat(
                question="What was revenue?",
                session_id=str(uuid.uuid4()),
                connection_id=None,
                persona="Generalist",
                has_docs=True,
                messages_table=MagicMock(),
                sessions_table=MagicMock(),
                db_connections_table=MagicMock(),
                schema_cache_table=MagicMock(),
                query_history_table=MagicMock(),
                query_embeddings_table=MagicMock(),
                async_session_factory=MagicMock(),
                expert_personas={},
                vector_stores={},
            )
            events = _events(asyncio.run(_collect(gen)))

        types = [e["type"] for e in events]
        assert types[0] == "documents_expired"
        assert "token" in types
        assert types[-1] == "done"
        assert "metadata" not in types  # no silent answer pipeline ran

    def test_hybrid_with_db_flags_expiry_then_continues_db_only(self, tmp_path):
        import api.hybrid_service as hs

        seen: dict = {}

        async def _fake_classify(**kwargs):
            seen["has_docs"] = kwargs["has_docs"]
            raise RuntimeError("stop after classification")

        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path), patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=EMBEDDINGS
        ), patch.object(hs, "classify_intent_safe", side_effect=_fake_classify):
            gen = hs.hybrid_chat(
                question="Compare revenue",
                session_id=str(uuid.uuid4()),
                connection_id="conn-1",
                persona="Generalist",
                has_docs=True,
                messages_table=MagicMock(),
                sessions_table=MagicMock(),
                db_connections_table=MagicMock(),
                schema_cache_table=MagicMock(),
                query_history_table=MagicMock(),
                query_embeddings_table=MagicMock(),
                async_session_factory=MagicMock(),
                expert_personas={},
                vector_stores={},
            )
            chunks: list[str] = []

            async def _run():
                with pytest.raises(RuntimeError):
                    async for c in gen:
                        chunks.append(c)

            asyncio.run(_run())

        events = _events(chunks)
        assert events[0]["type"] == "documents_expired"
        assert events[1]["type"] == "token" and "expired" in events[1]["content"]
        assert seen["has_docs"] is False  # routed DB-only, explicitly

    def test_hybrid_lazy_loads_instead_of_expiring(self, tmp_path):
        import api.hybrid_service as hs
        from api.utils.vector_store import create_store

        seen: dict = {}

        async def _fake_classify(**kwargs):
            seen["has_docs"] = kwargs["has_docs"]
            raise RuntimeError("stop")

        sid = str(uuid.uuid4())
        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path), patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=EMBEDDINGS
        ), patch.object(hs, "classify_intent_safe", side_effect=_fake_classify):
            create_store(sid, _docs(), EMBEDDINGS)
            stores: dict = {}  # fresh process: nothing in memory
            gen = hs.hybrid_chat(
                question="Revenue?",
                session_id=sid,
                connection_id="conn-1",
                persona="Generalist",
                has_docs=True,
                messages_table=MagicMock(),
                sessions_table=MagicMock(),
                db_connections_table=MagicMock(),
                schema_cache_table=MagicMock(),
                query_history_table=MagicMock(),
                query_embeddings_table=MagicMock(),
                async_session_factory=MagicMock(),
                expert_personas={},
                vector_stores=stores,
            )
            chunks: list[str] = []

            async def _run():
                with pytest.raises(RuntimeError):
                    async for c in gen:
                        chunks.append(c)

            asyncio.run(_run())

        assert sid in stores
        assert seen["has_docs"] is True
        assert all(e["type"] != "documents_expired" for e in _events(chunks))

    def test_autopilot_docs_only_emits_expired(self, tmp_path):
        from api.autopilot_service import run_autopilot

        with patch("api.utils.vector_store._PERSIST_DIR", tmp_path), patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=EMBEDDINGS
        ):
            gen = run_autopilot(
                question="Analyze revenue trend",
                session_id=str(uuid.uuid4()),
                connection_id="",
                persona="Generalist",
                db_connections_table=MagicMock(),
                schema_cache_table=MagicMock(),
                query_history_table=MagicMock(),
                query_embeddings_table=MagicMock(),
                session_artifacts_table=MagicMock(),
                table_embeddings_table=MagicMock(),
                async_session_factory=MagicMock(),
                expert_personas={},
                vector_stores={},
                has_docs=True,
            )
            events = _events(asyncio.run(_collect(gen)))

        types = [e["type"] for e in events]
        assert types == ["documents_expired", "answer", "done"]
