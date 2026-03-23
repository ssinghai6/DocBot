"""Unit tests for hybrid_chat pipeline — DOCBOT-402.

All tests are fully synchronous from the test runner's perspective; async
functions are driven by pytest-asyncio.  No real network calls are made.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_groq_streaming_response(tokens: list[str]) -> MagicMock:
    """Build a minimal mock streaming response that yields token chunks."""
    chunks = []
    for token in tokens:
        delta = MagicMock()
        delta.content = token
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        chunks.append(chunk)
    return iter(chunks)


def _make_groq_client(tokens: list[str] | None = None) -> MagicMock:
    """Return a mock Groq client whose chat.completions.create returns a streaming response."""
    if tokens is None:
        tokens = ["Hello", " world"]
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value=_make_groq_streaming_response(tokens)
    )
    return client


async def _collect_chunks(gen: AsyncGenerator) -> list[dict]:
    """Drain an SSE async generator and parse all data: lines into dicts."""
    chunks = []
    async for raw in gen:
        if raw.startswith("data: "):
            chunks.append(json.loads(raw[6:].strip()))
    return chunks


_noop_session_factory = MagicMock()


# ---------------------------------------------------------------------------
# rag_retrieve tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestRagRetrieve:

    async def test_returns_empty_when_session_not_in_vector_stores(self):
        """Must return ('', []) when the session has no vector store."""
        from api.hybrid_service import rag_retrieve

        context, citations = await rag_retrieve(
            question="What does the contract say?",
            session_id="unknown-session",
            vector_stores={},
        )
        assert context == ""
        assert citations == []

    async def test_retrieves_context_and_citations(self):
        """Must return non-empty context and citation list from mock vector store."""
        mock_doc = MagicMock()
        mock_doc.page_content = "This is the relevant content."
        mock_doc.metadata = {"source": "contract.pdf", "page": 3}

        mock_retriever = MagicMock()
        mock_retriever.invoke = MagicMock(return_value=[mock_doc])

        mock_db = MagicMock()
        mock_db.as_retriever = MagicMock(return_value=mock_retriever)

        from api.hybrid_service import rag_retrieve

        context, citations = await rag_retrieve(
            question="What does the contract say?",
            session_id="sess-abc",
            vector_stores={"sess-abc": mock_db},
        )

        assert "relevant content" in context
        assert len(citations) == 1
        assert citations[0]["source"] == "contract.pdf"
        assert citations[0]["page"] == 3

    async def test_returns_empty_on_retrieval_error(self):
        """Must return ('', []) and not raise when retriever raises."""
        mock_db = MagicMock()
        mock_db.as_retriever = MagicMock(side_effect=RuntimeError("Vector store error"))

        from api.hybrid_service import rag_retrieve

        context, citations = await rag_retrieve(
            question="Anything",
            session_id="sess-err",
            vector_stores={"sess-err": mock_db},
        )
        assert context == ""
        assert citations == []


# ---------------------------------------------------------------------------
# hybrid_chat intent routing tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestHybridChatIntentRouting:

    async def _run_hybrid(
        self,
        intent: str,
        has_docs: bool = True,
        connection_id: str | None = "conn-1",
        sql_result: dict | None = None,
        rag_result: tuple | None = None,
    ) -> list[dict]:
        """Run hybrid_chat with mocked intent and sources, return parsed chunks."""
        from api.hybrid_service import hybrid_chat

        groq_client = _make_groq_client(["The", " answer", "."])

        if rag_result is None:
            rag_result = ("Doc context here.", [{"source": "doc.pdf", "page": 1}])
        if sql_result is None:
            sql_result = {
                "type": "metadata",
                "result_preview": [{"col": 1}],
                "row_count": 1,
                "sources": ["orders"],
            }

        with (
            patch("api.hybrid_service.classify_intent_safe", new_callable=AsyncMock) as mock_cls,
            patch("api.hybrid_service.rag_retrieve", new_callable=AsyncMock) as mock_rag,
            patch("api.hybrid_service._collect_sql_result", new_callable=AsyncMock) as mock_sql,
            patch("api.hybrid_service.os.getenv", return_value="fake-groq-key"),
            patch("groq.Groq", return_value=groq_client),
        ):
            from api.hybrid_service import IntentClassification

            mock_cls.return_value = IntentClassification(
                intent=intent,
                fallback_applied=(not has_docs or connection_id is None),
                question_hash="abc123def456789a",
            )
            mock_rag.return_value = rag_result
            mock_sql.return_value = sql_result

            gen = hybrid_chat(
                question="What is the revenue?",
                session_id="sess-test",
                connection_id=connection_id,
                persona="Data Analyst",
                has_docs=has_docs,
                messages_table=MagicMock(),
                sessions_table=MagicMock(),
                db_connections_table=MagicMock(),
                schema_cache_table=MagicMock(),
                query_history_table=MagicMock(),
                query_embeddings_table=MagicMock(),
                async_session_factory=_noop_session_factory,
                expert_personas={"Data Analyst": {"persona_def": "You are a data analyst."}},
                vector_stores={"sess-test": MagicMock()},
            )
            return await _collect_chunks(gen)

    async def test_metadata_chunk_always_first(self):
        """First chunk must always be metadata regardless of intent."""
        chunks = await self._run_hybrid("hybrid")
        assert chunks[0]["type"] == "metadata"

    async def test_doc_intent_skips_sql(self):
        """When intent is 'doc', SQL pipeline must not be called."""
        with (
            patch("api.hybrid_service.classify_intent_safe", new_callable=AsyncMock) as mock_cls,
            patch("api.hybrid_service.rag_retrieve", new_callable=AsyncMock) as mock_rag,
            patch("api.hybrid_service._collect_sql_result", new_callable=AsyncMock) as mock_sql,
            patch("api.hybrid_service.os.getenv", return_value="fake-groq-key"),
            patch("groq.Groq", return_value=_make_groq_client()),
        ):
            from api.hybrid_service import hybrid_chat, IntentClassification

            mock_cls.return_value = IntentClassification(
                intent="doc", fallback_applied=True, question_hash="abc123def456789a"
            )
            mock_rag.return_value = ("Doc context.", [])

            gen = hybrid_chat(
                question="Summarize the document",
                session_id="sess-1",
                connection_id="conn-1",
                persona="Data Analyst",
                has_docs=True,
                messages_table=MagicMock(), sessions_table=MagicMock(),
                db_connections_table=MagicMock(), schema_cache_table=MagicMock(),
                query_history_table=MagicMock(), query_embeddings_table=MagicMock(),
                async_session_factory=_noop_session_factory,
                expert_personas={"Data Analyst": {"persona_def": "You are a data analyst."}},
                vector_stores={"sess-1": MagicMock()},
            )
            await _collect_chunks(gen)

        mock_sql.assert_not_called()

    async def test_sql_intent_skips_rag(self):
        """When intent is 'sql', RAG must not be called."""
        with (
            patch("api.hybrid_service.classify_intent_safe", new_callable=AsyncMock) as mock_cls,
            patch("api.hybrid_service.rag_retrieve", new_callable=AsyncMock) as mock_rag,
            patch("api.hybrid_service._collect_sql_result", new_callable=AsyncMock) as mock_sql,
            patch("api.hybrid_service.os.getenv", return_value="fake-groq-key"),
            patch("groq.Groq", return_value=_make_groq_client()),
        ):
            from api.hybrid_service import hybrid_chat, IntentClassification

            mock_cls.return_value = IntentClassification(
                intent="sql", fallback_applied=False, question_hash="abc123def456789a"
            )
            mock_sql.return_value = {
                "type": "metadata", "result_preview": [{"id": 1}],
                "row_count": 1, "sources": ["orders"],
            }

            gen = hybrid_chat(
                question="How many orders?",
                session_id="sess-2",
                connection_id="conn-1",
                persona="Data Analyst",
                has_docs=False,
                messages_table=MagicMock(), sessions_table=MagicMock(),
                db_connections_table=MagicMock(), schema_cache_table=MagicMock(),
                query_history_table=MagicMock(), query_embeddings_table=MagicMock(),
                async_session_factory=_noop_session_factory,
                expert_personas={"Data Analyst": {"persona_def": "You are a data analyst."}},
                vector_stores={},
            )
            await _collect_chunks(gen)

        mock_rag.assert_not_called()

    async def test_done_chunk_always_last(self):
        """Last chunk must always be done."""
        chunks = await self._run_hybrid("hybrid")
        assert chunks[-1]["type"] == "done"

    async def test_token_chunks_in_middle(self):
        """Must yield at least one token chunk between metadata and done."""
        chunks = await self._run_hybrid("hybrid")
        types = [c["type"] for c in chunks]
        assert "token" in types
        meta_idx = types.index("metadata")
        done_idx = types.index("done")
        assert meta_idx < done_idx
        assert any(t == "token" for t in types[meta_idx + 1:done_idx])
