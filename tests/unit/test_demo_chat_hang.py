"""DOCBOT-2583: demo chat must never hang when Groq rate-limits."""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.sandbox_service import df_preamble
from api.usage_service import LLM_CALLS_BILLING_MIGRATIONS
from api.utils import llm_provider


def test_gemini_stream_stall_raises_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(llm_provider, "GEMINI_TIMEOUT_S", 0.2)

    def _stalled():
        time.sleep(5)
        yield MagicMock(text="late", candidates=None)

    fake_model = MagicMock()
    fake_model.generate_content.return_value = _stalled()
    with patch("google.generativeai.GenerativeModel", return_value=fake_model), \
         patch("google.generativeai.configure"):
        with pytest.raises(TimeoutError):
            list(llm_provider._gemini_completion_stream([{"role": "user", "content": "hi"}]))
    assert fake_model.generate_content.call_args.kwargs["request_options"]["retry"] is None


def test_gemini_stream_error_propagates(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    def _boom():
        raise RuntimeError("429")
        yield  # pragma: no cover

    fake_model = MagicMock()
    fake_model.generate_content.return_value = _boom()
    with patch("google.generativeai.GenerativeModel", return_value=fake_model), \
         patch("google.generativeai.configure"):
        with pytest.raises(RuntimeError):
            list(llm_provider._gemini_completion_stream([{"role": "user", "content": "hi"}]))


def test_groq_stream_does_not_retry_429():
    assert llm_provider.GROQ_STREAM_MAX_RETRIES == 0


def test_df_preamble_binds_df():
    ns: dict = {}
    exec("import pandas as pd, json\n" + df_preamble([{"a": 1, "b": "x"}]) + "n = len(df)", ns)  # noqa: S102
    assert ns["n"] == 1 and list(ns["df"].columns) == ["a", "b"]


def test_prompt_version_migration_is_idempotent():
    ddl = [s for s in LLM_CALLS_BILLING_MIGRATIONS if "prompt_version" in s]
    assert ddl and all("IF NOT EXISTS" in s for s in ddl)


@pytest.mark.asyncio
async def test_hybrid_emits_error_and_done_when_llm_fails():
    from api.hybrid_service import IntentClassification, hybrid_chat

    async def _fail(*a, **k):
        raise RuntimeError("both providers down")
        yield  # pragma: no cover

    class _Sess:
        async def __aenter__(self):
            return MagicMock(execute=AsyncMock(), commit=AsyncMock())

        async def __aexit__(self, *a):
            return False

    with (
        patch("api.hybrid_service.classify_intent_safe", new_callable=AsyncMock) as cls,
        patch("api.hybrid_service.rag_retrieve", new_callable=AsyncMock) as rag,
        patch("api.utils.llm_provider.achat_completion_stream", _fail),
    ):
        cls.return_value = IntentClassification(
            intent="doc", fallback_applied=True, question_hash="abc123def456789a"
        )
        rag.return_value = ("ctx", [])
        chunks = []
        async for raw in hybrid_chat(
            question="q", session_id="s", connection_id=None, persona="Data Analyst",
            has_docs=True, messages_table=MagicMock(), sessions_table=MagicMock(),
            db_connections_table=MagicMock(), schema_cache_table=MagicMock(),
            query_history_table=MagicMock(), query_embeddings_table=MagicMock(),
            async_session_factory=lambda: _Sess(),
            expert_personas={"Data Analyst": {"persona_def": "x"}},
            vector_stores={"s": MagicMock()},
        ):
            chunks.append(json.loads(raw.removeprefix("data: ").strip()))
    types = [c["type"] for c in chunks]
    assert "error" in types and types[-1] == "done"
