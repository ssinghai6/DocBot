"""DOCBOT-2335: tie-out demo questions must not hang ~180s.

A hung LLM call in the DB lane is cut off at SQL_LANE_TIMEOUT_S and the user
gets an explicit notice; the Groq/Gemini clients are built with bounded
timeouts instead of the SDK default (60s x 3 attempts = ~180s). No network.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from api.hybrid_service import IntentClassification, hybrid_chat
from api.utils import llm_provider

QUESTION = "Does segment revenue in Note 12 agree with the books?"


async def _hung_sql(**_kwargs):
    await asyncio.sleep(3600)


async def _run():
    with (
        patch("api.hybrid_service.classify_intent_safe", new_callable=AsyncMock) as cls,
        patch("api.hybrid_service.rag_retrieve", new_callable=AsyncMock) as rag,
        patch("api.hybrid_service._collect_sql_result", side_effect=_hung_sql),
        patch("api.hybrid_service.SQL_LANE_TIMEOUT_S", 0.2),
        patch("api.utils.llm_provider.chat_completion_stream", return_value=iter(["ok"])),
    ):
        cls.return_value = IntentClassification(
            intent="hybrid", fallback_applied=False, question_hash="abc123def456789a"
        )
        rag.return_value = ("Lab Services: $104.9M", [{"source": "10k.pdf", "page": 9}])
        out = []
        async for chunk in hybrid_chat(
            question=QUESTION,
            session_id="s1",
            connection_id="c1",
            persona="Finance Expert",
            has_docs=True,
            messages_table=MagicMock(),
            sessions_table=MagicMock(),
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Finance Expert": {"persona_def": "cfo"}},
            vector_stores={"s1": MagicMock()},
        ):
            if chunk.startswith("data: "):
                out.append(json.loads(chunk[6:]))
        return out


async def test_hung_sql_lane_is_bounded_with_user_notice():
    start = time.monotonic()
    events = await _run()
    assert time.monotonic() - start < 5
    tokens = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert "database query did not finish" in tokens
    assert events[-1]["type"] == "done"


def test_groq_client_has_bounded_timeout(monkeypatch):
    monkeypatch.setenv("groq_api_key", "fake")
    resp = MagicMock()
    resp.choices[0].message.content = "SELECT 1"
    with patch("groq.Groq") as groq_cls:
        groq_cls.return_value.chat.completions.create.return_value = resp
        llm_provider.chat_completion([{"role": "user", "content": "q"}])
    kwargs = groq_cls.call_args.kwargs
    # SDK default was 60s x 3 attempts (~180s) before the Gemini fallback ran
    assert kwargs["timeout"] * (kwargs["max_retries"] + 1) <= 60
