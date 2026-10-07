"""DOCBOT-1510: hybrid_chat emits lineage + forwards SQL metadata."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from api.hybrid_service import IntentClassification, hybrid_chat


class _noop_session_factory:  # pragma: no cover - never used (tables are mocked)
    def __call__(self):
        return MagicMock()


def _groq_client():
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "ok"
    client = MagicMock()
    client.chat.completions.create.return_value = iter([chunk])
    return client


async def _run(sql_result, rag_result):
    with (
        patch("api.hybrid_service.classify_intent_safe", new_callable=AsyncMock) as cls,
        patch("api.hybrid_service.rag_retrieve", new_callable=AsyncMock) as rag,
        patch("api.hybrid_service._collect_sql_result", new_callable=AsyncMock) as sql,
        patch("api.hybrid_service.os.getenv", return_value="fake"),
        patch("groq.Groq", return_value=_groq_client()),
        patch("api.hybrid_service.chat_completion_stream", create=True),
        patch("api.utils.llm_provider.chat_completion_stream", return_value=iter(["ok"])),
    ):
        cls.return_value = IntentClassification(
            intent="hybrid", fallback_applied=False, question_hash="abc123def456789a"
        )
        rag.return_value = rag_result
        sql.return_value = sql_result
        gen = hybrid_chat(
            question="Q4 net income?",
            session_id="s1",
            connection_id="c1",
            persona="Data Analyst",
            has_docs=True,
            messages_table=MagicMock(),
            sessions_table=MagicMock(),
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            async_session_factory=_noop_session_factory(),
            expert_personas={"Data Analyst": {"persona_def": "analyst"}},
            vector_stores={"s1": MagicMock()},
        )
        out = []
        async for chunk in gen:
            if chunk.startswith("data: "):
                out.append(json.loads(chunk[6:]))
        return out


async def test_lineage_emitted_before_done_with_sql_and_steps():
    events = await _run(
        {
            "type": "metadata",
            "sql_query": "SELECT 1",
            "explanation": "e",
            "row_count": 1,
            "sources": ["income"],
            "result_preview": [{"net_income": 330}],
        },
        ("Net income was $325M in Q4", [{"source": "10k.pdf", "page": 2}]),
    )
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert types[-2] == "lineage"
    lineage = events[-2]
    assert lineage["mode"] == "hybrid"
    assert lineage["sql"]["sql"] == "SELECT 1"
    assert {s["name"] for s in lineage["steps"]} >= {"classify_intent", "synthesize"}


async def test_sql_metadata_forwarded_to_client():
    events = await _run(
        {"type": "metadata", "sql_query": "SELECT 1", "explanation": "e",
         "row_count": 1, "sources": ["income"], "result_preview": []},
        ("ctx", []),
    )
    forwarded = [e for e in events if e["type"] == "metadata" and e.get("sql_query")]
    assert len(forwarded) == 1
    assert "result_preview" not in forwarded[0]


async def test_no_sql_metadata_forwarded_without_sql_query():
    events = await _run(
        {"type": "metadata", "result_preview": [], "row_count": 0, "sources": []},
        ("ctx", []),
    )
    assert not [e for e in events if e["type"] == "metadata" and e.get("sql_query")]
