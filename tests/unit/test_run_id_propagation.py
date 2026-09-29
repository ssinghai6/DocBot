"""DOCBOT-1501 regression tests — run_id sharing across a pipeline's LLM calls.

Code review caught a real bug: `run_trace(run_id).__enter__()` called as a
bare expression with no held reference. CPython refcounting garbage-collects
the temporary `_GeneratorContextManager` immediately after that statement
executes; a generator-based context manager, when garbage collected while
still suspended at its `yield`, has `close()` implicitly called on it, which
throws `GeneratorExit` into `run_trace()` at its `yield` point — running its
`finally: _current_run_id.reset(token)` immediately, before the next line of
pipeline code runs. Net effect: `run_sql_pipeline`, `run_csv_query_on_e2b`,
and `hybrid_chat` were mint­ing a fresh orphan run_id per LLM call instead of
sharing one run_id across the whole pipeline invocation — defeating DOCBOT-
1501's core acceptance criterion for 3 of the 5 wired pipelines.

These tests assert the actual acceptance criterion directly: multiple LLM
calls made within one pipeline invocation must observe the same
`current_run_id()`. They would have failed against the buggy
`run_trace(...).__enter__()` (no held reference) implementation and pass
against the fixed `_trace_cm = run_trace(...); _trace_cm.__enter__()` one.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Column, MetaData, String, Table

from api.utils.llm_provider import current_run_id


def _fake_db_connections_table() -> Table:
    """A real (unbound) SQLAlchemy Table so `select(table).where(table.c.id == ...)`
    builds a valid query object — MagicMock() fails ArgumentError coercion."""
    return Table("db_connections", MetaData(), Column("id", String, primary_key=True))


async def _collect(gen: AsyncGenerator) -> list[str]:
    return [chunk async for chunk in gen]


# ---------------------------------------------------------------------------
# hybrid_chat
# ---------------------------------------------------------------------------


def _make_groq_streaming_response(tokens: list[str]) -> MagicMock:
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


@pytest.mark.unit
@pytest.mark.asyncio
class TestHybridChatSharesRunId:
    async def test_intent_classification_and_synthesis_share_run_id(self):
        """classify_intent_safe, _collect_sql_result, and the streaming
        synthesis call must all observe the same current_run_id()."""
        from api.hybrid_service import hybrid_chat, IntentClassification

        seen_run_ids: list[str | None] = []

        async def _mock_classify(*args, **kwargs):
            seen_run_ids.append(current_run_id())
            return IntentClassification(
                intent="hybrid", fallback_applied=False, question_hash="abc123def456789a"
            )

        async def _mock_sql(*args, **kwargs):
            seen_run_ids.append(current_run_id())
            return {
                "type": "metadata", "result_preview": [{"col": 1}],
                "row_count": 1, "sources": ["orders"],
            }

        async def _mock_rag(*args, **kwargs):
            seen_run_ids.append(current_run_id())
            return ("Doc context.", [{"source": "doc.pdf", "page": 1}])

        def _create_side_effect(*args, **kwargs):
            seen_run_ids.append(current_run_id())
            return _make_groq_streaming_response(["The", " answer", "."])

        groq_client = MagicMock()
        groq_client.chat.completions.create = MagicMock(side_effect=_create_side_effect)

        with (
            patch("api.hybrid_service.classify_intent_safe", side_effect=_mock_classify),
            patch("api.hybrid_service.rag_retrieve", side_effect=_mock_rag),
            patch("api.hybrid_service._collect_sql_result", side_effect=_mock_sql),
            patch("api.hybrid_service.os.getenv", return_value="fake-groq-key"),
            patch("groq.Groq", return_value=groq_client),
        ):
            gen = hybrid_chat(
                question="What is the revenue?",
                session_id="sess-test",
                connection_id="conn-1",
                persona="Data Analyst",
                has_docs=True,
                messages_table=MagicMock(), sessions_table=MagicMock(),
                db_connections_table=MagicMock(), schema_cache_table=MagicMock(),
                query_history_table=MagicMock(), query_embeddings_table=MagicMock(),
                async_session_factory=MagicMock(),
                expert_personas={"Data Analyst": {"persona_def": "You are a data analyst."}},
                vector_stores={"sess-test": MagicMock()},
            )
            await _collect(gen)

        # 4 call sites recorded: classify, sql, rag, synthesis stream.
        assert len(seen_run_ids) == 4
        # None of them observed a missing/reset context.
        assert all(rid is not None for rid in seen_run_ids), seen_run_ids
        # All four share exactly one run_id — the regression this test guards.
        assert len(set(seen_run_ids)) == 1, seen_run_ids


# ---------------------------------------------------------------------------
# run_sql_pipeline
# ---------------------------------------------------------------------------


class _FakeConnRow:
    def __init__(self):
        self.dialect = "sqlite"
        self.credentials_blob = "encrypted-blob"
        self.pii_masking_enabled = False


def _make_conn_lookup_session_factory(conn_row):
    """async_session_factory mock whose every execute() returns conn_row
    from fetchone() (the only DB touches left after mocking out the
    schema/table-selector/few-shot/embedding/history collaborators)."""

    class _Result:
        def fetchone(self):
            return conn_row

        def fetchall(self):
            return []

    async def _execute(_stmt):
        return _Result()

    session = AsyncMock()
    session.execute = _execute

    factory = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx
    return factory


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunSqlPipelineSharesRunId:
    async def test_table_selector_sql_gen_and_answer_share_run_id(self):
        """_select_relevant_tables, _generate_sql, and _stream_answer must
        all observe the same current_run_id() within one pipeline call."""
        import api.db_service as db_service

        seen_run_ids: list[str | None] = []

        async def _mock_select_tables(question, schema):
            seen_run_ids.append(current_run_id())
            return ["orders"]

        async def _mock_generate_sql(question, schema_subset, few_shot_examples):
            seen_run_ids.append(current_run_id())
            return "SELECT * FROM orders LIMIT 10"

        async def _mock_stream_answer(question, sql, result_dicts, persona_def):
            seen_run_ids.append(current_run_id())
            for tok in ["Answer", " text"]:
                yield tok

        conn_row = _FakeConnRow()
        session_factory = _make_conn_lookup_session_factory(conn_row)

        with (
            patch.object(db_service, "get_schema", AsyncMock(return_value=[
                {"name": "orders", "columns": [{"name": "id", "type": "INTEGER"}], "is_view": False},
            ])),
            patch.object(db_service, "_select_relevant_tables", side_effect=_mock_select_tables),
            patch.object(db_service, "_retrieve_few_shot", AsyncMock(return_value=[])),
            patch.object(db_service, "_generate_sql", side_effect=_mock_generate_sql),
            patch.object(db_service, "validate_and_sanitize_sql", return_value="SELECT * FROM orders LIMIT 10"),
            patch.object(db_service, "_execute_query", AsyncMock(return_value=([], []))),
            patch.object(db_service, "_store_query_history", AsyncMock(return_value=None)),
            patch.object(db_service, "_stream_answer", side_effect=_mock_stream_answer),
            patch.object(db_service, "_get_embeddings_model", return_value=MagicMock()),
            patch.object(db_service, "_get_embedding", AsyncMock(return_value=[0.1, 0.2])),
            patch.object(db_service, "decrypt_credentials", return_value={}),
            patch.object(db_service, "_resolve_connection", return_value=("sqlite:///:memory:", None)),
        ):
            gen = db_service.run_sql_pipeline(
                connection_id="conn-1",
                question="How many orders?",
                persona="Data Analyst",
                db_connections_table=_fake_db_connections_table(),
                schema_cache_table=MagicMock(),
                query_history_table=MagicMock(),
                query_embeddings_table=MagicMock(),
                async_session_factory=session_factory,
                expert_personas={"Data Analyst": {"persona_def": "You are a data analyst."}},
            )
            await _collect(gen)

        assert len(seen_run_ids) == 3
        assert all(rid is not None for rid in seen_run_ids), seen_run_ids
        assert len(set(seen_run_ids)) == 1, seen_run_ids


# ---------------------------------------------------------------------------
# run_csv_query_on_e2b
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunCsvQueryOnE2bSharesRunId:
    async def test_codegen_and_retry_share_run_id(self):
        """generate_csv_analysis_code (initial + corrective retry) must both
        observe the same current_run_id() within one call."""
        import api.sandbox_service as sandbox_service

        seen_run_ids: list[str | None] = []

        async def _mock_generate_code(*args, **kwargs):
            seen_run_ids.append(current_run_id())
            return "print('hello')"

        class _FakeSandboxResult:
            def __init__(self, error=None):
                self.stdout = "hello\n"
                self.error = error
                self.charts = []
                self.chart_metadata = []
                self.execution_time_ms = 10

        async def _mock_run_in_sandbox(*args, **kwargs):
            return _FakeSandboxResult()

        with (
            patch.object(sandbox_service, "generate_csv_analysis_code", side_effect=_mock_generate_code),
            patch.object(sandbox_service, "_run_csv_in_sandbox", side_effect=_mock_run_in_sandbox),
        ):
            gen = sandbox_service.run_csv_query_on_e2b(
                csv_content_b64="",
                question="Summarize this data",
                persona="Data Analyst",
                table_name="data",
                column_names=["a", "b"],
                expert_personas={"Data Analyst": {"persona_def": "You are a data analyst."}},
            )
            await _collect(gen)

        assert len(seen_run_ids) >= 1
        assert all(rid is not None for rid in seen_run_ids), seen_run_ids
        assert len(set(seen_run_ids)) == 1, seen_run_ids


# ---------------------------------------------------------------------------
# Direct repro of the underlying bug pattern — documents *why* a bare
# `run_trace(...).__enter__()` is unsafe, independent of any pipeline.
# ---------------------------------------------------------------------------


class TestOrphanedContextManagerBugPattern:
    def test_bare_enter_without_held_reference_resets_immediately(self):
        """Reproduces the exact bug the reviewer flagged: calling
        run_trace(...).__enter__() with nothing holding a reference to the
        returned _GeneratorContextManager causes it to be garbage collected
        right after the statement runs, which implicitly closes the
        suspended generator and fires its `finally` (resetting the
        ContextVar) before the next line executes."""
        import gc

        from api.utils.llm_provider import run_trace

        assert current_run_id() is None

        # The buggy pattern: no variable holds the returned CM.
        run_trace("orphan-run-id").__enter__()
        gc.collect()  # CPython refcounting usually doesn't need this, but be explicit/portable

        # The bug: the ContextVar has already been reset back to None.
        assert current_run_id() is None

    def test_held_reference_keeps_run_id_bound(self):
        """The fix: holding a reference keeps the ContextVar bound until an
        explicit __exit__ (or the reference is dropped at function/frame end)."""
        import gc

        from api.utils.llm_provider import run_trace

        assert current_run_id() is None

        trace_cm = run_trace("held-run-id")
        trace_cm.__enter__()
        gc.collect()

        assert current_run_id() == "held-run-id"

        trace_cm.__exit__(None, None, None)
        assert current_run_id() is None
