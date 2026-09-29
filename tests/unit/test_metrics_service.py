"""Unit tests for api.metrics_service — DOCBOT admin metrics.

All DB calls are mocked. No network or database access required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock DB session
# ---------------------------------------------------------------------------


class MockRow:
    """Simulates a SQLAlchemy row with index access."""
    def __init__(self, *values):
        self._values = values

    def __getitem__(self, idx):
        return self._values[idx]


class MockResult:
    """Simulates a SQLAlchemy result from execute()."""
    def __init__(self, scalar_value=None, rows=None):
        self._scalar = scalar_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._rows


def _make_session_factory(execute_results: list[MockResult]):
    """Create a mock async_session_factory that returns execute_results in order."""
    call_idx = 0

    async def mock_execute(_stmt):
        nonlocal call_idx
        if call_idx < len(execute_results):
            result = execute_results[call_idx]
            call_idx += 1
            return result
        return MockResult(scalar_value=0, rows=[])

    session = AsyncMock()
    session.execute = mock_execute

    # Support async context manager
    factory = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_platform_metrics_basic():
    """Metrics returns expected structure with all zero counts."""
    from api.metrics_service import get_platform_metrics

    # Sequence: sessions, queries, query metadata, uploads, db_connections, response_time
    results = [
        MockResult(scalar_value=5),
        MockResult(scalar_value=0),
        MockResult(rows=[]),
        MockResult(scalar_value=3),
        MockResult(scalar_value=2),
        MockResult(rows=[]),
    ]

    factory = _make_session_factory(results)

    metrics = await get_platform_metrics(async_session_factory=factory)

    assert metrics["total_sessions"] == 5
    assert metrics["total_queries"] == 0
    assert metrics["total_documents_uploaded"] == 3
    assert metrics["active_db_connections"] == 2
    assert metrics["avg_response_time_ms"] is None
    assert "uptime_seconds" in metrics
    assert metrics["uptime_seconds"] >= 0
    assert "uptime_since" in metrics
    assert "queries_by_type" in metrics


@pytest.mark.asyncio
async def test_get_platform_metrics_with_queries():
    """Metrics correctly counts queries by type from metadata_json."""
    from api.metrics_service import get_platform_metrics

    query_rows = [
        MockRow(json.dumps({"query_type": "doc"})),
        MockRow(json.dumps({"query_type": "doc"})),
        MockRow(json.dumps({"query_type": "db"})),
        MockRow(json.dumps({"query_type": "hybrid"})),
        MockRow(json.dumps({"query_type": "csv"})),
        MockRow(None),
    ]

    response_time_rows = [
        MockRow(json.dumps({"response_time_ms": 100})),
        MockRow(json.dumps({"response_time_ms": 200})),
        MockRow(json.dumps({"response_time_ms": 300})),
    ]

    results = [
        MockResult(scalar_value=10),
        MockResult(scalar_value=6),
        MockResult(rows=query_rows),
        MockResult(scalar_value=7),
        MockResult(scalar_value=1),
        MockResult(rows=response_time_rows),
    ]

    factory = _make_session_factory(results)

    metrics = await get_platform_metrics(async_session_factory=factory)

    assert metrics["total_sessions"] == 10
    assert metrics["total_queries"] == 6
    assert metrics["queries_by_type"]["doc"] == 2
    assert metrics["queries_by_type"]["db"] == 1
    assert metrics["queries_by_type"]["hybrid"] == 1
    assert metrics["queries_by_type"]["csv"] == 1
    assert metrics["queries_by_type"]["unknown"] == 1
    assert metrics["total_documents_uploaded"] == 7
    assert metrics["active_db_connections"] == 1
    assert metrics["avg_response_time_ms"] == 200.0


@pytest.mark.asyncio
async def test_get_platform_metrics_malformed_metadata():
    """Metrics handles malformed JSON in metadata_json gracefully."""
    from api.metrics_service import get_platform_metrics

    query_rows = [
        MockRow("not valid json"),
        MockRow(json.dumps({"query_type": "doc"})),
        MockRow(""),
    ]

    results = [
        MockResult(scalar_value=1),
        MockResult(scalar_value=3),
        MockResult(rows=query_rows),
        MockResult(scalar_value=0),
        MockResult(scalar_value=0),
        MockResult(rows=[MockRow("bad json")]),
    ]

    factory = _make_session_factory(results)

    metrics = await get_platform_metrics(async_session_factory=factory)

    assert metrics["queries_by_type"]["unknown"] == 2
    assert metrics["queries_by_type"]["doc"] == 1
    assert metrics["avg_response_time_ms"] is None


@pytest.mark.asyncio
async def test_get_platform_metrics_elapsed_ms_key():
    """Metrics uses 'elapsed_ms' as fallback key for response time."""
    from api.metrics_service import get_platform_metrics

    response_time_rows = [
        MockRow(json.dumps({"elapsed_ms": 150})),
        MockRow(json.dumps({"elapsed_ms": 250})),
    ]

    results = [
        MockResult(scalar_value=0),
        MockResult(scalar_value=0),
        MockResult(rows=[]),
        MockResult(scalar_value=0),
        MockResult(scalar_value=0),
        MockResult(rows=response_time_rows),
    ]

    factory = _make_session_factory(results)

    metrics = await get_platform_metrics(async_session_factory=factory)

    assert metrics["avg_response_time_ms"] == 200.0


@pytest.mark.asyncio
async def test_get_platform_metrics_includes_llm_metrics_block():
    """Metrics response always carries an llm_metrics key (DOCBOT-1502),
    even when the trace store has no data — degrades gracefully instead
    of omitting the field."""
    from api.metrics_service import get_platform_metrics

    results = [
        MockResult(scalar_value=0),
        MockResult(scalar_value=0),
        MockResult(rows=[]),
        MockResult(scalar_value=0),
        MockResult(scalar_value=0),
        MockResult(rows=[]),
    ]
    factory = _make_session_factory(results)

    metrics = await get_platform_metrics(async_session_factory=factory)

    assert "llm_metrics" in metrics
    assert metrics["llm_metrics"]["total_calls"] == 0
    assert metrics["llm_metrics"]["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_get_platform_metrics_llm_failure_degrades_gracefully(monkeypatch):
    """If get_llm_cost_metrics blows up, the whole /admin/metrics response
    must not fail — llm_metrics degrades to None instead."""
    import api.metrics_service as metrics_module

    async def _boom(days=7):
        raise RuntimeError("trace store unavailable")

    monkeypatch.setattr(metrics_module, "get_llm_cost_metrics", _boom)

    results = [
        MockResult(scalar_value=1),
        MockResult(scalar_value=0),
        MockResult(rows=[]),
        MockResult(scalar_value=0),
        MockResult(scalar_value=0),
        MockResult(rows=[]),
    ]
    factory = _make_session_factory(results)

    metrics = await metrics_module.get_platform_metrics(async_session_factory=factory)

    assert metrics["total_sessions"] == 1
    assert metrics["llm_metrics"] is None


# ---------------------------------------------------------------------------
# DOCBOT-1502: get_llm_cost_metrics
# ---------------------------------------------------------------------------


class TestGetLlmCostMetrics:
    @pytest.mark.asyncio
    async def test_empty_call_log(self, monkeypatch):
        from api import metrics_service

        async def _empty_stats(since=None):
            return []

        monkeypatch.setattr("api.llm_trace_service.get_call_stats", _empty_stats)

        result = await metrics_service.get_llm_cost_metrics(days=7)

        assert result.total_calls == 0
        assert result.total_cost_usd == 0.0
        assert result.by_day == []
        assert result.by_caller == []
        assert result.p50_latency_ms is None

    @pytest.mark.asyncio
    async def test_aggregates_cost_tokens_and_latency(self, monkeypatch):
        from api import metrics_service

        rows = [
            {
                "run_id": "r1", "provider": "groq", "model": "openai/gpt-oss-20b",
                "caller": "sql_gen", "latency_ms": 100.0, "input_tokens": 50,
                "output_tokens": 20, "estimated_cost_usd": 0.001,
                "success": True, "fallback_triggered": False,
                "created_at": "2026-09-20T10:00:00+00:00",
            },
            {
                "run_id": "r1", "provider": "groq", "model": "openai/gpt-oss-20b",
                "caller": "sql_gen", "latency_ms": 200.0, "input_tokens": 30,
                "output_tokens": 10, "estimated_cost_usd": 0.0005,
                "success": True, "fallback_triggered": False,
                "created_at": "2026-09-20T11:00:00+00:00",
            },
            {
                "run_id": "r2", "provider": "gemini", "model": "gemini-2.5-flash",
                "caller": "hybrid_synthesis", "latency_ms": 500.0, "input_tokens": 100,
                "output_tokens": 200, "estimated_cost_usd": 0.002,
                "success": False, "fallback_triggered": True,
                "created_at": "2026-09-21T09:00:00+00:00",
            },
        ]

        async def _mock_stats(since=None):
            return rows

        monkeypatch.setattr("api.llm_trace_service.get_call_stats", _mock_stats)

        result = await metrics_service.get_llm_cost_metrics(days=7)

        assert result.total_calls == 3
        assert result.total_input_tokens == 180
        assert result.total_output_tokens == 230
        assert round(result.total_cost_usd, 4) == round(0.001 + 0.0005 + 0.002, 4)
        assert result.success_rate == round(2 / 3, 4)
        assert result.fallback_rate == round(1 / 3, 4)
        assert result.p50_latency_ms is not None
        assert result.p95_latency_ms is not None

        by_day = {d.date: d for d in result.by_day}
        assert by_day["2026-09-20"].calls == 2
        assert by_day["2026-09-21"].calls == 1

        by_caller = {c.caller: c for c in result.by_caller}
        assert by_caller["sql_gen"].calls == 2
        assert by_caller["hybrid_synthesis"].calls == 1
        assert by_caller["hybrid_synthesis"].p50_latency_ms == 500.0

    @pytest.mark.asyncio
    async def test_missing_optional_fields_do_not_crash(self, monkeypatch):
        from api import metrics_service

        rows = [
            {"run_id": "r1", "provider": "groq", "model": "m", "caller": None,
             "latency_ms": None, "input_tokens": None, "output_tokens": None,
             "estimated_cost_usd": None, "success": True, "fallback_triggered": False,
             "created_at": None},
        ]

        async def _mock_stats(since=None):
            return rows

        monkeypatch.setattr("api.llm_trace_service.get_call_stats", _mock_stats)

        result = await metrics_service.get_llm_cost_metrics(days=7)
        assert result.total_calls == 1
        assert result.total_cost_usd == 0.0
        assert result.by_caller[0].caller == "unknown"
