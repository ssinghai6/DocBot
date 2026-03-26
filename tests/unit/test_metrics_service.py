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
