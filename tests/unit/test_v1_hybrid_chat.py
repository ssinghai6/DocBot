"""DOCBOT-2592: /v1/hybrid/chat folds the hybrid SSE stream correctly.

No network: auth, resource scoping, budget, billing, audit and the hybrid
pipeline itself are all mocked.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import api.index as idx

HEADERS = {"Authorization": "Bearer dk_test"}
BODY = {"question": "q", "session_id": "s-1"}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _fake_key(request: Request):
    request.state.org_id = "org-key"
    return SimpleNamespace(id="key-1", org_id="org-key")


def _client(monkeypatch, events: list[dict]):
    async def fake_hybrid(**_kw):
        for e in events:
            yield _sse(e)

    overrides = {idx._v1_hybrid_read.dependency: _fake_key}
    idx.app.dependency_overrides.update(overrides)
    monkeypatch.setattr(idx, "_require_v1_resources", AsyncMock())
    monkeypatch.setattr(idx, "charge_budget", AsyncMock())
    monkeypatch.setattr(idx, "_begin_v1_billing", AsyncMock(return_value=nullcontext()))
    with patch("api.audit_service.log_event", MagicMock()), \
         patch("api.hybrid_service.hybrid_chat", fake_hybrid):
        resp = TestClient(idx.app, raise_server_exceptions=False).post(
            "/v1/hybrid/chat", json=BODY, headers=HEADERS,
        )
    for dep in overrides:
        idx.app.dependency_overrides.pop(dep, None)
    assert resp.status_code == 200, resp.text
    return resp.json()


DONE = {"type": "done", "citations": [{"page": 1}], "run_id": "r-1", "trace_id": "t-1"}
SQL_META = {"type": "metadata", "sql_query": "SELECT 1", "row_count": 1, "sources": ["t"]}


def test_replace_event_returns_corrected_answer_and_keeps_marker_tail(monkeypatch):
    body = _client(monkeypatch, [
        {"type": "metadata", "intent": "hybrid", "has_sql": True, "has_docs": True},
        SQL_META,
        {"type": "token", "content": "Net income was "},
        {"type": "token", "content": "$325M."},
        {"type": "replace", "content": "Net income was $325M (DB: $330M) [[discrepancy:d1]]."},
        {"type": "token", "content": "\n[[discrepancy:d2]]"},
        DONE,
    ])
    assert body["answer"] == "Net income was $325M (DB: $330M) [[discrepancy:d1]].\n[[discrepancy:d2]]"
    assert body["citations"] == [{"page": 1}]
    assert body["run_id"] == "r-1" and body["trace_id"] == "t-1"


def test_no_replace_returns_streamed_answer(monkeypatch):
    body = _client(monkeypatch, [
        {"type": "metadata", "intent": "doc", "has_sql": False, "has_docs": True},
        {"type": "token", "content": "Hello "},
        {"type": "token", "content": "world"},
        DONE,
    ])
    assert body["answer"] == "Hello world"


@pytest.mark.parametrize(
    ("intent", "has_sql", "has_docs", "extra"),
    [
        ("doc", False, True, []),
        ("sql", True, False, [SQL_META]),
        ("hybrid", True, True, [SQL_META]),
    ],
)
def test_intent_flags_survive_sql_metadata_event(monkeypatch, intent, has_sql, has_docs, extra):
    body = _client(monkeypatch, [
        {"type": "metadata", "intent": intent, "has_sql": has_sql, "has_docs": has_docs},
        *extra,
        {"type": "token", "content": "a"},
        DONE,
    ])
    assert (body["intent"], body["has_sql"], body["has_docs"]) == (intent, has_sql, has_docs)
    # Response shape unchanged.
    assert set(body) == {"answer", "citations", "intent", "has_sql", "has_docs", "run_id", "trace_id"}
