"""DOCBOT-2728 — intent router service: regex fallback, output parsing, cache,
timeout fallback, and the ownership-scoped source summary. No network."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError
from sqlalchemy import create_engine, insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import os

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

import api.index as idx  # noqa: E402
import api.intent_service as svc  # noqa: E402
from api.utils.caller_scope import CallerScope  # noqa: E402

GOOD = {"route": "autopilot", "source": "sql", "task": "forecast", "data_ready": True, "missing": None, "confidence": 0.9}


@pytest.fixture(autouse=True)
def _clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


# ── regex fallback (port of the old frontend trigger) ──

@pytest.mark.parametrize("q,route,task", [
    ("plot revenue by month", "fast", "viz"),
    ("Show me a forecast of revenue", "fast", "viz"),  # SIMPLE_VIZ wins, like the frontend
    ("predict sales for next year", "autopilot", "forecast"),
    ("what if CAC rises 20%", "autopilot", "what_if"),
    ("find the root cause of churn", "autopilot", "diagnose"),
    ("compare Q1 revenue vs Q2 revenue", "autopilot", "compare"),
    ("where will diesel be in 2030?", "fast", "lookup"),  # the blind spot 2728 fixes
    ("what was revenue in 2024?", "fast", "lookup"),
])
def test_regex_route(q: str, route: str, task: str) -> None:
    r = svc.regex_route(q, has_doc=True, has_db=False)
    assert (r.route, r.task, r.confidence, r.data_ready, r.source) == (route, task, 0.0, True, "doc")


def test_regex_route_needs_a_source() -> None:
    assert svc.regex_route("forecast revenue", False, False).route == "fast"
    assert svc.regex_route("forecast revenue", True, True).source == "hybrid"


# ── LLM output parsing ──

def test_parse_plain_and_fenced_json() -> None:
    assert svc.parse_llm_output(json.dumps(GOOD)).task == "forecast"
    fenced = "```json\n" + json.dumps({**GOOD, "data_ready": False, "missing": "no series"}) + "\n```"
    r = svc.parse_llm_output(fenced)
    assert (r.data_ready, r.missing) == (False, "no series")


def test_parse_drops_missing_when_ready() -> None:
    assert svc.parse_llm_output(json.dumps({**GOOD, "missing": "stray"})).missing is None


@pytest.mark.parametrize("bad", [
    "no json here",
    json.dumps({**GOOD, "route": "slow"}),
    json.dumps({**GOOD, "task": "predict"}),
    json.dumps({**GOOD, "confidence": 3}),
    json.dumps({k: v for k, v in GOOD.items() if k != "data_ready"}),
])
def test_parse_rejects_bad_output(bad: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        svc.parse_llm_output(bad)


# ── classify(): LLM path, cache, fallback ──

async def test_classify_uses_llm_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(svc, "_call_llm", lambda m: calls.append(m) or json.dumps(GOOD))
    a = await svc.classify("where will diesel be in 2030?", [], "CSV file: data(year, price)", False, True)
    b = await svc.classify("where will diesel be in 2030?", [], "CSV file: data(year, price)", False, True)
    assert a.route == b.route == "autopilot" and a.confidence == 0.9
    assert len(calls) == 1
    assert "data(year, price)" in calls[0][1]["content"]


async def test_classify_history_is_last_three_turns_six_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(svc, "_call_llm", lambda m: seen.append(m) or json.dumps(GOOD))
    hist = [{"role": "user", "content": f"msg{i}"} for i in range(9)]
    await svc.classify("and next year?", hist, "Documents: a.pdf", True, False)
    prompt = seen[0][1]["content"]
    assert "msg2" not in prompt and all(f"msg{i}" in prompt for i in range(3, 9))


async def test_classify_falls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_m):
        raise RuntimeError("provider down")
    monkeypatch.setattr(svc, "_call_llm", boom)
    r = await svc.classify("predict revenue", [], "Documents: a.pdf", True, False)
    assert (r.route, r.confidence) == ("autopilot", 0.0)
    assert not svc._cache  # failures are not cached


async def test_classify_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "_call_llm", lambda m: "sure! route it to autopilot")
    r = await svc.classify("what was revenue?", [], "Documents: a.pdf", True, False)
    assert (r.route, r.confidence) == ("fast", 0.0)


async def test_classify_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "TIMEOUT_S", 0.05)
    monkeypatch.setattr(svc, "_call_llm", lambda m: time.sleep(0.3) or json.dumps(GOOD))
    r = await svc.classify("where will diesel be in 2030?", [], "Documents: a.pdf", True, False)
    assert r.confidence == 0.0


async def test_classify_without_sources_never_autopilot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "_call_llm", lambda m: json.dumps(GOOD))
    assert (await svc.classify("forecast revenue", [], "No sources attached.", False, False)).route == "fast"


def test_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "_CACHE_MAX", 3)
    r = svc.regex_route("x", True, False)
    for i in range(5):
        svc._cache_put(str(i), r)
    assert list(svc._cache) == ["2", "3", "4"]


# ── source summary: names only, caller-scoped ──

@pytest.fixture
def db(tmp_path: Path):
    path = tmp_path / "intent.db"
    eng = create_engine(f"sqlite:///{path}")
    tables = [idx.sessions_table, idx.db_connections_table, idx.schema_cache_table]
    idx.metadata.create_all(eng, tables=tables)
    with eng.begin() as c:
        c.execute(insert(idx.sessions_table).values(
            session_id="s-ann", user_id="ann", org_id="o1",
            files_info=json.dumps([{"filename": "fuelprice.pdf", "pages": 3, "size": 10}]),
        ))
        c.execute(insert(idx.sessions_table).values(
            session_id="s-bob", user_id="bob", org_id="o1",
            files_info=json.dumps([{"filename": "secret_plan.pdf"}]),
        ))
        c.execute(insert(idx.db_connections_table).values(
            id="c-ann", session_id="s-ann", dialect="csv", host="", port=0, dbname="prices",
            credentials_blob="ENCRYPTED-SECRET", user_id="ann", org_id="o1",
        ))
        c.execute(insert(idx.schema_cache_table).values(
            connection_id="c-ann", expires_at=idx.func.now(),
            schema_json=json.dumps([{"name": "data", "columns": [{"name": "year", "type": "TEXT"}, {"name": "diesel", "type": "TEXT"}]}]),
        ))
    factory = async_sessionmaker(create_async_engine(f"sqlite+aiosqlite:///{path}"), expire_on_commit=False)
    yield factory
    eng.dispose()


def _tables():
    return idx.sessions_table, idx.db_connections_table, idx.schema_cache_table


async def test_summary_for_owner(db) -> None:
    ann = CallerScope(user_id="ann", org_id="o1", anon_token=None)
    summary, has_doc, has_db = await svc.build_source_summary("s-ann", "c-ann", ann, *_tables(), db)
    assert (has_doc, has_db) == (True, True)
    assert "fuelprice.pdf" in summary and "data(year TEXT, diesel TEXT)" in summary and "CSV file" in summary
    assert "ENCRYPTED-SECRET" not in summary and "prices" not in summary


async def test_summary_ignores_foreign_ids(db) -> None:
    ann = CallerScope(user_id="ann", org_id="o1", anon_token=None)
    summary, has_doc, has_db = await svc.build_source_summary("s-bob", "c-missing", ann, *_tables(), db)
    assert (has_doc, has_db) == (False, False)
    assert "secret_plan" not in summary and summary == "No sources attached."
