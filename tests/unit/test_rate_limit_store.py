"""DOCBOT-1606: durable fixed-window counters (SQLite stands in for Postgres)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

import api.index as idx
from api.utils import request_guard


@pytest.fixture
async def db_url(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'counters.db'}"
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(idx.rate_limit_counters_table.create)
    await eng.dispose()
    return url


@pytest.fixture
async def engines(db_url):
    """Two independent engines on one DB file = two app processes."""
    a, b = create_async_engine(db_url), create_async_engine(db_url)
    yield a, b
    await a.dispose()
    await b.dispose()
    request_guard.reset_state()


def _req(ip="203.0.113.7"):
    return SimpleNamespace(headers={}, cookies={}, client=SimpleNamespace(host=ip))


async def test_limit_shared_across_two_processes(engines, monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 3)
    proc_a, proc_b = (request_guard.SqlCounterStore(e) for e in engines)

    request_guard.use_counter_store(proc_a)
    await request_guard.check_ip_rate_limit(_req())
    await request_guard.check_ip_rate_limit(_req())
    request_guard.use_counter_store(proc_b)
    await request_guard.check_ip_rate_limit(_req())
    with pytest.raises(HTTPException) as exc:
        await request_guard.check_ip_rate_limit(_req())
    assert exc.value.status_code == 429 and int(exc.value.headers["Retry-After"]) >= 1
    request_guard.use_counter_store(proc_a)
    with pytest.raises(HTTPException):
        await request_guard.check_ip_rate_limit(_req())
    await request_guard.check_ip_rate_limit(_req(ip="198.51.100.1"))  # other IP unaffected


async def test_counts_survive_redeploy(db_url, monkeypatch):
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 5)
    first = create_async_engine(db_url)
    request_guard.use_counter_store(request_guard.SqlCounterStore(first))
    await request_guard.enforce_user_budget("u1", "autopilot")  # weight 5
    await first.dispose()  # "redeploy"

    second = create_async_engine(db_url)
    request_guard.use_counter_store(request_guard.SqlCounterStore(second))
    try:
        with pytest.raises(HTTPException):
            await request_guard.enforce_user_budget("u1", "chat")
    finally:
        await second.dispose()
        request_guard.reset_state()


async def test_rejected_charge_is_refunded(engines, monkeypatch):
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 5)
    request_guard.use_counter_store(request_guard.SqlCounterStore(engines[0]))
    await request_guard.enforce_session_budget("s1", "chat")  # 1
    with pytest.raises(HTTPException):
        await request_guard.enforce_session_budget("s1", "autopilot")  # 1+5 > 5, refunded
    for _ in range(4):
        await request_guard.enforce_session_budget("s1", "chat")  # 2..5 still fit


async def test_concurrent_charges_are_atomic(engines):
    request_guard.use_counter_store(request_guard.SqlCounterStore(engines[0]))
    results = await asyncio.gather(*(request_guard._charge("t", "k", 3600, 10) for _ in range(25)))
    assert sum(r is None for r in results) == 10


async def test_cleanup_deletes_expired_windows(engines, monkeypatch):
    store = request_guard.SqlCounterStore(engines[0])
    request_guard.use_counter_store(store)
    clock = [3600.0 * 10]
    monkeypatch.setattr(request_guard, "time", SimpleNamespace(time=lambda: clock[0]))
    await request_guard._charge("t", "old", 60, 100)
    clock[0] += 3600
    await store.cleanup(int(clock[0]))
    async with engines[0].connect() as conn:
        rows = (await conn.execute(text("SELECT counter_key FROM rate_limit_counters"))).all()
    assert rows == []


class _BrokenStore:
    async def incr(self, *_a, **_k):
        raise OperationalError("UPSERT", {}, Exception("db down"))

    async def cleanup(self, now):
        raise OperationalError("DELETE", {}, Exception("db down"))


class _HangingStore:
    async def incr(self, *_a, **_k):
        await asyncio.sleep(60)

    async def cleanup(self, now):
        return None


@pytest.mark.parametrize("store", [_BrokenStore(), _HangingStore()], ids=["error", "timeout"])
async def test_fail_open_on_store_failure(store, monkeypatch, caplog):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 1)
    monkeypatch.setattr(request_guard, "_STORE_TIMEOUT_SEC", 0.05)
    request_guard.use_counter_store(store)
    try:
        with caplog.at_level(logging.WARNING, logger="api.utils.request_guard"):
            for _ in range(3):
                await request_guard.check_ip_rate_limit(_req())  # never raises
        assert "failing open" in caplog.text
    finally:
        request_guard.reset_state()


def test_defaults_are_non_zero():
    assert request_guard.RATE_LIMIT_PER_MIN > 0
    assert request_guard.SESSION_BUDGET > 0
    assert request_guard.USER_DAILY_BUDGET > 0
    assert request_guard.TRUSTED_PROXY_HOPS >= 1
