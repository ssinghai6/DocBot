"""DOCBOT-1623: demo-init + EDGAR rate limits, usage metering, batch cap, LRU wiring."""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.index as idx
from api import usage_service
from api.utils import request_guard
from api.utils.caller_scope import CallerScope


class _FakeRequest:
    def __init__(self, ip="1.2.3.4"):
        self.headers = {}
        self.query_params = {}
        self.cookies = {}
        self.client = SimpleNamespace(host=ip)
        self.state = SimpleNamespace()


def _drain() -> list:
    rows = []
    while True:
        try:
            rows.append(usage_service._queue.get_nowait())
        except queue.Empty:
            return rows


def _as_user(monkeypatch, user_id="u-1", org_id="o-1"):
    async def _scope(_request):
        return CallerScope(user_id=user_id, org_id=org_id, anon_token=None)
    monkeypatch.setattr(request_guard, "_caller_scope", _scope)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 0)
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 0)
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", False)
    request_guard.reset_state()
    _drain()
    yield
    request_guard.reset_state()
    _drain()


@pytest.fixture
def client():
    return TestClient(idx.app, raise_server_exceptions=False)


class TestDemoInit:
    async def test_per_user_limit_across_ips(self, monkeypatch):
        _as_user(monkeypatch)
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 100)
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_USER_PER_HOUR", 2)
        for i in range(2):
            await request_guard.check_demo_init_limit(_FakeRequest(ip=f"10.0.0.{i}"))
        with pytest.raises(HTTPException) as exc:
            await request_guard.check_demo_init_limit(_FakeRequest(ip="10.0.0.9"))
        assert exc.value.status_code == 429
        assert "per account" in exc.value.detail

    async def test_user_rejection_refunds_ip_slot(self, monkeypatch):
        _as_user(monkeypatch)
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 2)
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_USER_PER_HOUR", 1)
        req = _FakeRequest(ip="10.0.0.1")
        await request_guard.check_demo_init_limit(req)
        with pytest.raises(HTTPException):
            await request_guard.check_demo_init_limit(req)
        _as_user(monkeypatch, user_id="u-2")
        await request_guard.check_demo_init_limit(req)  # IP slot was refunded

    async def test_admitted_call_metered_with_route(self, monkeypatch):
        _as_user(monkeypatch)
        await request_guard.check_demo_init_limit(_FakeRequest())
        rows = [r for r in _drain() if r["kind"] == "request"]
        assert len(rows) == 1
        assert rows[0]["route"] == "demo_init"
        assert rows[0]["user_id"] == "u-1"
        assert rows[0]["org_id"] == "o-1"

    async def test_rejected_call_not_metered(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 1)
        await request_guard.check_demo_init_limit(_FakeRequest())
        _drain()
        with pytest.raises(HTTPException):
            await request_guard.check_demo_init_limit(_FakeRequest())
        assert not [r for r in _drain() if r["kind"] == "request"]


class TestEdgarLimits:
    def test_defaults_are_non_zero(self):
        assert request_guard.EDGAR_PER_IP_PER_HOUR > 0
        assert request_guard.EDGAR_PER_USER_PER_HOUR > 0
        assert request_guard.EDGAR_BATCH_MAX == 5

    async def test_per_ip_limit(self, monkeypatch):
        monkeypatch.setattr(request_guard, "EDGAR_PER_IP_PER_HOUR", 2)
        req = _FakeRequest()
        await request_guard.check_edgar_limit(req, "edgar_search")
        await request_guard.check_edgar_limit(req, "edgar_filings")
        with pytest.raises(HTTPException) as exc:
            await request_guard.check_edgar_limit(req, "edgar_ingest")
        assert exc.value.status_code == 429
        assert int(exc.value.headers["Retry-After"]) >= 1

    async def test_per_user_limit(self, monkeypatch):
        _as_user(monkeypatch)
        monkeypatch.setattr(request_guard, "EDGAR_PER_IP_PER_HOUR", 100)
        monkeypatch.setattr(request_guard, "EDGAR_PER_USER_PER_HOUR", 3)
        await request_guard.check_edgar_limit(_FakeRequest(ip="10.0.0.1"), "edgar_ingest_batch", amount=3)
        with pytest.raises(HTTPException):
            await request_guard.check_edgar_limit(_FakeRequest(ip="10.0.0.2"), "edgar_search")

    async def test_owner_bypasses_but_is_metered(self, monkeypatch):
        monkeypatch.setattr(request_guard, "OWNER_KEY", "owner-secret")
        monkeypatch.setattr(request_guard, "EDGAR_PER_IP_PER_HOUR", 1)
        req = _FakeRequest()
        req.cookies = {request_guard.OWNER_COOKIE_NAME: request_guard.owner_cookie_value()}
        for _ in range(5):
            await request_guard.check_edgar_limit(req, "edgar_search")
        assert len([r for r in _drain() if r["route"] == "edgar_search"]) == 5

    def test_search_route_429_and_metered(self, client, monkeypatch):
        monkeypatch.setattr(request_guard, "EDGAR_PER_IP_PER_HOUR", 2)
        with patch(
            "api.connectors.edgar_connector.EdgarConnector.search_company",
            new_callable=AsyncMock, return_value=[],
        ):
            codes = [client.post("/api/edgar/search", json={"query": "AAPL"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert [r["route"] for r in _drain() if r["kind"] == "request"] == ["edgar_search"] * 2

    @pytest.mark.parametrize("count", [0, 6, 50])
    def test_batch_over_cap_is_400(self, client, count):
        with patch("api.edgar_service.ingest_multiple_filings", new_callable=AsyncMock) as ingest:
            resp = client.post("/api/edgar/ingest-batch", json={
                "cik": "320193", "ticker": "AAPL", "company_name": "Apple", "count": count,
            })
        assert resp.status_code == 400
        ingest.assert_not_called()

    def test_batch_charges_per_filing(self, client, monkeypatch):
        monkeypatch.setattr(request_guard, "EDGAR_PER_IP_PER_HOUR", 6)
        body = {"cik": "320193", "ticker": "AAPL", "company_name": "Apple", "count": 5}
        with patch("api.edgar_service.ingest_multiple_filings", new_callable=AsyncMock, return_value=[]):
            first = client.post("/api/edgar/ingest-batch", json=body)
            second = client.post("/api/edgar/ingest-batch", json=body)
        assert (first.status_code, second.status_code) == (200, 429)


class TestUsageRow:
    def test_meter_route_row_shape(self):
        usage_service.meter_route("edgar_ingest", user_id="u-9", org_id=None)
        (row,) = _drain()
        assert row["kind"] == "request"
        assert row["route"] == "edgar_ingest"
        assert row["user_id"] == "u-9"

    def test_migration_adds_route_column(self):
        assert any("route" in ddl and "IF NOT EXISTS" in ddl for ddl in usage_service.USAGE_EVENTS_MIGRATIONS)


def test_vector_stores_is_lru_capped():
    from api.utils.vector_store import SessionStoreCache

    assert isinstance(idx.VECTOR_STORES, SessionStoreCache)
    cache = SessionStoreCache(maxsize=2)
    cache["a"], cache["b"], cache["c"] = 1, 2, 3
    assert list(cache) == ["b", "c"]  # oldest evicted
