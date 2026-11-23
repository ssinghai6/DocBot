"""DOCBOT-3901: per-key /v1 rate limit — 429 + Retry-After + X-RateLimit-*, no sleeping.

The real ``require_api_key`` runs; key validation, the audit write and the
connector store are mocked, and the counter store is a fresh in-memory one.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api import api_key_service  # noqa: E402
from api.api_key_service import APIKeyRecord  # noqa: E402
from api.utils import request_guard  # noqa: E402


def _record(prefix: str) -> APIKeyRecord:
    return APIKeyRecord(id=f"key-{prefix}", org_id="org-1", key_prefix=prefix, name="k",
                        scopes=["connectors:manage"], mode="test" if prefix.startswith("dk_test_") else "live")


@pytest.fixture
def call(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(request_guard, "_store", request_guard.MemoryCounterStore())
    monkeypatch.setattr(request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(request_guard, "V1_RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(request_guard, "V1_TEST_RATE_LIMIT_PER_MIN", 1)

    client = TestClient(idx.app, raise_server_exceptions=False)

    def _call(prefix: str = "dk_live_abc"):
        with patch.object(api_key_service, "validate_api_key", new=AsyncMock(return_value=_record(prefix))), \
             patch("api.audit_service.log_event", MagicMock()), \
             patch("api.connector_store.list_active_connectors", AsyncMock(return_value=[])):
            return client.get("/v1/connectors", headers={"Authorization": f"Bearer {prefix}"})

    return _call


def test_headers_on_success_then_429_over_limit(call) -> None:
    first, second, third = call(), call(), call()
    assert first.status_code == 200 and second.status_code == 200, first.text
    assert first.headers["X-RateLimit-Limit"] == "2"
    assert first.headers["X-RateLimit-Remaining"] == "1"
    assert second.headers["X-RateLimit-Remaining"] == "0"
    assert int(first.headers["X-RateLimit-Reset"]) > 0
    assert first.headers["X-Request-Id"].startswith("req_")

    # Rejected immediately (the old in-memory bucket slept instead of rejecting).
    assert third.status_code == 429, third.text
    assert third.elapsed.total_seconds() < 1
    err = third.json()["error"]
    assert (err["type"], err["code"]) == ("rate_limit", "rate_limited")
    assert int(third.headers["Retry-After"]) >= 1
    assert third.headers["X-RateLimit-Limit"] == "2"
    assert third.headers["X-RateLimit-Remaining"] == "0"
    assert "X-RateLimit-Reset" in third.headers


def test_rejected_call_does_not_consume_quota(call) -> None:
    for _ in range(4):
        call()
    # Store holds exactly the limit, not 4: rejected charges are refunded.
    rows = request_guard._store._rows
    assert [count for (bucket, *_), (count, _) in rows.items() if bucket == "v1_key"] == [2]


def test_test_keys_get_the_lower_limit(call) -> None:
    assert call("dk_test_abc").status_code == 200
    resp = call("dk_test_abc")
    assert resp.status_code == 429
    assert resp.headers["X-RateLimit-Limit"] == "1"


def test_limits_are_per_key(call) -> None:
    call("dk_live_aaa"), call("dk_live_aaa")
    assert call("dk_live_aaa").status_code == 429
    assert call("dk_live_bbb").status_code == 200


async def test_charge_window_fails_open_when_store_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        async def incr(self, *_a, **_k):
            raise OSError("down")

    monkeypatch.setattr(request_guard, "_store", Broken())
    retry, remaining, _reset = await request_guard.charge_window("v1_key", "k", 60, 5)
    assert (retry, remaining) == (None, 5)
