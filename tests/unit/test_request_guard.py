"""Unit tests for api.utils.request_guard — public-deployment abuse guards."""

import pytest
from fastapi import HTTPException

from api.utils import request_guard


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    """Minimal stand-in for starlette Request (headers + client)."""

    def __init__(self, ip="1.2.3.4", forwarded=None, method="POST", path="/api/chat",
                 headers=None, query_params=None, cookies=None):
        hdrs = dict(headers or {})
        if forwarded is not None:
            hdrs["x-forwarded-for"] = forwarded
        self.headers = hdrs
        self.query_params = dict(query_params or {})
        self.cookies = dict(cookies or {})
        self.client = _FakeClient(ip)
        self.method = method

        class _URL:
            def __init__(self, p):
                self.path = p

        self.url = _URL(path)


@pytest.fixture(autouse=True)
def _clean_state():
    request_guard.reset_state()
    yield
    request_guard.reset_state()


# --- client_ip -------------------------------------------------------------

def test_client_ip_prefers_forwarded_header():
    req = _FakeRequest(ip="10.0.0.1", forwarded="203.0.113.7, 10.0.0.1")
    assert request_guard.client_ip(req) == "203.0.113.7"


def test_client_ip_falls_back_to_socket_peer():
    req = _FakeRequest(ip="198.51.100.2")
    assert request_guard.client_ip(req) == "198.51.100.2"


# --- per-IP rate limit -----------------------------------------------------

def test_rate_limit_disabled_by_default(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    for _ in range(1000):
        request_guard.check_ip_rate_limit(_FakeRequest())  # never raises


def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 3)
    req = _FakeRequest(ip="5.5.5.5")
    for _ in range(3):
        request_guard.check_ip_rate_limit(req)
    with pytest.raises(HTTPException) as exc:
        request_guard.check_ip_rate_limit(req)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_rate_limit_is_per_ip(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 2)
    a, b = _FakeRequest(ip="1.1.1.1"), _FakeRequest(ip="2.2.2.2")
    request_guard.check_ip_rate_limit(a)
    request_guard.check_ip_rate_limit(a)
    # b has its own bucket — unaffected by a hitting the limit
    request_guard.check_ip_rate_limit(b)
    request_guard.check_ip_rate_limit(b)
    with pytest.raises(HTTPException):
        request_guard.check_ip_rate_limit(a)


def test_rate_limit_window_slides(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_WINDOW_SEC", 60)
    fake_now = [1000.0]
    monkeypatch.setattr(request_guard.time, "monotonic", lambda: fake_now[0])
    req = _FakeRequest(ip="9.9.9.9")
    request_guard.check_ip_rate_limit(req)
    request_guard.check_ip_rate_limit(req)
    with pytest.raises(HTTPException):
        request_guard.check_ip_rate_limit(req)
    # advance past the window — old hits expire
    fake_now[0] += 61
    request_guard.check_ip_rate_limit(req)  # no raise


# --- per-session budget ----------------------------------------------------

def test_budget_disabled_by_default(monkeypatch):
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 0)
    for _ in range(1000):
        request_guard.enforce_session_budget("s1", "autopilot")  # never raises


def test_budget_empty_session_is_noop(monkeypatch):
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 1)
    for _ in range(10):
        request_guard.enforce_session_budget("", "autopilot")  # anonymous → skip


def test_budget_exhausts_by_weight(monkeypatch):
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 5)
    # autopilot weight is 5 → one call consumes the whole budget
    request_guard.enforce_session_budget("s1", "autopilot")
    with pytest.raises(HTTPException) as exc:
        request_guard.enforce_session_budget("s1", "chat")
    assert exc.value.status_code == 429


def test_budget_counts_cheap_ops(monkeypatch):
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 3)
    for _ in range(3):
        request_guard.enforce_session_budget("s2", "chat")  # weight 1 each
    with pytest.raises(HTTPException):
        request_guard.enforce_session_budget("s2", "chat")


def test_budget_is_per_session(monkeypatch):
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 1)
    request_guard.enforce_session_budget("sA", "chat")
    with pytest.raises(HTTPException):
        request_guard.enforce_session_budget("sA", "chat")
    request_guard.enforce_session_budget("sB", "chat")  # separate bucket


# --- demo path blocking ----------------------------------------------------

def test_demo_block_off_by_default(monkeypatch):
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", False)
    assert request_guard.is_demo_blocked("POST", "/api/upload") is False


def test_demo_blocks_upload_paths(monkeypatch):
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", True)
    assert request_guard.is_demo_blocked("POST", "/api/upload") is True
    assert request_guard.is_demo_blocked("POST", "/api/db/connect") is True
    assert request_guard.is_demo_blocked("POST", "/api/db/upload/csv") is True
    assert request_guard.is_demo_blocked("POST", "/api/connectors/register") is True
    assert request_guard.is_demo_blocked("POST", "/api/edgar/ingest-batch") is True


def test_demo_allows_chat_and_demo_init(monkeypatch):
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", True)
    assert request_guard.is_demo_blocked("POST", "/api/chat") is False
    assert request_guard.is_demo_blocked("POST", "/api/demo/init") is False
    assert request_guard.is_demo_blocked("POST", "/api/hybrid/chat") is False


def test_demo_ignores_non_post(monkeypatch):
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", True)
    assert request_guard.is_demo_blocked("GET", "/api/upload") is False


# --- owner bypass ----------------------------------------------------------

def test_is_owner_false_when_key_unset(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    req = _FakeRequest(headers={"x-owner-key": "anything"})
    assert request_guard.is_owner(req) is False


def test_is_owner_via_header(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "s3cret")
    assert request_guard.is_owner(_FakeRequest(headers={"x-owner-key": "s3cret"})) is True
    assert request_guard.is_owner(_FakeRequest(headers={"x-owner-key": "wrong"})) is False


def test_is_owner_via_query_param(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "s3cret")
    assert request_guard.is_owner(_FakeRequest(query_params={"owner_key": "s3cret"})) is True


def test_is_owner_via_cookie(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "s3cret")
    cookies = {request_guard.OWNER_COOKIE_NAME: "s3cret"}
    assert request_guard.is_owner(_FakeRequest(cookies=cookies)) is True


def test_is_owner_no_credentials(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "s3cret")
    assert request_guard.is_owner(_FakeRequest()) is False


# --- per-user daily budget -------------------------------------------------

def test_user_budget_disabled_by_default(monkeypatch):
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 0)
    for _ in range(1000):
        request_guard.enforce_user_budget("u1", "autopilot")  # never raises


def test_user_budget_empty_id_is_noop(monkeypatch):
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 1)
    for _ in range(10):
        request_guard.enforce_user_budget("", "autopilot")


def test_user_budget_exhausts_and_429(monkeypatch):
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 80)
    # 16 autopilot runs (weight 5) = 80 exactly; 17th exceeds
    for _ in range(16):
        request_guard.enforce_user_budget("u1", "autopilot")
    with pytest.raises(HTTPException) as exc:
        request_guard.enforce_user_budget("u1", "autopilot")
    assert exc.value.status_code == 429


def test_user_budget_is_per_user(monkeypatch):
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 1)
    request_guard.enforce_user_budget("uA", "chat")
    with pytest.raises(HTTPException):
        request_guard.enforce_user_budget("uA", "chat")
    request_guard.enforce_user_budget("uB", "chat")  # separate quota


def test_user_budget_resets_on_new_day(monkeypatch):
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 1)
    # Day 1: consume quota
    class _D1:
        @staticmethod
        def now(tz=None):
            import datetime as _dt
            return _dt.datetime(2026, 7, 1, 12, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(request_guard, "datetime", _D1)
    request_guard.enforce_user_budget("u1", "chat")
    with pytest.raises(HTTPException):
        request_guard.enforce_user_budget("u1", "chat")
    # Day 2: quota resets
    class _D2:
        @staticmethod
        def now(tz=None):
            import datetime as _dt
            return _dt.datetime(2026, 7, 2, 0, 5, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(request_guard, "datetime", _D2)
    request_guard.enforce_user_budget("u1", "chat")  # no raise
