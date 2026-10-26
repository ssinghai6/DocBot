"""Request-level abuse guards for public deployment.

Three independent, env-gated controls designed to make a *public* DocBot link
(e.g. shared on LinkedIn) safe from runaway LLM/sandbox cost and from the
live-DB connector attack surface:

1. Per-IP fixed-window rate limit    — caps request bursts from one client so a
   single visitor cannot spray the expensive LLM/sandbox endpoints.
2. Per-session weighted budget       — caps cumulative expensive ops per chat
   session. Weight is a proxy for token/compute cost (autopilot >> chat).
3. PUBLIC_DEMO_MODE path blocking     — disables uploads, live-DB connect and
   connector/EDGAR ingestion so a public demo can only exercise the hardcoded
   demo dataset. Zero external cost, no DB connector exposure.

State (DOCBOT-1606): fixed-window counters in the ``rate_limit_counters``
table, one row per (bucket, key, window_start), bumped with an atomic
INSERT ... ON CONFLICT DO UPDATE ... RETURNING count. Counters therefore
survive redeploys and are shared by every container. ``index.py`` wires the
SQL store at startup; until then (tests, scripts) an in-memory store is used.
On a DB error or timeout the guards FAIL OPEN with a warning — an outage of
the counter table must never take chat down. Expired windows are deleted
opportunistically every ``_CLEANUP_EVERY`` charges.

Limits are ON by default with safe values (0 disables any of them):

    PUBLIC_DEMO_MODE=true          # block uploads / live-DB / ingestion (off by default)
    RATE_LIMIT_PER_MIN=60          # per client IP, POST/DELETE /api/*
    SESSION_BUDGET=100             # weighted ops per anonymous session per 7 days
    USER_DAILY_BUDGET=500          # weighted ops per signed-in user per UTC day

Exception — ON by default (DOCBOT-1612): POST /api/demo/init is IP-limited
because every call builds a vector store + SQLite DB in memory:

    DEMO_INIT_PER_IP_PER_HOUR=10   # per client IP, sliding 1h window; 0 disables
    DEMO_INIT_GLOBAL_PER_HOUR=300  # all clients combined; 0 disables. Bounds
                                   # total cost even if X-Forwarded-For is spoofed.

Exception — ON by default (DOCBOT-1607): POST /api/sandbox/execute is
metered per org (falling back to per user) — never per session_id, which a
client can mint freely:

    SANDBOX_RUNS_PER_HOUR=60       # per org (or user), sliding 1h window; 0 disables

Client IP (DOCBOT-1606):

    TRUSTED_PROXY_HOPS=1           # number of trusted proxies that APPEND to
                                   # X-Forwarded-For. Client IP = parts[-N]
                                   # (never the client-controlled left-most).
                                   # 0 = ignore the header, use the socket peer.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import math
import os
import time
from typing import Dict, Optional, Protocol, Tuple

from fastapi import HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (read once at import; process restart picks up new env)
# ---------------------------------------------------------------------------

def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


PUBLIC_DEMO_MODE: bool = _env_bool("PUBLIC_DEMO_MODE")
RATE_LIMIT_PER_MIN: int = _env_int("RATE_LIMIT_PER_MIN", 60)
RATE_LIMIT_WINDOW_SEC: int = 60
SESSION_BUDGET: int = _env_int("SESSION_BUDGET", 100)
SESSION_BUDGET_WINDOW_SEC: int = 7 * 86400

# DOCBOT-1612: /api/demo/init limits — non-zero by default (safe default).
DEMO_INIT_WINDOW_SEC: int = 3600
DEMO_INIT_PER_IP_PER_HOUR: int = _env_int("DEMO_INIT_PER_IP_PER_HOUR", 10)
DEMO_INIT_GLOBAL_PER_HOUR: int = _env_int("DEMO_INIT_GLOBAL_PER_HOUR", 300)

# DOCBOT-1607: raw sandbox executions per org (or user) per hour.
SANDBOX_WINDOW_SEC: int = 3600
SANDBOX_RUNS_PER_HOUR: int = _env_int("SANDBOX_RUNS_PER_HOUR", 60)

# Number of trusted reverse proxies that APPEND to X-Forwarded-For in front of
# the app (see client_ip()). 0 ignores the header entirely.
TRUSTED_PROXY_HOPS: int = _env_int("TRUSTED_PROXY_HOPS", 1)

# Sign-in gate: when on, upload / live-DB routes require a logged-in user (or
# owner). Anonymous visitors keep demo + chat access but must sign in (free
# GitHub/Google OAuth) to test their own data. This is the recommended public
# model — it lets people actually try uploads while capturing identity and
# capping cost per real user.
GATE_UPLOADS: bool = _env_bool("GATE_UPLOADS")
USER_DAILY_BUDGET: int = _env_int("USER_DAILY_BUDGET", 500)
# Epoch-aligned 86400s windows start at UTC midnight.
USER_BUDGET_WINDOW_SEC: int = 86400

# Owner bypass: when OWNER_KEY is set, a browser holding the docbot_owner cookie
# skips ALL guards — demo blocking, rate limit, and session budget. The raw key
# is only ever accepted in the X-Owner-Key header of POST /api/auth/owner-unlock
# (DOCBOT-1605); the cookie holds an HMAC of the key, never the key itself.
OWNER_KEY: str = os.getenv("OWNER_KEY", "").strip()
OWNER_COOKIE_NAME = "docbot_owner"
OWNER_HEADER_NAME = "x-owner-key"

# POST paths disabled when PUBLIC_DEMO_MODE is on. Matched by prefix so that
# parametrised routes (e.g. /api/connectors/{id}/sync) are covered.
_DEMO_BLOCKED_PREFIXES = (
    "/api/upload",              # PDF upload → embeddings cost
    "/api/db/connect",          # live-DB connect → SSRF / data-exfil surface
    "/api/db/upload",           # CSV / SQLite upload → sandbox + embeddings cost
    "/api/connectors/register", # marketplace connector registration
    "/api/edgar/ingest",        # EDGAR download + chunk + embed cost
)

# Weight per expensive endpoint — proxy for LLM/sandbox cost per call.
# Autopilot and deep retrieval fan out into many LLM calls, so they cost more
# of the per-session budget than a single chat turn.
_ENDPOINT_WEIGHTS: Dict[str, int] = {
    "chat": 1,
    "hybrid": 2,
    "sandbox": 2,
    "autopilot": 5,
}


# ---------------------------------------------------------------------------
# Counter store (DOCBOT-1606)
# ---------------------------------------------------------------------------

COUNTER_TABLE = "rate_limit_counters"
_STORE_TIMEOUT_SEC = 2.0
_CLEANUP_EVERY = 500

_UPSERT_SQL = text(
    "INSERT INTO rate_limit_counters (bucket, counter_key, window_start, count, expires_at) "
    "VALUES (:bucket, :key, :window_start, :amount, :expires_at) "
    "ON CONFLICT (bucket, counter_key, window_start) "
    "DO UPDATE SET count = rate_limit_counters.count + excluded.count "
    "RETURNING count"
)
_CLEANUP_SQL = text("DELETE FROM rate_limit_counters WHERE expires_at < :now")


class CounterStore(Protocol):
    async def incr(self, bucket: str, key: str, window_start: int, expires_at: int, amount: int) -> int: ...
    async def cleanup(self, now: int) -> None: ...


class MemoryCounterStore:
    """Per-process fallback used before the SQL store is wired (tests, scripts)."""

    def __init__(self) -> None:
        self._rows: Dict[Tuple[str, str, int], Tuple[int, int]] = {}

    async def incr(self, bucket: str, key: str, window_start: int, expires_at: int, amount: int) -> int:
        count = self._rows.get((bucket, key, window_start), (0, expires_at))[0] + amount
        self._rows[(bucket, key, window_start)] = (count, expires_at)
        return count

    async def cleanup(self, now: int) -> None:
        self._rows = {k: v for k, v in self._rows.items() if v[1] >= now}


class SqlCounterStore:
    """Durable counters in Postgres (SQLite in tests) — one atomic upsert per charge."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def incr(self, bucket: str, key: str, window_start: int, expires_at: int, amount: int) -> int:
        async with self._engine.begin() as conn:
            result = await conn.execute(_UPSERT_SQL, {
                "bucket": bucket, "key": key, "window_start": window_start,
                "amount": amount, "expires_at": expires_at,
            })
            return int(result.scalar_one())

    async def cleanup(self, now: int) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(_CLEANUP_SQL, {"now": now})


_store: CounterStore = MemoryCounterStore()
_charges_since_cleanup = 0
_STORE_ERRORS = (SQLAlchemyError, OSError, asyncio.TimeoutError)


def use_counter_store(store: CounterStore) -> None:
    """Swap the counter backend (index.py wires SqlCounterStore at startup)."""
    global _store
    _store = store


async def _maybe_cleanup(now: int) -> None:
    global _charges_since_cleanup
    _charges_since_cleanup += 1
    if _charges_since_cleanup < _CLEANUP_EVERY:
        return
    _charges_since_cleanup = 0
    try:
        await asyncio.wait_for(_store.cleanup(now), _STORE_TIMEOUT_SEC)
    except _STORE_ERRORS as exc:
        logger.warning("rate-limit counter cleanup failed: %s", type(exc).__name__)


async def _charge(bucket: str, key: str, window_sec: int, limit: int, amount: int = 1) -> Optional[int]:
    """Charge ``amount`` against a fixed window. Returns None if allowed, else Retry-After seconds.

    A rejected charge is refunded so it does not consume quota. Fails OPEN
    (returns None) when the store errors or times out.
    """
    now = time.time()
    window_start = int(now // window_sec) * window_sec
    expires_at = window_start + window_sec
    try:
        count = await asyncio.wait_for(
            _store.incr(bucket, key, window_start, expires_at, amount), _STORE_TIMEOUT_SEC
        )
    except _STORE_ERRORS as exc:
        logger.warning("rate-limit store unavailable (%s) — failing open for %s", type(exc).__name__, bucket)
        return None
    if count > limit:
        try:
            await asyncio.wait_for(
                _store.incr(bucket, key, window_start, expires_at, -amount), _STORE_TIMEOUT_SEC
            )
        except _STORE_ERRORS as exc:
            logger.warning("rate-limit refund failed: %s", type(exc).__name__)
        return max(math.ceil(expires_at - now), 1)
    await _maybe_cleanup(int(now))
    return None


def client_ip(request: Request) -> str:
    """Client IP from the right-most trusted X-Forwarded-For hop (DOCBOT-1606).

    Each proxy APPENDS the address it received the request from, so only the
    right-most TRUSTED_PROXY_HOPS entries are written by infrastructure we
    trust; anything to their left is client-supplied and never used. With N
    hops the client is ``parts[-N]`` (clamped to the left-most entry when the
    header is shorter — then every entry was proxy-written). With
    TRUSTED_PROXY_HOPS=0 the header is ignored and the socket peer is used.
    """
    fwd = request.headers.get("x-forwarded-for") if TRUSTED_PROXY_HOPS > 0 else None
    if fwd:
        parts = [p.strip() for p in fwd.split(",") if p.strip()]
        if parts:
            return parts[max(len(parts) - TRUSTED_PROXY_HOPS, 0)]
    return request.client.host if request.client else "unknown"


def owner_cookie_value() -> str:
    """HMAC-SHA256 of OWNER_KEY — what the docbot_owner cookie stores. "" when unset."""
    if not OWNER_KEY:
        return ""
    return hmac.new(OWNER_KEY.encode(), b"docbot-owner-cookie-v1", hashlib.sha256).hexdigest()


def verify_owner_key(presented: str | None) -> bool:
    """Constant-time check of a raw key (from the unlock header) against OWNER_KEY."""
    if not OWNER_KEY or not presented:
        return False
    return hmac.compare_digest(presented.encode(), OWNER_KEY.encode())


def is_owner(request: Request) -> bool:
    """True if the request carries a valid docbot_owner cookie.

    Owner requests bypass every guard. Returns False when OWNER_KEY is unset, so
    a deployment with no owner key configured never grants a bypass. The raw
    key is NOT accepted here (no header, no query param) — see verify_owner_key.
    """
    expected = owner_cookie_value()
    cookie = request.cookies.get(OWNER_COOKIE_NAME)
    if not expected or not cookie:
        return False
    return hmac.compare_digest(cookie.encode(), expected.encode())


def is_demo_blocked(method: str, path: str) -> bool:
    """True if PUBLIC_DEMO_MODE is on and this write path must be refused."""
    if not PUBLIC_DEMO_MODE:
        return False
    if method.upper() != "POST":
        return False
    return any(path.startswith(prefix) for prefix in _DEMO_BLOCKED_PREFIXES)


async def check_ip_rate_limit(request: Request) -> None:
    """Fixed-window per-IP limit. Raises HTTP 429 when exceeded. No-op if disabled."""
    if RATE_LIMIT_PER_MIN <= 0:
        return
    retry_after = await _charge("ip", client_ip(request), RATE_LIMIT_WINDOW_SEC, RATE_LIMIT_PER_MIN)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MIN}/min). Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


def _too_many(limit: int, retry_after: int, scope: str) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            f"Demo limit reached ({limit}/hour {scope}). "
            f"Retry in {retry_after}s, or sign in to upload your own data."
        ),
        headers={"Retry-After": str(retry_after)},
    )


async def check_demo_init_limit(request: Request) -> None:
    """IP-based limit for POST /api/demo/init — DOCBOT-1612.

    Two 1-hour windows: per client IP (``DEMO_INIT_PER_IP_PER_HOUR``) and
    global (``DEMO_INIT_GLOBAL_PER_HOUR``). The global cap bounds total cost
    even when a client rotates spoofed X-Forwarded-For values. Owner requests
    bypass. Either limit set to 0 disables that window. Raises 429 with
    Retry-After. A rejected call does not consume quota.
    """
    if is_owner(request):
        return

    ip = client_ip(request)
    if DEMO_INIT_PER_IP_PER_HOUR > 0:
        retry = await _charge("demo_init_ip", ip, DEMO_INIT_WINDOW_SEC, DEMO_INIT_PER_IP_PER_HOUR)
        if retry is not None:
            raise _too_many(DEMO_INIT_PER_IP_PER_HOUR, retry, "per client")

    if DEMO_INIT_GLOBAL_PER_HOUR > 0:
        retry = await _charge("demo_init_global", "*", DEMO_INIT_WINDOW_SEC, DEMO_INIT_GLOBAL_PER_HOUR)
        if retry is not None:
            if DEMO_INIT_PER_IP_PER_HOUR > 0:  # give the per-IP slot back
                await _charge("demo_init_ip", ip, DEMO_INIT_WINDOW_SEC, DEMO_INIT_PER_IP_PER_HOUR, -1)
            raise _too_many(DEMO_INIT_GLOBAL_PER_HOUR, retry, "across all visitors")


def sandbox_principal_key(user: object, request: Request) -> str:
    """Metering key for raw sandbox runs — org first, then user, then IP.

    Never derived from session_id (client-mintable). The IP fallback only
    applies to the dev-only anonymous sandbox (ALLOW_ANONYMOUS_SANDBOX).
    """
    org_id = getattr(user, "org_id", None) if user is not None else None
    if org_id:
        return f"org:{org_id}"
    user_id = getattr(user, "id", None) if user is not None else None
    if user_id:
        return f"user:{user_id}"
    return f"ip:{client_ip(request)}"


async def enforce_sandbox_quota(request: Request, user: object) -> None:
    """Rate-limit + budget POST /api/sandbox/execute by org/user — DOCBOT-1607.

    1. Fixed 1h window of ``SANDBOX_RUNS_PER_HOUR`` per org (or user).
    2. The authenticated user's daily weighted budget (``USER_DAILY_BUDGET``).

    Owner requests bypass. A rejected call does not consume quota. Raises 429.
    """
    if is_owner(request):
        return

    if SANDBOX_RUNS_PER_HOUR > 0:
        retry = await _charge(
            "sandbox", sandbox_principal_key(user, request), SANDBOX_WINDOW_SEC, SANDBOX_RUNS_PER_HOUR
        )
        if retry is not None:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Sandbox limit reached ({SANDBOX_RUNS_PER_HOUR}/hour for your "
                    f"organization). Retry in {retry}s."
                ),
                headers={"Retry-After": str(retry)},
            )

    user_id = getattr(user, "id", None) if user is not None else None
    if user_id:
        await enforce_user_budget(str(user_id), "sandbox")


async def enforce_session_budget(session_id: str, endpoint: str) -> None:
    """Charge the session's weighted budget for an expensive op.

    Raises HTTP 429 once the session exceeds SESSION_BUDGET (per 7-day window).
    No-op if disabled or session_id is empty. Weight comes from
    ``_ENDPOINT_WEIGHTS`` (default 1).
    """
    if SESSION_BUDGET <= 0 or not session_id:
        return
    weight = _ENDPOINT_WEIGHTS.get(endpoint, 1)
    if await _charge("session", session_id, SESSION_BUDGET_WINDOW_SEC, SESSION_BUDGET, weight) is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                "Session budget exhausted. This is a public demo with a per-session "
                "usage cap. Start a new session to continue."
            ),
        )


async def enforce_user_budget(user_id: str, endpoint: str) -> None:
    """Charge an authenticated user's daily weighted budget for an expensive op.

    Raises HTTP 429 once the user exceeds USER_DAILY_BUDGET for the current UTC
    day (window resets at midnight UTC). No-op if disabled or user_id is empty.
    """
    if USER_DAILY_BUDGET <= 0 or not user_id:
        return
    weight = _ENDPOINT_WEIGHTS.get(endpoint, 1)
    if await _charge("user_day", user_id, USER_BUDGET_WINDOW_SEC, USER_DAILY_BUDGET, weight) is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                "Daily free-tier limit reached. Your quota resets at midnight UTC. "
                "Thanks for testing DocBot!"
            ),
        )


def reset_state() -> None:
    """Reset to a fresh in-memory counter store. Test-only helper."""
    global _charges_since_cleanup
    use_counter_store(MemoryCounterStore())
    _charges_since_cleanup = 0
