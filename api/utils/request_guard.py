"""Request-level abuse guards for public deployment.

Three independent, env-gated controls designed to make a *public* DocBot link
(e.g. shared on LinkedIn) safe from runaway LLM/sandbox cost and from the
live-DB connector attack surface:

1. Per-IP sliding-window rate limit  — caps request bursts from one client so a
   single visitor cannot spray the expensive LLM/sandbox endpoints.
2. Per-session weighted budget       — caps cumulative expensive ops per chat
   session. Weight is a proxy for token/compute cost (autopilot >> chat).
3. PUBLIC_DEMO_MODE path blocking     — disables uploads, live-DB connect and
   connector/EDGAR ingestion so a public demo can only exercise the hardcoded
   demo dataset. Zero external cost, no DB connector exposure.

State is in-memory, which is correct for a single-container Railway deploy.
For a multi-instance deploy, swap the module-level dicts for Redis.

All limits are OFF by default (open/dev mode). Enable per-env:

    PUBLIC_DEMO_MODE=true          # block uploads / live-DB / ingestion
    RATE_LIMIT_PER_MIN=30          # 0 disables the per-IP limit
    SESSION_BUDGET=40              # 0 disables the per-session budget
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, Tuple

from fastapi import HTTPException, Request


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
RATE_LIMIT_PER_MIN: int = _env_int("RATE_LIMIT_PER_MIN", 0)
RATE_LIMIT_WINDOW_SEC: int = 60
SESSION_BUDGET: int = _env_int("SESSION_BUDGET", 0)

# Sign-in gate: when on, upload / live-DB routes require a logged-in user (or
# owner). Anonymous visitors keep demo + chat access but must sign in (free
# GitHub/Google OAuth) to test their own data. This is the recommended public
# model — it lets people actually try uploads while capturing identity and
# capping cost per real user.
GATE_UPLOADS: bool = _env_bool("GATE_UPLOADS")
# Per-authenticated-user daily weighted budget. 0 disables. Resets at UTC midnight.
USER_DAILY_BUDGET: int = _env_int("USER_DAILY_BUDGET", 0)

# Owner bypass: when OWNER_KEY is set, a request presenting it (header, query
# param, or cookie) skips ALL guards — demo blocking, rate limit, and session
# budget. Lets the owner keep full access to a locked-down public deployment.
OWNER_KEY: str = os.getenv("OWNER_KEY", "").strip()
OWNER_COOKIE_NAME = "docbot_owner"

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
# In-memory state
# ---------------------------------------------------------------------------

_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)
_session_spend: Dict[str, int] = defaultdict(int)
# user_id -> (utc_date_str, spent_today)
_user_spend: Dict[str, Tuple[str, int]] = {}


def client_ip(request: Request) -> str:
    """Best-effort client IP, honouring the proxy hop Railway/Vercel add.

    X-Forwarded-For is a comma-separated list; the left-most entry is the
    original client. Falls back to the socket peer.
    """
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def is_owner(request: Request) -> bool:
    """True if the request presents the OWNER_KEY via header, query, or cookie.

    Owner requests bypass every guard. Returns False when OWNER_KEY is unset, so
    a deployment with no owner key configured never grants a bypass.
    """
    if not OWNER_KEY:
        return False
    if request.headers.get("x-owner-key") == OWNER_KEY:
        return True
    if request.query_params.get("owner_key") == OWNER_KEY:
        return True
    if request.cookies.get(OWNER_COOKIE_NAME) == OWNER_KEY:
        return True
    return False


def is_demo_blocked(method: str, path: str) -> bool:
    """True if PUBLIC_DEMO_MODE is on and this write path must be refused."""
    if not PUBLIC_DEMO_MODE:
        return False
    if method.upper() != "POST":
        return False
    return any(path.startswith(prefix) for prefix in _DEMO_BLOCKED_PREFIXES)


def check_ip_rate_limit(request: Request) -> None:
    """Sliding-window per-IP limit. Raises HTTP 429 when exceeded. No-op if disabled."""
    if RATE_LIMIT_PER_MIN <= 0:
        return

    ip = client_ip(request)
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    hits = _ip_hits[ip]

    # Drop timestamps outside the window.
    while hits and hits[0] < window_start:
        hits.popleft()

    if len(hits) >= RATE_LIMIT_PER_MIN:
        retry_after = int(hits[0] + RATE_LIMIT_WINDOW_SEC - now) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MIN}/min). Retry in {retry_after}s.",
            headers={"Retry-After": str(max(retry_after, 1))},
        )

    hits.append(now)


def enforce_session_budget(session_id: str, endpoint: str) -> None:
    """Charge the session's weighted budget for an expensive op.

    Raises HTTP 429 once the session exceeds SESSION_BUDGET. No-op if disabled
    or session_id is empty. Weight comes from ``_ENDPOINT_WEIGHTS`` (default 1).
    """
    if SESSION_BUDGET <= 0 or not session_id:
        return

    weight = _ENDPOINT_WEIGHTS.get(endpoint, 1)
    spent = _session_spend[session_id]

    if spent + weight > SESSION_BUDGET:
        raise HTTPException(
            status_code=429,
            detail=(
                "Session budget exhausted. This is a public demo with a per-session "
                "usage cap. Start a new session to continue."
            ),
        )

    _session_spend[session_id] = spent + weight


def enforce_user_budget(user_id: str, endpoint: str) -> None:
    """Charge an authenticated user's daily weighted budget for an expensive op.

    Raises HTTP 429 once the user exceeds USER_DAILY_BUDGET for the current UTC
    day. Counter resets automatically at midnight UTC. No-op if disabled or
    user_id is empty.
    """
    if USER_DAILY_BUDGET <= 0 or not user_id:
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    weight = _ENDPOINT_WEIGHTS.get(endpoint, 1)

    day, spent = _user_spend.get(user_id, (today, 0))
    if day != today:  # new UTC day → reset
        spent = 0

    if spent + weight > USER_DAILY_BUDGET:
        raise HTTPException(
            status_code=429,
            detail=(
                "Daily free-tier limit reached. Your quota resets at midnight UTC. "
                "Thanks for testing DocBot!"
            ),
        )

    _user_spend[user_id] = (today, spent + weight)


def reset_state() -> None:
    """Clear all in-memory counters. Test-only helper."""
    _ip_hits.clear()
    _session_spend.clear()
    _user_spend.clear()
