"""Route-guard manifest and sweep — DOCBOT-1625 (YC plan §4 security gate #1).

Every FastAPI route must fall into exactly one bucket:

* **dependency-guarded**: an auth dependency (``require_role``, ``require_admin``,
  ``require_principal``, ``require_api_key`` ...) appears somewhere in the
  route's dependency tree. Detection is by callable *name*, not identity, so
  a refactor that adds a new ``require_*`` guard factory keeps working.
* ``HANDLER_GUARDED``: the handler itself rejects anonymous callers with 401
  (verified behaviourally below, not just by listing).
* ``DEV_ONLY``: returns 404 unless ``ENV=dev`` (also verified below).
* ``PUBLIC_ROUTES``: intentionally anonymous, with a written justification.
* ``KNOWN_UNGUARDED``: a ratchet of existing gaps, each mapped to the ticket
  that closes it.

The ratchet only tightens:

* a NEW route that is not guarded and not listed fails ``test_every_route_is_classified``;
* a KNOWN_UNGUARDED route that becomes guarded (or is deleted) fails
  ``test_known_unguarded_ratchet`` until its entry is removed.

Under ``AUTH_REQUIRED=true``, anonymous requests to every guarded route must
get 401. Those requests never reach the database: guards reject before the
handler runs, and the TestClient is built without running the app lifespan
(so ``init_db()`` never connects).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import os
import re
from typing import Callable, Iterable, Iterator

import pytest
from cryptography.fernet import Fernet

# api.index reads env at import time; set the same fakes CI uses.
if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.dependencies.models import Dependant  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

try:
    # fastapi >= 0.162 wraps include_router() output in a private _IncludedRouter
    # entry instead of flattening every sub-route into app.routes. Older
    # fastapi versions flatten directly, so the class may not exist.
    from fastapi.routing import _IncludedRouter  # noqa: E402
except ImportError:  # pragma: no cover — only hit on fastapi < 0.162
    _IncludedRouter = ()

import api.index as idx  # noqa: E402

RouteKey = tuple[str, str]  # (METHOD, path)

# ---------------------------------------------------------------------------
# Guard detection
# ---------------------------------------------------------------------------

# A dependency callable counts as an auth guard when any segment of its
# __qualname__ matches one of these names. Factories return closures such as
# ``require_role.<locals>._dependency``; the outer name is what we match.
GUARD_CALLABLE_NAMES: frozenset[str] = frozenset({
    "require_role",        # api/rbac_service.py — session cookie + role (viewer/analyst/admin)
    "require_admin",       # DOCBOT-1610 (Lane S) — admin guard that ignores open mode
    "require_platform_admin",  # DOCBOT-1610 — admin + PLATFORM_ADMIN_USER_IDS (/admin/metrics)
    "require_sandbox_user",    # DOCBOT-1612 — analyst+ in every mode; anon only via dev-only ALLOW_ANONYMOUS_SANDBOX
    "require_principal",   # DOCBOT-1711 — user / demo principal, else 401
    "require_user",
    "require_auth",
    "get_current_user",
    "require_api_key",     # api/api_key_service.py — /v1 Bearer key + scope
    "get_caller_scope",    # DOCBOT-1601 — 401 under AUTH_REQUIRED; open mode scopes rows to the anon owner cookie
})

# Dependencies that look auth-ish but are NOT guards: they only gate when an
# env flag is set, so in default config they let anonymous callers through.
NON_GUARD_CALLABLE_NAMES: frozenset[str] = frozenset({
    "gate_uploads",        # api/index.py — enforces only when GATE_UPLOADS=true
    "get_optional_user",   # returns None for anonymous, never raises
})


def _is_guard_callable(call: Callable[..., object] | None) -> bool:
    if call is None:
        return False
    module = getattr(call, "__module__", "") or ""
    if not module.startswith("api."):
        return False
    qualname = getattr(call, "__qualname__", "") or getattr(call, "__name__", "")
    segments = set(qualname.split("."))
    if segments & NON_GUARD_CALLABLE_NAMES:
        return False
    return bool(segments & GUARD_CALLABLE_NAMES)


def _walk(dependant: Dependant) -> Iterator[Dependant]:
    for dep in dependant.dependencies:
        yield dep
        yield from _walk(dep)


def _has_guard(route: APIRoute) -> bool:
    return any(_is_guard_callable(d.call) for d in _walk(route.dependant))


def _api_routes() -> dict[RouteKey, APIRoute]:
    """Flatten every reachable HTTP route, including nested included routers.

    fastapi >= 0.162 keeps ``include_router()`` output behind an
    ``_IncludedRouter`` wrapper instead of appending each sub-route directly to
    ``app.routes``, so a plain iteration only sees top-level routes. Descend
    into the wrapped router and apply the include prefix; guard detection still
    uses each sub-route's own dependency (include-level ``Depends`` ratchets the
    manifest test's "guarded" signal, never loosens it).
    """
    routes: dict[RouteKey, APIRoute] = {}

    def _collect(items: Iterable[object], prefix: str) -> None:
        for route in items:
            if isinstance(route, _IncludedRouter):
                ctx = route.include_context
                _collect(route.original_router.routes, prefix + ctx.prefix)
                continue
            if not isinstance(route, APIRoute):
                continue  # /openapi.json, /docs, /redoc — FastAPI built-ins
            for method in route.methods:
                if method in ("HEAD", "OPTIONS"):
                    continue
                routes[(method, prefix + route.path)] = route

    _collect(idx.app.routes, "")
    return routes


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

PUBLIC_ROUTES: dict[RouteKey, str] = {
    ("GET", "/api/health"): "Liveness probe for Railway/healthcheck.sh; returns no user data.",
    ("GET", "/api/personas"): "Static persona catalogue rendered before login.",
    ("GET", "/api/tools"): "Static tool registry metadata for the tool picker (DOCBOT-1514); read-only.",
    ("GET", "/api/auth/config"): "Lists enabled login methods; the login screen needs it pre-auth.",
    ("POST", "/api/demo/init"): "One-click anonymous demo (EPIC-13), open even when APP_AUTH_MODE=required. "
                                "Abuse controls: IP rate limit (DOCBOT-1612), metering (DOCBOT-1623) "
                                "and the DEMO_DAILY_USD_CAP breaker (DOCBOT-1711), not auth.",
    ("POST", "/api/auth/owner-unlock"): "Checks the X-Owner-Key header itself (compare_digest) and returns 403 on mismatch (DOCBOT-1605).",
    ("GET", "/api/auth/saml/metadata"): "SAML SP metadata must be fetchable by the IdP.",
    ("GET", "/api/auth/saml/login"): "Starts the SAML login redirect.",
    ("POST", "/api/auth/saml/acs"): "SAML Assertion Consumer Service; the signed assertion is the credential.",
    ("GET", "/api/auth/github"): "Starts GitHub OAuth (returns authorize URL + state).",
    ("GET", "/api/auth/github/callback"): "GitHub OAuth callback; validated via state + code exchange.",
    ("GET", "/api/auth/google"): "Starts Google OAuth.",
    ("GET", "/api/auth/google/callback"): "Google OAuth callback; validated via state + code exchange.",
    ("POST", "/api/auth/register"): "Email/password sign-up (abuse controls: DOCBOT-1622, 1701).",
    ("POST", "/api/auth/login"): "Email/password login (throttling: DOCBOT-1622).",
    ("POST", "/api/auth/verify"): "DOCBOT-1713: consumes the single-use emailed verification "
                                  "token (SHA-256-hashed at rest; the token itself IS the "
                                  "credential; expiry + single-use enforced atomically).",
    ("POST", "/api/auth/logout"): "Clears the caller's own session cookie; no-op when anonymous.",
    ("GET", "/api/billing/packs"): "DOCBOT-1803: static credit-pack catalogue (names, credits, "
                                   "prices); Stripe price ids / payment links are stripped.",
    ("POST", "/api/marketplace/webhook/shopify"): "Shopify server-to-server webhook; no browser session. "
                                                  "Authenticated by HMAC-SHA256 over the raw body with the "
                                                  "connector's webhook_secret; shop domain header validated "
                                                  "against *.myshopify.com (DOCBOT-1602; further: DOCBOT-1627).",
}

# Handler rejects anonymous callers itself (no Depends guard). Verified by
# test_anonymous_requests_get_401_under_auth_required.
HANDLER_GUARDED: dict[RouteKey, str] = {
    ("GET", "/api/auth/me"): "401 without a valid docbot_session cookie; returns only the caller.",
    ("GET", "/api/auth/workspace"): "401 without a logged-in user; scoped to the caller.",
}

# Return 404 unless ENV=dev. Verified by test_dev_only_routes_404_outside_dev.
DEV_ONLY: dict[RouteKey, str] = {
    ("POST", "/api/test-persona-switch"): "Persona debug helper.",
    ("GET", "/api/test-personas"): "Persona debug helper.",
    ("GET", "/api/auth/debug-urls"): "OAuth redirect-URI debug helper.",
}

# Ratchet: current gaps -> ticket that closes them (YC plan §2, EPIC-16/17).
# Remove an entry once its route is guarded; the test will tell you to.
KNOWN_UNGUARDED: dict[RouteKey, str] = {
    # EDGAR: outbound SEC calls + embedding cost, unmetered (SEC-21).
    ("POST", "/api/edgar/search"): "DOCBOT-1623",
    ("POST", "/api/edgar/filings"): "DOCBOT-1623",
    ("GET", "/api/edgar/cache"): "DOCBOT-1623",
}

_TICKET_RE = re.compile(r"^DOCBOT-\d{3,4}$")


def _fmt(keys: Iterable[RouteKey]) -> str:
    return "\n".join(f"  {m:6} {p}" for m, p in sorted(keys))


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


def test_manifest_buckets_are_disjoint() -> None:
    buckets = {
        "PUBLIC_ROUTES": set(PUBLIC_ROUTES),
        "HANDLER_GUARDED": set(HANDLER_GUARDED),
        "DEV_ONLY": set(DEV_ONLY),
        "KNOWN_UNGUARDED": set(KNOWN_UNGUARDED),
    }
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = buckets[a] & buckets[b]
            assert not overlap, f"Routes listed in both {a} and {b}:\n{_fmt(overlap)}"


def test_known_unguarded_entries_cite_a_ticket() -> None:
    bad = {k: v for k, v in KNOWN_UNGUARDED.items() if not _TICKET_RE.match(v)}
    assert not bad, f"KNOWN_UNGUARDED values must be 'DOCBOT-XXXX' ticket ids: {bad}"


def test_every_route_is_classified() -> None:
    """Fails on any NEW route that has no auth guard and is not listed."""
    listed = set(PUBLIC_ROUTES) | set(HANDLER_GUARDED) | set(DEV_ONLY) | set(KNOWN_UNGUARDED)
    unclassified = {
        key for key, route in _api_routes().items()
        if not _has_guard(route) and key not in listed
    }
    assert not unclassified, (
        "New route(s) without an auth guard dependency:\n"
        f"{_fmt(unclassified)}\n"
        "Add a guard (e.g. Depends(require_role(...)) / require_api_key(...)), or, if the route "
        "is intentionally anonymous, add it to PUBLIC_ROUTES in tests/unit/test_route_guards.py "
        "with a justification. Do not add new entries to KNOWN_UNGUARDED."
    )


def test_known_unguarded_ratchet() -> None:
    """Ratchet: a KNOWN_UNGUARDED entry that got fixed or deleted must be removed."""
    routes = _api_routes()
    missing = {k for k in KNOWN_UNGUARDED if k not in routes}
    now_guarded = {k for k in KNOWN_UNGUARDED if k in routes and _has_guard(routes[k])}
    msgs = []
    if now_guarded:
        msgs.append(
            "These routes are now guarded; delete them from KNOWN_UNGUARDED "
            f"(tickets: {sorted({KNOWN_UNGUARDED[k] for k in now_guarded})}):\n{_fmt(now_guarded)}"
        )
    if missing:
        msgs.append(f"These KNOWN_UNGUARDED routes no longer exist; delete their entries:\n{_fmt(missing)}")
    assert not msgs, "\n\n".join(msgs)


@pytest.mark.parametrize("bucket_name", ["PUBLIC_ROUTES", "HANDLER_GUARDED", "DEV_ONLY"])
def test_listed_routes_exist_and_are_not_double_classified(bucket_name: str) -> None:
    """Stale entries rot the manifest; a listed route that gained a guard should be unlisted."""
    bucket: dict[RouteKey, str] = globals()[bucket_name]
    routes = _api_routes()
    missing = {k for k in bucket if k not in routes}
    guarded = {k for k in bucket if k in routes and _has_guard(routes[k])}
    assert not missing, f"{bucket_name} lists routes that no longer exist:\n{_fmt(missing)}"
    assert not guarded, f"{bucket_name} lists routes that now have a guard; remove them:\n{_fmt(guarded)}"


def test_guard_detection_sees_known_guards() -> None:
    """Sanity check so a refactor can't silently make every route look unguarded."""
    routes = _api_routes()
    for key in [("POST", "/api/chat"), ("GET", "/admin/audit-log"), ("POST", "/v1/hybrid/chat")]:
        assert key in routes, f"expected route {key} to exist"
        assert _has_guard(routes[key]), f"guard detection failed for {key}"
    assert not _is_guard_callable(idx.gate_uploads), "gate_uploads must not count as a guard"


# ---------------------------------------------------------------------------
# Behavioural tests (no network, no DB)
# ---------------------------------------------------------------------------

_PATH_PARAM_RE = re.compile(r"\{[^}]+\}")


def _concrete(path: str) -> str:
    return _PATH_PARAM_RE.sub("route-guard-test-id", path)


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No `with` block: the app lifespan (init_db -> Postgres) never runs.
    yield TestClient(idx.app, raise_server_exceptions=False)


def _guarded_keys() -> list[RouteKey]:
    keys = [k for k, r in _api_routes().items() if _has_guard(r)]
    return sorted(keys) + sorted(HANDLER_GUARDED)


@pytest.mark.parametrize("method,path", _guarded_keys(), ids=lambda v: str(v))
def test_anonymous_requests_get_401_under_auth_required(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    # Keep the abuse-guard middleware from short-circuiting with 403/429.
    monkeypatch.setattr(idx.request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(idx.request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))

    resp = client.request(method, _concrete(path), json={})
    assert resp.status_code == 401, (
        f"{method} {path} returned {resp.status_code} for an anonymous caller with "
        f"AUTH_REQUIRED=true (expected 401): {resp.text[:200]}"
    )


@pytest.mark.parametrize("method,path", sorted(DEV_ONLY), ids=lambda v: str(v))
def test_dev_only_routes_404_outside_dev(client: TestClient, method: str, path: str) -> None:
    if idx._DEV_ROUTES_ENABLED:
        pytest.skip("ENV=dev at import time; dev routes are enabled")
    resp = client.request(method, _concrete(path), json={"session_id": "x", "test_question": "x"})
    assert resp.status_code == 404, f"{method} {path} should 404 outside ENV=dev, got {resp.status_code}"
