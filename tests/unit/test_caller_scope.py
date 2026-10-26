"""DOCBOT-1601 (SEC-02): CallerScope + owner-scoped session routes.

Covers the helper (scope_filter / owner_values / anon cookie / resolution)
and the four session routes end to end against SQLite:
anon-A vs anon-B, user in org X vs org Y, org-admin behaviour, NULL-owner
legacy rows, and delete of a foreign id -> 404 with the row left intact.
No network, no PostgreSQL.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Column, MetaData, String, Table, create_engine, insert, select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from starlette.requests import Request  # noqa: E402

import api.index as idx  # noqa: E402
import api.rbac_service as rbac  # noqa: E402
from api.utils import caller_scope as cs  # noqa: E402
from api.utils.caller_scope import (  # noqa: E402
    ANON_COOKIE_NAME,
    CallerScope,
    anon_token_digest,
    owner_values,
    scope_filter,
)

ANON_A = "a" * 43
ANON_B = "b" * 43
ORG_X = "org-x"
ORG_Y = "org-y"

USERS: dict[str, Any] = {
    "tok-alice": SimpleNamespace(id="alice", org_id=ORG_X, role="analyst"),
    "tok-bob": SimpleNamespace(id="bob", org_id=ORG_Y, role="analyst"),
    "tok-carol": SimpleNamespace(id="carol", org_id=ORG_X, role="analyst"),
    "tok-admin-x": SimpleNamespace(id="admin-x", org_id=ORG_X, role="admin"),
    "tok-admin-default": SimpleNamespace(id="admin-d", org_id="default-org", role="admin"),
}


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def _sql(clause: Any) -> str:
    return str(clause.compile(compile_kwargs={"literal_binds": True}))


def test_scope_filter_user_non_admin_is_owner_and_org() -> None:
    sql = _sql(scope_filter(idx.sessions_table, CallerScope("u1", ORG_X, None, False)))
    assert "sessions.user_id = 'u1'" in sql
    assert "sessions.org_id = 'org-x'" in sql


def test_scope_filter_user_without_org_is_owner_only() -> None:
    sql = _sql(scope_filter(idx.sessions_table, CallerScope("u1", None, None, False)))
    assert sql == "sessions.user_id = 'u1'"


def test_scope_filter_admin_is_whole_org_never_unscoped() -> None:
    sql = _sql(scope_filter(idx.sessions_table, CallerScope("adm", ORG_X, None, True)))
    assert sql == "sessions.org_id = 'org-x' AND sessions.user_id IS NOT NULL"
    # Admin without an org falls back to their own rows, never everything.
    sql = _sql(scope_filter(idx.sessions_table, CallerScope("adm", None, None, True)))
    assert sql == "sessions.user_id = 'adm'"


def test_scope_filter_anonymous_matches_anon_owner_and_null_user() -> None:
    sql = _sql(scope_filter(idx.sessions_table, CallerScope(None, None, "d1", False)))
    assert "sessions.anon_owner = 'd1'" in sql
    assert "sessions.user_id IS NULL" in sql


def test_scope_filter_no_identity_matches_nothing() -> None:
    assert _sql(scope_filter(idx.sessions_table, CallerScope(None, None, None, False))) == "false"


def test_scope_filter_anonymous_on_table_without_anon_owner_matches_nothing() -> None:
    t = Table("t", MetaData(), Column("user_id", String), Column("org_id", String))
    assert _sql(scope_filter(t, CallerScope(None, None, "d1", False))) == "false"


def test_owner_values() -> None:
    assert owner_values(CallerScope("u", ORG_X, None, True)) == {
        "user_id": "u", "org_id": ORG_X, "anon_owner": None,
    }
    assert owner_values(CallerScope(None, None, "d", False)) == {
        "user_id": None, "org_id": None, "anon_owner": "d",
    }


def test_anon_digest_is_not_the_raw_cookie() -> None:
    digest = anon_token_digest(ANON_A)
    assert digest != ANON_A and len(digest) == 64


def _request(cookies: Optional[dict[str, str]] = None) -> Request:
    header = "; ".join(f"{k}={v}" for k, v in (cookies or {}).items())
    headers = [(b"cookie", header.encode())] if header else []
    return Request({"type": "http", "method": "GET", "path": "/api/x", "headers": headers, "state": {}})


def test_prime_anon_cookie_issues_once_and_is_visible_in_same_request() -> None:
    req = _request()
    token = cs.prime_anon_cookie(req)
    assert token and len(token) >= 32
    assert cs._raw_anon_cookie(req) == token
    assert cs.prime_anon_cookie(_request({ANON_COOKIE_NAME: ANON_A})) is None


def test_short_or_malformed_anon_cookie_is_ignored() -> None:
    assert cs._raw_anon_cookie(_request({ANON_COOKIE_NAME: "x"})) is None
    assert cs.prime_anon_cookie(_request({ANON_COOKIE_NAME: "short"})) is not None


@pytest.fixture
def fake_users(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _resolve(token: Optional[str]) -> Any:
        return USERS.get(token or "")

    monkeypatch.setattr(rbac, "resolve_session_user", _resolve)


async def test_resolve_scope_user(fake_users: None) -> None:
    scope = await cs.resolve_caller_scope(_request({"docbot_session": "tok-admin-x"}))
    assert scope == CallerScope("admin-x", ORG_X, None, True)


async def test_resolve_scope_anonymous(fake_users: None) -> None:
    scope = await cs.resolve_caller_scope(_request({ANON_COOKIE_NAME: ANON_A}))
    assert scope == CallerScope(None, None, anon_token_digest(ANON_A), False)


async def test_get_caller_scope_401_under_auth_required(
    fake_users: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    with pytest.raises(HTTPException) as exc:
        await cs.get_caller_scope(_request({ANON_COOKIE_NAME: ANON_A}))
    assert exc.value.status_code == 401
    scope = await cs.get_caller_scope(_request({"docbot_session": "tok-alice"}))
    assert scope.user_id == "alice"


async def test_get_caller_scope_open_mode_allows_anonymous(
    fake_users: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    scope = await cs.get_caller_scope(_request())
    assert scope.is_anonymous and scope.anon_token is None


async def test_resolve_session_user_unwired_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rbac, "_users_table", None)
    assert await rbac.resolve_session_user("anything") is None
    assert await rbac.resolve_session_user(None) is None


# ---------------------------------------------------------------------------
# Route tests (SQLite)
# ---------------------------------------------------------------------------

ROWS = [
    # session_id, user_id, org_id, anon_owner
    ("s-anon-a", None, "default-org", anon_token_digest(ANON_A)),
    ("s-anon-b", None, "default-org", anon_token_digest(ANON_B)),
    ("s-alice", "alice", ORG_X, None),
    ("s-carol", "carol", ORG_X, None),
    ("s-bob", "bob", ORG_Y, None),
    ("s-legacy-x", None, ORG_X, None),      # pre-1601 row backfilled into org X
    ("s-legacy-null", None, None, None),    # pre-1601 row, no org at all
]


DELETED_STORES: list[str] = []  # vector-store deletions recorded by the db fixture


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_users: None) -> Iterator[str]:
    path = tmp_path / "sessions.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    idx.metadata.create_all(sync_engine, tables=[idx.sessions_table, idx.messages_table])
    with sync_engine.begin() as conn:
        for sid, uid, org, anon in ROWS:
            conn.execute(insert(idx.sessions_table).values(
                session_id=sid, user_id=uid, org_id=org, anon_owner=anon,
                persona="Generalist", file_count=1, files_info='[{"filename": "f.pdf"}]',
            ))
            conn.execute(insert(idx.messages_table).values(
                session_id=sid, role="user", content=f"secret of {sid}", sources="[]",
            ))
    sync_engine.dispose()

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(idx, "async_session_factory", async_sessionmaker(async_engine, expire_on_commit=False))
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(idx.request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(idx.request_guard, "check_ip_rate_limit", lambda *_a, **_k: None)
    import api.utils.vector_store as vs
    monkeypatch.setattr(vs, "delete_store", DELETED_STORES.append)
    DELETED_STORES.clear()
    yield str(path)


def _row_exists(path: str, session_id: str) -> bool:
    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        found = conn.execute(
            select(idx.sessions_table.c.session_id).where(idx.sessions_table.c.session_id == session_id)
        ).fetchone()
        msgs = conn.execute(
            select(idx.messages_table.c.id).where(idx.messages_table.c.session_id == session_id)
        ).fetchall()
    engine.dispose()
    return found is not None and len(msgs) == 1


def _client(**cookies: str) -> TestClient:
    client = TestClient(idx.app, raise_server_exceptions=True)
    for k, v in cookies.items():
        client.cookies.set(k, v)
    return client


def anon_a() -> TestClient:
    return _client(**{ANON_COOKIE_NAME: ANON_A})


def anon_b() -> TestClient:
    return _client(**{ANON_COOKIE_NAME: ANON_B})


def user(token: str) -> TestClient:
    return _client(docbot_session=token)


def _listed(client: TestClient) -> set[str]:
    resp = client.get("/api/sessions")
    assert resp.status_code == 200, resp.text
    return {s["session_id"] for s in resp.json()["sessions"]}


def test_list_is_scoped_per_caller(db: str) -> None:
    assert _listed(anon_a()) == {"s-anon-a"}
    assert _listed(anon_b()) == {"s-anon-b"}
    assert _listed(user("tok-alice")) == {"s-alice"}
    assert _listed(user("tok-bob")) == {"s-bob"}


def test_admin_sees_own_org_users_only_not_unowned_or_other_orgs(db: str) -> None:
    assert _listed(user("tok-admin-x")) == {"s-alice", "s-carol"}
    assert user("tok-admin-x").get("/api/session/s-carol").status_code == 200


def test_default_org_admin_never_sees_backfilled_anonymous_sessions(db: str) -> None:
    """DOCBOT-2317: backfill stamps default-org on anon rows; admin must not see them."""
    admin = user("tok-admin-default")
    assert _listed(admin) == set()
    for sid in ("s-anon-a", "s-anon-b"):
        assert admin.get(f"/api/session/{sid}").status_code == 404
        assert admin.get(f"/api/export/{sid}").status_code == 404
        assert admin.delete(f"/api/session/{sid}").status_code == 404
        assert _row_exists(db, sid)
    assert anon_a().get("/api/session/s-anon-a").status_code == 200


def test_null_owner_rows_are_not_visible_to_non_admins(db: str) -> None:
    fresh = _client()  # no cookies at all
    assert _listed(fresh) == set()
    for c in (anon_a(), user("tok-alice"), user("tok-carol"), fresh):
        assert c.get("/api/session/s-legacy-x").status_code == 404
        assert c.get("/api/session/s-legacy-null").status_code == 404
    assert user("tok-admin-x").get("/api/session/s-legacy-null").status_code == 404
    assert user("tok-admin-x").get("/api/session/s-legacy-x").status_code == 404


def test_first_anonymous_request_gets_anon_owner_cookie(db: str) -> None:
    resp = _client().get("/api/sessions")
    set_cookie = resp.headers.get("set-cookie", "")
    assert ANON_COOKIE_NAME in set_cookie and "httponly" in set_cookie.lower()
    resp = anon_a().get("/api/sessions")
    assert ANON_COOKIE_NAME not in resp.headers.get("set-cookie", "")


@pytest.mark.parametrize(
    "client_factory,own,foreign",
    [
        (anon_a, "s-anon-a", "s-anon-b"),
        (anon_b, "s-anon-b", "s-anon-a"),
        (lambda: user("tok-alice"), "s-alice", "s-bob"),     # org X vs org Y
        (lambda: user("tok-alice"), "s-alice", "s-carol"),   # same org, other user
        (lambda: user("tok-bob"), "s-bob", "s-alice"),
        (lambda: user("tok-alice"), "s-alice", "s-anon-a"),  # user vs anon row
        (anon_a, "s-anon-a", "s-alice"),                     # anon vs user row
    ],
)
def test_get_and_export_own_ok_foreign_404(client_factory: Any, own: str, foreign: str, db: str) -> None:
    client = client_factory()
    resp = client.get(f"/api/session/{own}")
    assert resp.status_code == 200
    assert resp.json()["messages"][0]["content"] == f"secret of {own}"
    for fmt in ("txt", "json", "markdown"):
        ok = client.get(f"/api/export/{own}", params={"format": fmt})
        assert ok.status_code == 200
        assert f"secret of {own}" in ok.text

    resp = client.get(f"/api/session/{foreign}")
    assert resp.status_code == 404
    assert "secret" not in resp.text
    for fmt in ("txt", "json", "markdown"):
        resp = client.get(f"/api/export/{foreign}", params={"format": fmt})
        assert resp.status_code == 404
        assert "secret" not in resp.text


def test_export_bad_format_400(db: str) -> None:
    assert anon_a().get("/api/export/s-anon-a", params={"format": "pdf"}).status_code == 400


@pytest.mark.parametrize(
    "client_factory,foreign",
    [
        (anon_a, "s-anon-b"),
        (anon_b, "s-anon-a"),
        (lambda: user("tok-alice"), "s-bob"),
        (lambda: user("tok-alice"), "s-carol"),
        (lambda: user("tok-admin-x"), "s-bob"),   # admin cannot cross orgs
        (anon_a, "s-legacy-null"),
        (anon_a, "does-not-exist"),
    ],
)
def test_delete_foreign_id_404_and_row_survives(client_factory: Any, foreign: str, db: str) -> None:
    resp = client_factory().delete(f"/api/session/{foreign}")
    assert resp.status_code == 404
    if foreign != "does-not-exist":
        assert _row_exists(db, foreign)
    assert foreign not in DELETED_STORES  # vector store untouched


def test_delete_own_session(db: str) -> None:
    resp = anon_a().delete("/api/session/s-anon-a")
    assert resp.status_code == 200
    engine = create_engine(f"sqlite:///{db}")
    with engine.connect() as conn:
        assert conn.execute(
            select(idx.sessions_table).where(idx.sessions_table.c.session_id == "s-anon-a")
        ).fetchone() is None
        assert conn.execute(
            select(idx.messages_table).where(idx.messages_table.c.session_id == "s-anon-a")
        ).fetchone() is None
    engine.dispose()
    assert DELETED_STORES == ["s-anon-a"]
    assert _row_exists(db, "s-anon-b")


def test_admin_can_delete_within_own_org(db: str) -> None:
    assert user("tok-admin-x").delete("/api/session/s-carol").status_code == 200
    assert not _row_exists(db, "s-carol")


def test_routes_401_for_anonymous_under_auth_required(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    client = anon_a()
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/session/s-anon-a").status_code == 401
    assert client.get("/api/export/s-anon-a").status_code == 401
    assert client.delete("/api/session/s-anon-a").status_code == 401
    assert _row_exists(db, "s-anon-a")
    assert _listed(user("tok-alice")) == {"s-alice"}
