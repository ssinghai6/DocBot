"""DOCBOT-2581 — waitlist service + routes + Zod parity. SQLite, no network."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

import api.index as idx  # noqa: E402
from api import waitlist_models, waitlist_service  # noqa: E402
from api.routes import waitlist_routes  # noqa: E402
from api.utils import request_guard  # noqa: E402

TS_FILE = Path(__file__).resolve().parents[2] / "src" / "lib" / "schemas" / "waitlist.ts"
GOOD = {"email": "Ann@Example.com", "name": "Ann", "company": "Acme CPA", "source": "landing"}
NOTIFY_ENV = ("RESEND_API_KEY", "EMAIL_FROM", "WAITLIST_NOTIFY_EMAIL")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, object]]:
    path = tmp_path / "wl.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    idx.metadata.create_all(sync_engine, tables=[idx.waitlist_table])
    factory = async_sessionmaker(create_async_engine(f"sqlite+aiosqlite:///{path}"), expire_on_commit=False)
    monkeypatch.setattr(waitlist_routes, "_state", {"table": idx.waitlist_table, "async_session_factory": factory})
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    for k in NOTIFY_ENV:
        monkeypatch.delenv(k, raising=False)
    yield TestClient(idx.app), sync_engine
    sync_engine.dispose()


def _rows(engine):
    with engine.connect() as conn:
        return conn.execute(select(idx.waitlist_table)).mappings().all()


def test_valid_submit_stores_lowercased(env):
    client, engine = env
    r = client.post("/api/waitlist", json=GOOD)
    assert r.status_code == 202 and r.json() == {"ok": True}
    rows = _rows(engine)
    assert len(rows) == 1 and rows[0]["email"] == "ann@example.com" and rows[0]["source"] == "landing"


def test_duplicate_email_upserts(env):
    client, engine = env
    client.post("/api/waitlist", json=GOOD)
    r = client.post(
        "/api/waitlist",
        json={**GOOD, "email": "ann@example.com", "note": "Need 5 seats", "source": "out_of_credits"},
    )
    assert r.status_code == 202
    rows = _rows(engine)
    assert len(rows) == 1 and rows[0]["note"] == "Need 5 seats" and rows[0]["source"] == "out_of_credits"


def test_signed_in_user_ids_recorded(env):
    client, engine = env
    user = SimpleNamespace(id="u1", org_id="o1", role="viewer", email="ann@example.com")
    with patch("api.rbac_service.resolve_session_user", new_callable=AsyncMock, return_value=user):
        client.post("/api/waitlist", json=GOOD)
    row = _rows(engine)[0]
    assert row["user_id"] == "u1" and row["org_id"] == "o1"


def test_honeypot_returns_202_and_stores_nothing(env):
    client, engine = env
    r = client.post("/api/waitlist", json={**GOOD, "website": "http://spam.example"})
    assert r.status_code == 202 and r.json() == {"ok": True}
    assert _rows(engine) == []


@pytest.mark.parametrize(
    "override",
    [
        {"email": "not-an-email"},
        {"name": ""},
        {"name": "x" * 121},
        {"company": "x" * 121},
        {"role": "x" * 121},
        {"note": "x" * 1001},
        {"source": "elsewhere"},
    ],
)
def test_validation_422(env, override):
    client, engine = env
    assert client.post("/api/waitlist", json={**GOOD, **override}).status_code == 422
    assert _rows(engine) == []


def test_admin_list_requires_admin(env):
    client, _ = env
    assert client.get("/api/admin/waitlist").status_code == 401


def test_admin_list_returns_entries(env):
    client, _ = env
    client.post("/api/waitlist", json=GOOD)
    client.post("/api/waitlist", json={**GOOD, "email": "bob@example.com", "name": "Bob"})
    admin = MagicMock(id="op1", role="admin")
    with patch("api.rbac_service._resolve_user_with_role", new_callable=AsyncMock, return_value=admin), patch(
        "api.rbac_service.platform_admin_ids", return_value={"op1"}
    ):
        r = client.get("/api/admin/waitlist", cookies={"docbot_session": "t"})
    assert r.status_code == 200
    assert sorted(e["email"] for e in r.json()["entries"]) == ["ann@example.com", "bob@example.com"]


@pytest.mark.asyncio
async def test_notify_noop_when_unset(monkeypatch):
    for k in NOTIFY_ENV:
        monkeypatch.delenv(k, raising=False)
    with patch("api.waitlist_service.httpx.AsyncClient") as client_cls:
        await waitlist_service.notify_founder(waitlist_models.WaitlistRequest(**GOOD))
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_notify_posts_when_set_and_swallows_errors(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "k")
    monkeypatch.setenv("EMAIL_FROM", "DocBot <a@b.co>")
    monkeypatch.setenv("WAITLIST_NOTIFY_EMAIL", "me@b.co")
    body = waitlist_models.WaitlistRequest(**GOOD, note="secret note")
    with patch("api.waitlist_service.httpx.AsyncClient") as client_cls:
        post = client_cls.return_value.__aenter__.return_value.post
        post.return_value = MagicMock(raise_for_status=MagicMock())
        await waitlist_service.notify_founder(body)
        assert post.await_count == 1
        sent = post.await_args.kwargs["json"]
        assert sent["to"] == ["me@b.co"] and "secret note" not in sent["text"]
        post.side_effect = httpx.ConnectError("boom")
        await waitlist_service.notify_founder(body)  # must not raise


def test_zod_mirror_has_same_fields():
    ts = TS_FILE.read_text()
    for model in (waitlist_models.WaitlistRequest, waitlist_models.WaitlistEntry):
        for field in model.model_fields:
            assert re.search(rf"\b{field}\b", ts), f"{field} missing from waitlist.ts"
