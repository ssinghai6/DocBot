"""DOCBOT-1712 — POST /api/auth/onboarding (auth required, saves fields). SQLite, no network."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, insert, select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

import api.index as idx  # noqa: E402
from api.org_service import DEFAULT_ORG_NAME  # noqa: E402
from api.routes import onboarding_routes  # noqa: E402

TABLES = [idx.organizations_table, idx.users_table]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, object]]:
    path = tmp_path / "onb.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    idx.metadata.create_all(sync_engine, tables=TABLES)
    with sync_engine.begin() as conn:
        conn.execute(insert(idx.organizations_table).values(id="org-a", name="Ann's workspace"))
        conn.execute(insert(idx.organizations_table).values(id="org-d", name=DEFAULT_ORG_NAME))
        for uid, org, role in (("ann", "org-a", "admin"), ("dan", "org-d", "admin"), ("vic", "org-a", "viewer")):
            conn.execute(insert(idx.users_table).values(
                id=uid, email=f"{uid}@x.com", name=uid, provider="email", role=role, org_id=org,
            ))
    factory = async_sessionmaker(create_async_engine(f"sqlite+aiosqlite:///{path}"), expire_on_commit=False)
    monkeypatch.setattr(onboarding_routes, "_state", {
        "users_table": idx.users_table,
        "organizations_table": idx.organizations_table,
        "async_session_factory": factory,
    })
    yield TestClient(idx.app), sync_engine
    sync_engine.dispose()


def _as(uid: str, org: str, role: str):
    return patch(
        "api.auth_service.get_user_from_session", new_callable=AsyncMock,
        return_value=SimpleNamespace(id=uid, org_id=org, role=role),
    )


def _row(engine, table, key: str):
    with engine.connect() as conn:
        return conn.execute(select(table).where(table.c.id == key)).fetchone()


def test_anonymous_gets_401_even_in_open_mode(env, monkeypatch):
    client, _ = env
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    resp = client.post("/api/auth/onboarding", json={"use_case": "tie_out"})
    assert resp.status_code == 401


def test_saves_workspace_name_and_use_case(env):
    client, engine = env
    client.cookies.set("docbot_session", "tok")
    with _as("ann", "org-a", "admin"):
        resp = client.post(
            "/api/auth/onboarding",
            json={"workspace_name": "  Acme Finance ", "use_case": "close_reconciliation"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "workspace_renamed": True}
    assert _row(engine, idx.organizations_table, "org-a").name == "Acme Finance"
    assert _row(engine, idx.users_table, "ann").use_case == "close_reconciliation"


def test_rejects_unknown_use_case(env):
    client, _ = env
    client.cookies.set("docbot_session", "tok")
    with _as("ann", "org-a", "admin"):
        resp = client.post("/api/auth/onboarding", json={"use_case": "crypto"})
    assert resp.status_code == 422


def test_never_renames_default_org_or_as_non_admin(env):
    client, engine = env
    client.cookies.set("docbot_session", "tok")
    with _as("dan", "org-d", "admin"):
        resp = client.post("/api/auth/onboarding", json={"workspace_name": "Mine now"})
    assert resp.json()["workspace_renamed"] is False
    assert _row(engine, idx.organizations_table, "org-d").name == DEFAULT_ORG_NAME
    with _as("vic", "org-a", "viewer"):
        resp = client.post("/api/auth/onboarding", json={"workspace_name": "Hijack", "use_case": "other"})
    assert resp.json()["workspace_renamed"] is False
    assert _row(engine, idx.organizations_table, "org-a").name == "Ann's workspace"
    assert _row(engine, idx.users_table, "vic").use_case == "other"
