"""DOCBOT-1602 (SEC-03): connector + commerce routes RBAC, org scoping, SSRF.

Runs the connector router on a standalone FastAPI app backed by a temp
SQLite file — no network, no Postgres, no API keys.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import MetaData, create_engine, insert, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import api.auth_service as auth_service
import api.connector_store as connector_store
from api.connectors.base import ConnectorCredentials
from api.connectors.shopify_connector import (
    InvalidShopDomainError,
    ShopifyConnector,
    validate_shop_domain,
)
from api.routes import connector_routes
from api.routes.connector_routes import ConnectorRegisterRequest

ORG_X = "org-x"
ORG_Y = "org-y"
CONN_X = "conn-x"
CONN_Y = "conn-y"

USERS = {
    "tok-analyst-x": SimpleNamespace(id="u-ax", role="analyst", org_id=ORG_X),
    "tok-admin-x": SimpleNamespace(id="u-adx", role="admin", org_id=ORG_X),
    "tok-analyst-y": SimpleNamespace(id="u-ay", role="analyst", org_id=ORG_Y),
    "tok-viewer-x": SimpleNamespace(id="u-vx", role="viewer", org_id=ORG_X),
    "tok-analyst-noorg": SimpleNamespace(id="u-n", role="analyst", org_id=None),
}

BAD_SHOP_DOMAINS = [
    "evil.com",
    "x.myshopify.com.evil.com",
    "169.254.169.254",
    "x.myshopify.com/@evil",
    "x.myshopify.com@evil.com",
    "x.myshopify.com:8443",
    "x.myshopify.com\nevil.com",
    "a.myshopify.com\n.evil.com",
    "-x.myshopify.com",
    "myshopify.com",
    "",
]


class FakeConnector:
    connector_type = "amazon"

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.calls = 0

    async def fetch_orders(self, start: str, end: str) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return [{"marketplace_order_id": "1"}]

    async def fetch_financials(self, start: str, end: str) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return []


@pytest.fixture
def env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)  # open mode: must still enforce

    db_file = tmp_path / "conn.db"
    metadata = MetaData()
    table = connector_store.register_connector_tables(metadata)
    sync_engine = create_engine(f"sqlite:///{db_file}")
    metadata.create_all(sync_engine)
    with sync_engine.begin() as c:
        for cid, org in ((CONN_X, ORG_X), (CONN_Y, ORG_Y), ("conn-legacy", None)):
            c.execute(insert(table).values(
                id=cid, connector_type="amazon", encrypted_credentials="x", is_active=True, org_id=org,
            ))

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}", poolclass=NullPool)
    factory = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    connector_store.wire_connector_store(table, factory)

    async def fake_get_user(token, *_a, **_k):
        return USERS.get(token)

    monkeypatch.setattr(auth_service, "get_user_from_session", fake_get_user)

    app = FastAPI()
    app.include_router(connector_routes.router)
    app.state.connectors = {CONN_X: FakeConnector(), CONN_Y: FakeConnector(), "conn-legacy": FakeConnector()}

    yield {"app": app, "table": table, "sync_engine": sync_engine}

    connector_store.wire_connector_store(None, None)
    sync_engine.dispose()


def _client(env: dict[str, Any], token: str | None = None) -> TestClient:
    client = TestClient(env["app"], raise_server_exceptions=False)
    if token:
        client.cookies.set("docbot_session", token)
    return client


ROUTES = [
    ("POST", "/api/connectors/register", {"connector_type": "amazon", "credentials": {}}),
    ("GET", "/api/connectors", None),
    ("GET", "/api/connectors/types", None),
    ("DELETE", f"/api/connectors/{CONN_X}", None),
    ("POST", f"/api/connectors/{CONN_X}/orders", {"start_date": "2024-01-01", "end_date": "2024-01-31"}),
    ("POST", f"/api/connectors/{CONN_X}/financials", {"start_date": "2024-01-01", "end_date": "2024-01-31"}),
    ("POST", f"/api/connectors/{CONN_X}/sync", {"start_date": "2024-01-01", "end_date": "2024-01-31"}),
    ("GET", f"/api/commerce/{CONN_X}/orders", None),
    ("GET", f"/api/commerce/{CONN_X}/financials", None),
]


# ---------------------------------------------------------------------------
# AuthN / AuthZ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,path,body", ROUTES, ids=lambda v: str(v))
def test_anonymous_gets_401_in_open_mode(env, method, path, body) -> None:
    resp = _client(env).request(method, path, json=body)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path,body", ROUTES, ids=lambda v: str(v))
def test_viewer_gets_403(env, method, path, body) -> None:
    resp = _client(env, "tok-viewer-x").request(method, path, json=body)
    assert resp.status_code == 403


def test_expired_session_gets_401(env) -> None:
    assert _client(env, "tok-unknown").get("/api/connectors").status_code == 401


def test_analyst_without_org_gets_403(env) -> None:
    assert _client(env, "tok-analyst-noorg").get("/api/connectors").status_code == 403


def test_types_listed_for_analyst(env) -> None:
    resp = _client(env, "tok-analyst-x").get("/api/connectors/types")
    assert resp.status_code == 200
    assert "shopify" in resp.json()["types"]


# ---------------------------------------------------------------------------
# Org scoping
# ---------------------------------------------------------------------------


def test_list_is_scoped_to_caller_org(env) -> None:
    ids_x = {c["connector_id"] for c in _client(env, "tok-analyst-x").get("/api/connectors").json()["connectors"]}
    ids_y = {c["connector_id"] for c in _client(env, "tok-analyst-y").get("/api/connectors").json()["connectors"]}
    assert ids_x == {CONN_X}
    assert ids_y == {CONN_Y}


def test_admin_list_still_org_scoped(env) -> None:
    ids = {c["connector_id"] for c in _client(env, "tok-admin-x").get("/api/connectors").json()["connectors"]}
    assert ids == {CONN_X}


@pytest.mark.parametrize("method,path,body", [
    ("POST", f"/api/connectors/{CONN_Y}/orders", {"start_date": "a", "end_date": "b"}),
    ("POST", f"/api/connectors/{CONN_Y}/financials", {"start_date": "a", "end_date": "b"}),
    ("POST", f"/api/connectors/{CONN_Y}/sync", {"start_date": "a", "end_date": "b"}),
    ("GET", f"/api/commerce/{CONN_Y}/orders", None),
    ("GET", f"/api/commerce/{CONN_Y}/financials", None),
    ("GET", "/api/commerce/conn-legacy/orders", None),
    ("POST", "/api/connectors/conn-legacy/orders", {"start_date": "a", "end_date": "b"}),
], ids=lambda v: str(v))
def test_other_org_connector_is_404(env, method, path, body) -> None:
    resp = _client(env, "tok-analyst-x").request(method, path, json=body)
    assert resp.status_code == 404
    assert env["app"].state.connectors[CONN_Y].calls == 0


def test_cannot_delete_other_org_connector(env) -> None:
    resp = _client(env, "tok-analyst-x").delete(f"/api/connectors/{CONN_Y}")
    assert resp.status_code == 404
    with env["sync_engine"].connect() as c:
        row = c.execute(select(env["table"].c.is_active).where(env["table"].c.id == CONN_Y)).first()
    assert row.is_active is True or row.is_active == 1
    assert CONN_Y in env["app"].state.connectors


def test_delete_own_connector(env, monkeypatch) -> None:
    import api.connectors.sync_jobs as sync_jobs
    monkeypatch.setattr(sync_jobs, "deregister_sync_schedules", lambda _cid: None)
    resp = _client(env, "tok-analyst-x").delete(f"/api/connectors/{CONN_X}")
    assert resp.status_code == 200
    assert CONN_X not in env["app"].state.connectors


def test_own_connector_orders_ok(env) -> None:
    resp = _client(env, "tok-analyst-x").post(
        f"/api/connectors/{CONN_X}/orders", json={"start_date": "a", "end_date": "b"},
    )
    assert resp.status_code == 200
    assert resp.json()["order_count"] == 1


def test_register_stamps_caller_org(env, monkeypatch) -> None:
    import api.connectors.registry as registry
    import api.connectors.sync_jobs as sync_jobs

    class OkConnector:
        connector_type = "amazon"

        def __init__(self, creds: ConnectorCredentials) -> None:
            self.creds = creds

        async def test_connection(self) -> bool:
            return True

    monkeypatch.setattr(registry, "get_connector_class", lambda _t: OkConnector)
    monkeypatch.setattr(sync_jobs, "register_sync_schedules", lambda _cid: None)
    resp = _client(env, "tok-analyst-y").post(
        "/api/connectors/register", json={"connector_type": "amazon", "credentials": {"k": "v"}},
    )
    assert resp.status_code == 200, resp.text
    new_id = resp.json()["connector_id"]
    with env["sync_engine"].connect() as c:
        org = c.execute(select(env["table"].c.org_id).where(env["table"].c.id == new_id)).scalar_one()
    assert org == ORG_Y


# ---------------------------------------------------------------------------
# No raw exception text to clients
# ---------------------------------------------------------------------------


def test_upstream_error_text_not_echoed(env) -> None:
    env["app"].state.connectors[CONN_X] = FakeConnector(RuntimeError("Shopify API error 500: token=shpat_SECRET"))
    resp = _client(env, "tok-analyst-x").post(
        f"/api/connectors/{CONN_X}/orders", json={"start_date": "a", "end_date": "b"},
    )
    assert resp.status_code == 502
    assert "SECRET" not in resp.text and "RuntimeError" not in resp.text


def test_register_failure_text_not_echoed(env, monkeypatch) -> None:
    import api.connectors.registry as registry

    class BoomConnector:
        def __init__(self, creds: ConnectorCredentials) -> None:
            pass

        async def test_connection(self) -> bool:
            raise RuntimeError("upstream said: secret-body")

    monkeypatch.setattr(registry, "get_connector_class", lambda _t: BoomConnector)
    resp = _client(env, "tok-analyst-x").post(
        "/api/connectors/register", json={"connector_type": "amazon", "credentials": {}},
    )
    assert resp.status_code == 502
    assert "secret-body" not in resp.text


def test_unknown_connector_type_generic_400(env) -> None:
    resp = _client(env, "tok-analyst-x").post(
        "/api/connectors/register", json={"connector_type": "nope<script>", "credentials": {}},
    )
    assert resp.status_code == 400
    assert "script" not in resp.text


# ---------------------------------------------------------------------------
# Shopify shop_domain SSRF validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", BAD_SHOP_DOMAINS, ids=repr)
def test_validate_shop_domain_rejects(domain: str) -> None:
    with pytest.raises(InvalidShopDomainError):
        validate_shop_domain(domain)


@pytest.mark.parametrize("domain", BAD_SHOP_DOMAINS, ids=repr)
def test_pydantic_model_rejects_bad_shop_domain(domain: str) -> None:
    with pytest.raises(ValidationError):
        ConnectorRegisterRequest(connector_type="shopify", credentials={"shop_domain": domain, "access_token": "t"})


@pytest.mark.parametrize("domain", BAD_SHOP_DOMAINS, ids=repr)
def test_connector_refuses_bad_shop_domain(domain: str) -> None:
    conn = ShopifyConnector(ConnectorCredentials("shopify", {"shop_domain": domain, "access_token": "t"}))
    with pytest.raises(ValueError):
        conn._base_url()


@pytest.mark.parametrize("domain", ["evil.com", "x.myshopify.com.evil.com", "169.254.169.254", "x.myshopify.com/@evil"])
def test_register_route_rejects_bad_shop_domain(env, domain: str) -> None:
    resp = _client(env, "tok-analyst-x").post(
        "/api/connectors/register",
        json={"connector_type": "shopify", "credentials": {"shop_domain": domain, "access_token": "t"}},
    )
    assert resp.status_code == 422


def test_shop_domain_normalized_and_accepted() -> None:
    req = ConnectorRegisterRequest(
        connector_type="Shopify", credentials={"shop_domain": " My-Store.myshopify.com ", "access_token": "t"},
    )
    assert req.connector_type == "shopify"
    assert req.credentials["shop_domain"] == "my-store.myshopify.com"


def test_bad_api_version_rejected() -> None:
    with pytest.raises(ValidationError):
        ConnectorRegisterRequest(
            connector_type="shopify",
            credentials={"shop_domain": "a.myshopify.com", "access_token": "t", "api_version": "../../x"},
        )


def test_pagination_link_to_other_host_is_ignored() -> None:
    conn = ShopifyConnector(ConnectorCredentials("shopify", {"shop_domain": "a.myshopify.com", "access_token": "t"}))
    for url in ("https://169.254.169.254/x", "http://a.myshopify.com/x", "https://a.myshopify.com@evil.com/x"):
        headers = SimpleNamespace(get=lambda k, d="", u=url: f'<{u}>; rel="next"' if k == "link" else d)
        assert conn._extract_next_page_url(headers) is None


# ---------------------------------------------------------------------------
# Shopify webhook stays public + HMAC-authenticated
# ---------------------------------------------------------------------------


def _shopify_env(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import api.commerce_service as commerce_service

    synced: list[str] = []

    async def fake_sync(cid, _conn, _s, _e):
        synced.append(cid)
        return {"orders": 0}

    monkeypatch.setattr(commerce_service, "sync_connector_data", fake_sync)
    env["app"].state.connectors["conn-shop"] = ShopifyConnector(ConnectorCredentials(
        "shopify", {"shop_domain": "store.myshopify.com", "access_token": "t", "webhook_secret": "whsec"},
    ))
    return synced


def _sign(body: bytes, secret: str) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


def test_webhook_valid_hmac_without_session(env, monkeypatch) -> None:
    synced = _shopify_env(env, monkeypatch)
    body = b'{"id": 1}'
    resp = _client(env).post("/api/marketplace/webhook/shopify", content=body, headers={
        "X-Shopify-Hmac-SHA256": _sign(body, "whsec"),
        "X-Shopify-Shop-Domain": "store.myshopify.com",
        "X-Shopify-Topic": "orders/create",
    })
    assert resp.status_code == 200, resp.text
    assert synced == ["conn-shop"]


def test_webhook_bad_hmac_401(env, monkeypatch) -> None:
    synced = _shopify_env(env, monkeypatch)
    resp = _client(env).post("/api/marketplace/webhook/shopify", content=b"{}", headers={
        "X-Shopify-Hmac-SHA256": _sign(b"{}", "wrong"),
        "X-Shopify-Shop-Domain": "store.myshopify.com",
    })
    assert resp.status_code == 401
    assert synced == []


def test_webhook_bad_domain_header_401(env, monkeypatch) -> None:
    _shopify_env(env, monkeypatch)
    resp = _client(env).post("/api/marketplace/webhook/shopify", content=b"{}", headers={
        "X-Shopify-Hmac-SHA256": "x", "X-Shopify-Shop-Domain": "evil.com",
    })
    assert resp.status_code == 401
    assert "evil" not in resp.text



# ---------------------------------------------------------------------------
# DOCBOT-2591: /v1/connectors developer-API routes scoped to the API key's org
# ---------------------------------------------------------------------------


def _v1_key(org_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=f"key-{org_id}", org_id=org_id)


def _range() -> connector_routes.ConnectorDateRangeRequest:
    return connector_routes.ConnectorDateRangeRequest(start_date="2024-01-01", end_date="2024-01-31")


@pytest.fixture
def v1(env, monkeypatch: pytest.MonkeyPatch) -> Any:
    import api.index as index

    monkeypatch.setattr(index.app.state, "connectors", env["app"].state.connectors, raising=False)
    return index


async def test_v1_list_scoped_to_key_org(v1) -> None:
    ids_x = {c["connector_id"] for c in (await v1.v1_list_connectors(_key=_v1_key(ORG_X)))["connectors"]}
    ids_y = {c["connector_id"] for c in (await v1.v1_list_connectors(_key=_v1_key(ORG_Y)))["connectors"]}
    assert ids_x == {CONN_X}
    assert ids_y == {CONN_Y}


@pytest.mark.parametrize("cid", [CONN_Y, "conn-legacy", "does-not-exist"])
async def test_v1_sync_other_org_or_orgless_is_404(v1, env, cid: str) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await v1.v1_sync_connector(cid, _range(), _key=_v1_key(ORG_X))
    assert exc.value.status_code == 404
    assert all(c.calls == 0 for c in env["app"].state.connectors.values())


async def test_v1_sync_own_connector_ok(v1, monkeypatch) -> None:
    import api.commerce_service as commerce_service

    async def fake_sync(connector_id: str, connector: Any, start: str, end: str) -> dict[str, str]:
        return {"connector_id": connector_id}

    async def noop(_cid: str) -> None:
        return None

    monkeypatch.setattr(commerce_service, "sync_connector_data", fake_sync)
    monkeypatch.setattr(connector_store, "update_last_sync", noop)
    assert await v1.v1_sync_connector(CONN_X, _range(), _key=_v1_key(ORG_X)) == {"connector_id": CONN_X}


async def test_v1_register_stamps_key_org(v1, env, monkeypatch) -> None:
    import api.connectors.registry as registry
    import api.connectors.sync_jobs as sync_jobs

    class OkConnector:
        def __init__(self, creds: ConnectorCredentials) -> None:
            self.creds = creds

        async def test_connection(self) -> bool:
            return True

    monkeypatch.setattr(registry, "get_connector_class", lambda _t: OkConnector)
    monkeypatch.setattr(sync_jobs, "register_sync_schedules", lambda _cid: None)

    resp = await v1.v1_register_connector(
        v1.V1ConnectorRegisterRequest(connector_type="amazon", credentials={"k": "v"}),
        _key=_v1_key(ORG_Y),
    )
    new_id = resp["connector_id"]
    with env["sync_engine"].connect() as c:
        org = c.execute(select(env["table"].c.org_id).where(env["table"].c.id == new_id)).scalar_one()
    assert org == ORG_Y
    ids_x = {c["connector_id"] for c in (await v1.v1_list_connectors(_key=_v1_key(ORG_X)))["connectors"]}
    assert new_id not in ids_x
    v1.app.state.connectors.pop(new_id, None)
