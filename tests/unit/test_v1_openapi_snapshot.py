"""DOCBOT-3901: GET /v1/openapi.json lists only /v1 routes and matches the committed snapshot.

The snapshot (``api/openapi/v1.json``) changes whenever a /v1 route or model
changes, including routes added by other router modules. Review the diff, then
regenerate with:

    UPDATE_V1_OPENAPI_SNAPSHOT=1 pytest tests/unit/test_v1_openapi_snapshot.py -q

Generate it with the pinned requirements (fastapi/pydantic versions change the spec).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api.utils.v1_contract import build_v1_openapi  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parents[2] / "api" / "openapi" / "v1.json"


def _served() -> dict:
    resp = TestClient(idx.app).get("/v1/openapi.json")  # no Authorization: public
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_spec_lists_only_v1_routes() -> None:
    spec = _served()
    assert spec["paths"], "expected /v1 paths"
    assert all(p.startswith("/v1/") for p in spec["paths"]), sorted(spec["paths"])
    assert "/v1/hybrid/chat" in spec["paths"]
    assert "/v1/openapi.json" not in spec["paths"]


def test_spec_declares_bearer_auth_with_key_prefixes() -> None:
    spec = _served()
    scheme = spec["components"]["securitySchemes"]["bearerAuth"]
    assert (scheme["type"], scheme["scheme"]) == ("http", "bearer")
    assert "dk_test_" in scheme["description"] and "dk_live_" in scheme["description"]
    assert spec["security"] == [{"bearerAuth": []}]
    for item in spec["paths"].values():
        for op in item.values():
            assert all(p.get("name", "").lower() != "authorization" for p in op.get("parameters", []))


def test_spec_includes_routes_from_any_router() -> None:
    from fastapi import APIRouter, FastAPI

    app = FastAPI()
    extra = APIRouter()

    @extra.post("/v1/sandbox/sessions")
    async def _sandbox() -> dict:
        return {}

    @app.get("/api/internal")
    async def _internal() -> dict:
        return {}

    app.include_router(extra)
    assert list(build_v1_openapi(app)["paths"]) == ["/v1/sandbox/sessions"]


def test_spec_matches_committed_snapshot() -> None:
    spec = _served()
    if os.environ.get("UPDATE_V1_OPENAPI_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    assert SNAPSHOT.exists(), "missing api/openapi/v1.json; regenerate (see module docstring)"
    assert spec == json.loads(SNAPSHOT.read_text()), (
        "/v1 OpenAPI spec drifted from api/openapi/v1.json; review and regenerate (see module docstring)"
    )
