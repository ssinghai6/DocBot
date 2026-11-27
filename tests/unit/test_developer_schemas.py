"""DOCBOT-3905: /apis Zod contracts and error parsing (src/lib/schemas/developer.ts).

No frontend test runner exists, so this evaluates the TS module with the local
``node`` binary (type stripping, no network), like test_security_headers.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - runs the local node binary on a repo file, fixed args
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None or not (REPO_ROOT / "node_modules" / "zod").exists(),
    reason="needs node + installed node_modules",
)

SCRIPT = """
import { register } from "node:module"
// Extensionless relative imports (./v1) resolve to .ts, as the bundler does.
register("data:text/javascript," + encodeURIComponent(
  "export async function resolve(s, c, n) { try { return await n(s, c) } catch (e) {" +
  " if (s.startsWith('.')) return n(s + '.ts', c); throw e } }"))
const s = await import("./src/lib/schemas/developer.ts")
const ts = "2026-11-23T10:00:00Z"
const summary = {id: "k1", prefix: "dk_test_ab", scopes: ["query:run"], created_at: ts, expires_at: ts, last_used_at: null}
const out = {
  v1: s.parseApiError({error: {type: "authentication", code: "invalid_api_key", message: "Invalid key", request_id: "req_1"}}, 401),
  detailObj: s.parseApiError({detail: {code: "email_not_verified", message: "Verify your email to get a test key."}}, 403),
  limit: s.parseApiError({detail: {code: "test_key_limit", message: "x"}}, 409),
  detailStr: s.parseApiError({detail: "Not authenticated"}, 401),
  junk: s.parseApiError("<html>", 502),
  created: s.TestKeyCreateResponseSchema.safeParse({...summary, key: "dk_test_x", mode: "test"}).success,
  noKey: s.TestKeyCreateResponseSchema.safeParse(summary).success,
  revoked: s.TestKeyRevokeResponseSchema.safeParse({status: "revoked", id: "k1"}).success,
}
console.log(JSON.stringify(out))
"""


@pytest.fixture(scope="module")
def out() -> dict:
    res = subprocess.run(  # nosec B603 - fixed argv, local binary
        [NODE, "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(res.stdout)


def test_v1_error_envelope(out: dict) -> None:
    assert out["v1"] == {"code": "invalid_api_key", "message": "Invalid key", "requestId": "req_1"}


def test_fastapi_detail_object_gets_friendly_message(out: dict) -> None:
    assert out["detailObj"]["code"] == "email_not_verified"
    assert "Verify your email" in out["detailObj"]["message"]
    assert out["limit"]["code"] == "test_key_limit"
    assert "2 active test keys" in out["limit"]["message"]


def test_fastapi_detail_string_and_unknown_body(out: dict) -> None:
    assert out["detailStr"] == {"code": None, "message": "Not authenticated", "requestId": None}
    assert out["junk"]["message"] == "Request failed (HTTP 502)."


def test_response_schemas(out: dict) -> None:
    assert out["created"] is True
    assert out["noKey"] is False  # plaintext key required on create
    assert out["revoked"] is True
