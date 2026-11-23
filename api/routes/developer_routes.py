"""Self-serve personal test API keys — DOCBOT-2598.

Session-cookie routes for the ``/apis`` portal. A signed-in user with a
verified email mints a ``dk_test_`` key for themselves (preset scopes,
30-day expiry, max 2 active). Live keys stay admin-only (``/admin/api-keys``).
The raw key is returned once and never stored or logged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api import api_key_service as keys
from api.developer_models import (
    TestKeyCreateResponse,
    TestKeyListResponse,
    TestKeyRevokeResponse,
    TestKeySummary,
)
from api.rbac_service import UserRole, require_role

router = APIRouter(prefix="/api/developer", tags=["developer"])

# Viewer+ in every mode: a test key is personal and sandbox-only.
_signed_in = Depends(require_role(UserRole.viewer, enforce_in_open_mode=True))


def _verified_user(user: Any = _signed_in) -> Any:
    if not getattr(user, "email_verified", False):
        raise HTTPException(
            status_code=403,
            detail={"code": "email_not_verified", "message": "Verify your email to get a test key."},
        )
    if not getattr(user, "org_id", None):
        raise HTTPException(
            status_code=403,
            detail={"code": "no_organization", "message": "Your account is not attached to an organization."},
        )
    return user


def _summary(record: keys.APIKeyRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "prefix": record.key_prefix,
        "scopes": record.scopes,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
        "last_used_at": record.last_used_at,
    }


@router.post("/test-key", response_model=TestKeyCreateResponse, status_code=201)
async def create_test_key(user: Any = Depends(_verified_user)) -> TestKeyCreateResponse:
    # shortcut: count-then-insert is not atomic; two parallel mints can reach 3. Fine for a personal sandbox key.
    if await keys.count_active_test_keys(user.id) >= keys.MAX_ACTIVE_TEST_KEYS_PER_USER:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "test_key_limit",
                "message": f"You already have {keys.MAX_ACTIVE_TEST_KEYS_PER_USER} active test keys. Revoke one first.",
            },
        )
    record, raw_key = await keys.create_api_key(
        org_id=user.org_id,
        scopes=list(keys.TEST_KEY_SCOPES),
        name="Personal test key",
        mode="test",
        expires_at=datetime.now(timezone.utc) + keys.TEST_KEY_TTL,
        created_by=user.id,
    )
    return TestKeyCreateResponse(**_summary(record), key=raw_key, mode="test")


@router.get("/test-keys", response_model=TestKeyListResponse)
async def list_test_keys(user: Any = Depends(_verified_user)) -> TestKeyListResponse:
    records = await keys.list_test_keys(user.id)
    return TestKeyListResponse(keys=[TestKeySummary(**_summary(r)) for r in records])


@router.delete("/test-key/{key_id}", response_model=TestKeyRevokeResponse)
async def revoke_test_key(key_id: str, user: Any = Depends(_verified_user)) -> TestKeyRevokeResponse:
    if not await keys.revoke_api_key(key_id, created_by=user.id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Test key not found."})
    return TestKeyRevokeResponse(status="revoked", id=key_id)
