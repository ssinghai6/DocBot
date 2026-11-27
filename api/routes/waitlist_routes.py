"""Waitlist routes — DOCBOT-2581 (F-4).

``POST /api/waitlist`` is public (PUBLIC_ROUTES): the global per-IP limiter in
``abuse_guard`` already covers every ``POST /api/*``, plus a honeypot field.
``GET /api/admin/waitlist`` is platform-operator only (cross-org data).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Request

from api import waitlist_service
from api.rbac_service import require_platform_admin
from api.utils.caller_scope import resolve_caller_scope
from api.waitlist_models import WaitlistEntry, WaitlistList, WaitlistRequest, WaitlistResponse

router = APIRouter(tags=["waitlist"])
# Module-level so tests/unit/test_admin_auth_guard.py can match it by identity.
_rbac_platform_admin = Depends(require_platform_admin())

_state: dict[str, Any] = {}
_bg: set[asyncio.Task] = set()


def wire_waitlist(table: Any, async_session_factory: Any) -> None:
    _state["table"] = table
    _state["async_session_factory"] = async_session_factory


@router.post("/api/waitlist", status_code=202, response_model=WaitlistResponse)
async def join_waitlist(body: WaitlistRequest, request: Request) -> WaitlistResponse:
    if body.website:  # honeypot: pretend success, store nothing
        return WaitlistResponse()
    scope = await resolve_caller_scope(request)
    await waitlist_service.upsert_entry(
        _state["table"], _state["async_session_factory"], body, scope.user_id, scope.org_id
    )
    task = asyncio.create_task(waitlist_service.notify_founder(body))
    _bg.add(task)
    task.add_done_callback(_bg.discard)
    return WaitlistResponse()


@router.get("/api/admin/waitlist", response_model=WaitlistList)
async def list_waitlist(_user: Any = _rbac_platform_admin) -> WaitlistList:
    rows = await waitlist_service.list_entries(_state["table"], _state["async_session_factory"])
    return WaitlistList(entries=[WaitlistEntry(**r) for r in rows])
