"""Post-signup onboarding — DOCBOT-1712.

``POST /api/auth/onboarding`` stores the two answers from /onboarding:
the workspace name (written onto the caller's organization) and the primary
use case (written onto the user row). Signed-in users only, in every mode.

The workspace rename applies only to an org the caller administers and never
to the legacy shared default org (it is located by its fixed name).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import update

from api.org_service import DEFAULT_ORG_NAME
from api.rbac_service import UserRole, require_role

router = APIRouter(prefix="/api/auth", tags=["auth"])

USER_ONBOARDING_COLUMN_MIGRATIONS: list[str] = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS use_case TEXT",
]

_state: dict[str, Any] = {}


def wire_onboarding(users_table: Any, organizations_table: Any, async_session_factory: Any) -> None:
    """Bind shared tables + session factory at include time (see email_verify_routes)."""
    _state["users_table"] = users_table
    _state["organizations_table"] = organizations_table
    _state["async_session_factory"] = async_session_factory


class OnboardingRequest(BaseModel):
    workspace_name: Optional[str] = Field(default=None, max_length=80)
    use_case: Optional[Literal["tie_out", "close_reconciliation", "investor_reporting", "other"]] = None


class OnboardingResponse(BaseModel):
    status: Literal["ok"]
    workspace_renamed: bool


@router.post("/onboarding", response_model=OnboardingResponse)
async def save_onboarding(
    body: OnboardingRequest,
    user: Any = Depends(require_role(UserRole.viewer, enforce_in_open_mode=True)),
) -> OnboardingResponse:
    users_t = _state["users_table"]
    orgs_t = _state["organizations_table"]
    name = (body.workspace_name or "").strip()
    renamed = False
    async with _state["async_session_factory"]() as session:
        async with session.begin():
            if body.use_case is not None:
                await session.execute(
                    update(users_t).where(users_t.c.id == user.id).values(use_case=body.use_case)
                )
            if name and user.org_id and user.role == "admin" and name != DEFAULT_ORG_NAME:
                result = await session.execute(
                    update(orgs_t)
                    .where(orgs_t.c.id == user.org_id, orgs_t.c.name != DEFAULT_ORG_NAME)
                    .values(name=name)
                )
                renamed = result.rowcount > 0
    return OnboardingResponse(status="ok", workspace_renamed=renamed)
