"""``POST /v1/sandbox/sessions`` — DOCBOT-2597.

Creates a ready-to-query demo session (``api/demo_service.py``) owned by the
API key's user + org and marked ``is_sandbox``. The starting point for test
keys; live keys may use it for demos. Logic lives in
``api/sandbox_session_service.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from api.api_key_service import require_api_key
from api.sandbox_session_service import create_sandbox_session
from api.v1_sandbox_models import V1SandboxSessionRequest, V1SandboxSessionResponse

router = APIRouter(prefix="/v1/sandbox", tags=["v1"])


@router.post("/sessions", response_model=V1SandboxSessionResponse, status_code=201)
async def create_v1_sandbox_session(
    body: V1SandboxSessionRequest,
    key: Any = Depends(require_api_key("sessions:write")),
) -> V1SandboxSessionResponse:
    return V1SandboxSessionResponse(**await create_sandbox_session(key, body.dataset))
