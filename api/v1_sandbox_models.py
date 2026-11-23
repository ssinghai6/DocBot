"""Pydantic contracts for DOCBOT-2597 ``POST /v1/sandbox/sessions``.

Zod mirror: ``src/lib/schemas/v1Sandbox.ts`` (identical field names).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class V1SandboxSessionRequest(BaseModel):
    """Body. ``dataset`` is one of quickbite / fuel / tieout (400 ``unsupported_dataset`` otherwise)."""

    dataset: str = Field(default="quickbite", max_length=32)


class V1SandboxSessionResponse(BaseModel):
    """201 body: ids to pass to /v1/hybrid/chat and friends."""

    session_id: str
    connection_id: str
    dataset: str
    expires_at: datetime
