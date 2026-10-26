"""Pydantic contracts for DOCBOT-1713 email verification routes.

Zod mirror: ``src/lib/schemas/emailVerify.ts`` (identical field names).

Messages returned to unauthenticated callers are deliberately generic — the
verify endpoint must not distinguish "unknown token" from "expired" or
"already used" (that would leak when a link was clicked).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    """One-shot token from the emailed link (body or ?token= query param)."""

    token: str = Field(min_length=1, max_length=512)


class VerifyResponse(BaseModel):
    """POST /api/auth/verify success body."""

    status: Literal["verified"]
    email: str


# NOTE on POST /api/auth/register (existing route, lives in api/index.py):
# dormant mode keeps today's body {"status": "registered", "email": ...};
# with RESEND_API_KEY + EMAIL_FROM set it answers 202 with
# {"status": "verification_required", "email": ...} and the request gains an
# optional `turnstile_token` field — mirrored in
# src/lib/schemas/emailVerify.ts (RegisterRequestSchema).
