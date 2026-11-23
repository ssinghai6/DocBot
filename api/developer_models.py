"""Pydantic contracts for DOCBOT-2598 self-serve test API keys.

Zod mirror: ``src/lib/schemas/developer.ts`` (identical field names).
The raw key appears only in ``TestKeyCreateResponse`` — never in a listing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class TestKeySummary(BaseModel):
    """One of the caller's active test keys. No secret material."""

    __test__ = False  # not a pytest test class

    id: str
    prefix: str
    scopes: list[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]


class TestKeyCreateResponse(TestKeySummary):
    """POST /api/developer/test-key. ``key`` is shown exactly once."""

    key: str
    mode: Literal["test"]


class TestKeyListResponse(BaseModel):
    """GET /api/developer/test-keys."""

    keys: list[TestKeySummary]


class TestKeyRevokeResponse(BaseModel):
    """DELETE /api/developer/test-key/{id}."""

    status: Literal["revoked"]
    id: str
