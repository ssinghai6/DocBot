"""Pydantic contracts for the DOCBOT-2581 waitlist routes.

Zod mirror: ``src/lib/schemas/waitlist.ts`` (identical field names).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Deliberately loose (no email-validator dependency); the founder replies by hand.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

WaitlistSource = Literal["out_of_credits", "landing"]


class WaitlistRequest(BaseModel):
    email: str = Field(max_length=254, pattern=EMAIL_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    company: str = Field(min_length=1, max_length=120)
    role: Optional[str] = Field(default=None, max_length=120)
    note: Optional[str] = Field(default=None, max_length=1000)
    source: WaitlistSource
    # Honeypot: real users never see this field; bots fill it.
    website: Optional[str] = Field(default=None, max_length=500)

    @field_validator("email")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()


class WaitlistResponse(BaseModel):
    ok: Literal[True] = True


class WaitlistEntry(BaseModel):
    id: str
    email: str
    name: str
    company: str
    role: Optional[str] = None
    note: Optional[str] = None
    source: WaitlistSource
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class WaitlistList(BaseModel):
    entries: list[WaitlistEntry]
