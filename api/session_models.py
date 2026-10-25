"""Response models for the chat-session history routes — DOCBOT-1601."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    session_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    persona: Optional[str] = None
    file_count: Optional[int] = None
    files_info: list[dict[str, Any]] = Field(default_factory=list)


class SessionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionDetailResponse(BaseModel):
    session: SessionSummary
    messages: list[SessionMessage]


class SessionDeleteResponse(BaseModel):
    message: str
