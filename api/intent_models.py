"""Intent router contracts (Pydantic v2) — DOCBOT-2728.

``POST /api/route-intent`` decides, before a chat turn runs, whether it needs
the fast path or Autopilot, which source it targets, what kind of task it is,
and whether the attached sources can actually answer it. Mirrored 1:1 (same
field names) by ``src/lib/schemas/intent.ts`` (Zod).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

IntentRoute = Literal["fast", "autopilot"]
IntentSource = Literal["doc", "sql", "hybrid"]
IntentTask = Literal["lookup", "compare", "forecast", "what_if", "diagnose", "viz"]


class IntentTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field(max_length=32)
    content: str = Field(default="", max_length=20_000)


class RouteIntentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Only the last 3 turns reach the prompt; the client may send more.
    history: list[IntentTurn] = Field(default_factory=list, max_length=20)
    session_id: Optional[str] = Field(default=None, max_length=200)
    connection_id: Optional[str] = Field(default=None, max_length=200)


class RouteIntentResponse(BaseModel):
    """``confidence == 0`` marks the deterministic regex fallback."""

    model_config = ConfigDict(extra="ignore")

    route: IntentRoute
    source: IntentSource
    task: IntentTask
    data_ready: bool
    missing: Optional[str] = Field(default=None, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
