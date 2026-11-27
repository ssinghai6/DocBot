"""Answer recovery contracts (Pydantic v2) — DOCBOT-2636.

``GET /api/hybrid/answer/{run_id}`` returns the persisted final answer of a
hybrid turn whose SSE stream broke off in the browser. Mirrored 1:1 (same
field names) by ``src/lib/schemas/answer.ts`` (Zod).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from api.utils.lineage import Lineage


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = ""
    page: Optional[int] = None
    text: str = ""


class RecoveredAnswer(BaseModel):
    """``pending`` while the turn is still running; ``ready`` once persisted."""

    run_id: str
    status: Literal["pending", "ready"]
    answer: Optional[str] = None
    citations: list[AnswerCitation] = Field(default_factory=list)
    trace_id: Optional[str] = None
    lineage: Optional[Lineage] = None
