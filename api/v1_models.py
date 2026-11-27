"""Public ``/v1`` API contracts (Pydantic v2) — DOCBOT-3901, EPIC-39.

Mirrored field-for-field by ``src/lib/schemas/v1.ts`` (Zod);
``tests/unit/test_v1_models.py`` fails if a field drifts between them.

Request models keep the exact fields and defaults the handlers accepted when
they lived in ``api/index.py`` (DOCBOT-1530), so moving them is not a
contract change. Loosely-typed pipeline payloads (citations, plan steps,
connector rows, sync summaries) stay ``dict`` and allow extra keys so the
response_model never drops a field the handler returned before.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class V1ChatMessage(BaseModel):
    role: str
    content: str


class V1HybridChatRequest(BaseModel):
    question: str
    session_id: str
    connection_id: Optional[str] = None
    persona: str = "Data Analyst"
    has_docs: bool = True
    history: List[V1ChatMessage] = []
    chart_type: str = "auto"


class V1HybridChatResponse(BaseModel):
    answer: str
    citations: Optional[List[Dict[str, Any]]] = []
    intent: Optional[str] = None
    has_sql: Optional[bool] = None
    has_docs: Optional[bool] = None
    run_id: Optional[str] = None
    trace_id: Optional[str] = None


class V1AutopilotRequest(BaseModel):
    session_id: str
    connection_id: Optional[str] = None
    question: str
    persona: str = "Generalist"
    # Client hints only — resolved server-side in resolve_autopilot_sources().
    has_docs: bool = False
    has_db: bool = False
    has_csv: bool = False
    history: List[V1ChatMessage] = []


class V1AutopilotResponse(BaseModel):
    plan: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    answer: str = ""
    citations: Optional[List[Dict[str, Any]]] = []
    warnings: List[str] = []


class V1ConnectorRegisterRequest(BaseModel):
    connector_type: str
    credentials: Dict[str, str]


class V1ConnectorRegisterResponse(BaseModel):
    connector_id: str
    connector_type: str
    status: str


class V1ConnectorListResponse(BaseModel):
    connectors: List[Dict[str, Any]]


class V1ConnectorSyncResponse(BaseModel):
    """``commerce_service.sync_connector_data`` summary (counts); keys pass through."""

    model_config = ConfigDict(extra="allow")


ErrorType = Literal["invalid_request", "authentication", "permission", "not_found", "rate_limit", "server"]


class V1Error(BaseModel):
    type: ErrorType
    code: str
    message: str
    request_id: str


class V1ErrorResponse(BaseModel):
    """Every ``/v1`` error body (``api/utils/v1_contract.py``)."""

    error: V1Error


__all__ = [
    "ErrorType",
    "V1AutopilotRequest",
    "V1AutopilotResponse",
    "V1ChatMessage",
    "V1ConnectorListResponse",
    "V1ConnectorRegisterRequest",
    "V1ConnectorRegisterResponse",
    "V1ConnectorSyncResponse",
    "V1Error",
    "V1ErrorResponse",
    "V1HybridChatRequest",
    "V1HybridChatResponse",
]
