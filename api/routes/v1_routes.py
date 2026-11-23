"""Developer API — versioned ``/v1/*`` programmatic routes (DOCBOT-1530, moved by DOCBOT-3901).

Auth: API key (``Authorization: Bearer dk_...``) via ``require_api_key()``,
never the session cookie — a programmatic caller has no browser. Each route
delegates to the same service function the session-cookie frontend route
calls; only the auth dependency and (for hybrid/autopilot) the response shape
differ.

Streaming vs buffered: ``/api/hybrid/chat`` and ``/api/autopilot/run`` return
``text/event-stream`` for the chat UI. A programmatic integration almost
always wants one JSON object, so the ``/v1`` equivalents buffer the same event
stream internally and return a single JSON response. The LLM call still runs
once.

Errors use the ``/v1`` envelope (``api/utils/v1_contract.py``). ``GET
/v1/openapi.json`` is public and lists every ``/v1/*`` route in the app.

Shared tables, caches and helpers are injected once from ``api/index.py`` via
``wire_v1_routes`` (same pattern as the other route modules).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api import credit_service
from api.api_key_service import require_api_key
from api.routes.connector_routes import ConnectorDateRangeRequest
from api.utils.caller_scope import CallerScope
from api.utils.lineage import Lineage
from api.utils.spend_guard import check_platform_capacity
from api.utils.v1_contract import build_v1_openapi
from api.v1_models import (
    V1AutopilotRequest,
    V1AutopilotResponse,
    V1ConnectorListResponse,
    V1ConnectorRegisterRequest,
    V1ConnectorRegisterResponse,
    V1ConnectorSyncResponse,
    V1ErrorResponse,
    V1HybridChatRequest,
    V1HybridChatResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["v1-developer-api"],
    responses={"4XX": {"model": V1ErrorResponse}, "5XX": {"model": V1ErrorResponse}},
)

_v1_hybrid_read = Depends(require_api_key("hybrid:read"))
_v1_autopilot_run = Depends(require_api_key("autopilot:run"))
_v1_lineage_read = Depends(require_api_key("lineage:read"))
_v1_connectors_manage = Depends(require_api_key("connectors:manage"))
_spend_cap = Depends(check_platform_capacity)

_REQUIRED_DEPS = frozenset({
    "sessions_table", "db_connections_table", "messages_table", "schema_cache_table",
    "query_history_table", "query_embeddings_table", "session_artifacts_table",
    "table_embeddings_table", "org_encryption_keys_table", "audit_log_table",
    "answer_lineage_table", "async_session_factory", "expert_personas", "vector_stores",
    "extracted_fields", "charge_budget", "run_row_accessible", "app",
})
_d: dict[str, Any] = {}


def wire_v1_routes(**deps: Any) -> None:
    """Inject index.py's tables, caches and helpers. Called once at import of index.py."""
    missing = _REQUIRED_DEPS - deps.keys()
    if missing:
        raise ValueError(f"wire_v1_routes missing: {sorted(missing)}")
    _d.update(deps)


async def _require_v1_resources(key: Any, session_id: str, connection_id: Optional[str]) -> None:
    """DOCBOT-1603: body-supplied session/connection ids must be in the key's org → else 404."""
    from api.resource_scope_service import v1_resources_in_org
    ok = await v1_resources_in_org(
        getattr(key, "org_id", None),
        session_id,
        connection_id,
        _d["sessions_table"],
        _d["db_connections_table"],
        _d["async_session_factory"],
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Session or connection not found.")


async def _begin_v1_billing(
    raw_request: Request, route: str, key: Any, session_id: str, connection_id: Optional[str],
) -> "credit_service.BillingRun":
    """DOCBOT-2330: reserve credits against the API key's org (402 only when BILLING_ENFORCE).

    Call outside any try that rewrites errors, so a 402 propagates unchanged.
    """
    return await credit_service.begin_billing(
        raw_request, route, session_id=session_id, connection_id=connection_id,
        session_exists=session_id in _d["vector_stores"],
        scope=CallerScope(user_id=None, org_id=key.org_id, anon_token=None),
    )


def _parse_sse_event(chunk: str) -> Optional[dict]:
    """Parse one `data: {...}\\n\\n` SSE line back into its dict payload."""
    if not chunk.startswith("data: "):
        return None
    try:
        return json.loads(chunk[len("data: "):].strip())
    except json.JSONDecodeError:
        return None


def _get_connector(connector_id: str) -> Any:
    """Retrieve a registered connector by ID or raise 404."""
    connector = getattr(_d["app"].state, "connectors", {}).get(connector_id)
    if connector is None:
        raise HTTPException(status_code=404, detail="Connector not found.")
    return connector


@router.get("/v1/openapi.json", include_in_schema=False)
async def v1_openapi(request: Request) -> JSONResponse:
    """Public spec of the /v1 surface only (PUBLIC_ROUTES; describes no data)."""
    spec = getattr(request.app.state, "v1_openapi", None)
    if spec is None:
        spec = build_v1_openapi(request.app)
        request.app.state.v1_openapi = spec
    return JSONResponse(spec)


@router.post(
    "/v1/hybrid/chat",
    response_model=V1HybridChatResponse,
    summary="Hybrid Docs+DB chat (buffered)",
    description=(
        "Runs the same intent-classification → SQL + RAG → synthesis pipeline as "
        "/api/hybrid/chat, buffered into a single JSON response instead of SSE. "
        "Requires an API key with the `hybrid:read` scope."
    ),
)
async def v1_hybrid_chat(
    raw_request: Request, request: V1HybridChatRequest, _key=_v1_hybrid_read, _cap=_spend_cap,
):
    from api.hybrid_service import hybrid_chat
    from api.audit_service import log_event, AuditEventType, get_client_ip

    await _require_v1_resources(_key, request.session_id, request.connection_id)
    await _d["charge_budget"](raw_request, request.session_id, "hybrid")

    log_event(
        AuditEventType.query,
        _d["audit_log_table"],
        _d["async_session_factory"],
        session_id=request.session_id,
        detail=request.question[:500],
        metadata={
            "source": "v1_hybrid_chat",
            "connection_id": request.connection_id,
            "persona": request.persona,
            "api_key_id": _key.id,
            "ip_address": get_client_ip(raw_request),
        },
        org_id=_key.org_id,
    )

    _hybrid_history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    answer_parts: list[str] = []
    citations: list[dict] = []
    metadata_evt: dict = {}
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    error_detail: Optional[str] = None

    billing = await _begin_v1_billing(raw_request, "hybrid", _key, request.session_id, request.connection_id)
    async with billing:
        try:
            async for chunk in hybrid_chat(
                question=request.question,
                session_id=request.session_id,
                connection_id=request.connection_id,
                persona=request.persona,
                has_docs=request.has_docs,
                messages_table=_d["messages_table"],
                sessions_table=_d["sessions_table"],
                db_connections_table=_d["db_connections_table"],
                schema_cache_table=_d["schema_cache_table"],
                query_history_table=_d["query_history_table"],
                query_embeddings_table=_d["query_embeddings_table"],
                async_session_factory=_d["async_session_factory"],
                expert_personas=_d["expert_personas"],
                vector_stores=_d["vector_stores"],
                extracted_fields=_d["extracted_fields"].get(request.session_id),
                chat_history=_hybrid_history if _hybrid_history else None,
                chart_type=request.chart_type,
            ):
                evt = _parse_sse_event(chunk)
                if not evt:
                    continue
                etype = evt.get("type")
                if etype == "metadata":
                    # DOCBOT-2592: a later SQL-summary metadata event has no intent
                    # keys; merge so it can't null out intent/has_sql/has_docs.
                    metadata_evt.update(evt)
                elif etype == "token":
                    answer_parts.append(evt.get("content", ""))
                elif etype == "replace":
                    # DOCBOT-2592: discrepancy correction is a full rewrite; tokens
                    # after it (tie-out marker tail) still append.
                    answer_parts[:] = [evt.get("content", "")]
                elif etype == "done":
                    citations = evt.get("citations", [])
                    run_id = evt.get("run_id")
                    trace_id = evt.get("trace_id")
                elif etype == "error":
                    error_detail = evt.get("detail") or evt.get("content") or "An internal error occurred."
        except Exception as exc:
            logger.error("v1_hybrid_chat error: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="An internal error occurred.") from exc

    if error_detail:
        raise HTTPException(status_code=502, detail=error_detail)

    return {
        "answer": "".join(answer_parts),
        "citations": citations,
        "intent": metadata_evt.get("intent"),
        "has_sql": metadata_evt.get("has_sql"),
        "has_docs": metadata_evt.get("has_docs"),
        "run_id": run_id,
        "trace_id": trace_id,
    }


@router.post(
    "/v1/autopilot/run",
    response_model=V1AutopilotResponse,
    summary="Analytical Autopilot run (buffered)",
    description=(
        "Runs the same plan → execute (≤5 steps) → synthesize Autopilot pipeline as "
        "/api/autopilot/run, buffered into a single JSON response instead of SSE. "
        "Requires an API key with the `autopilot:run` scope."
    ),
)
async def v1_autopilot_run(
    raw_request: Request, request: V1AutopilotRequest, _key=_v1_autopilot_run, _cap=_spend_cap,
):
    from api.autopilot_service import run_autopilot
    from api.audit_service import log_event, AuditEventType, get_client_ip

    await _require_v1_resources(_key, request.session_id, request.connection_id)
    await _d["charge_budget"](raw_request, request.session_id, "autopilot")

    log_event(
        AuditEventType.query,
        _d["audit_log_table"],
        _d["async_session_factory"],
        session_id=request.session_id,
        detail=request.question[:500],
        metadata={
            "source": "v1_autopilot",
            "connection_id": request.connection_id or "",
            "persona": request.persona,
            "api_key_id": _key.id,
            "ip_address": get_client_ip(raw_request),
        },
        org_id=_key.org_id,
    )

    _autopilot_history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    plan_steps: list[dict] = []
    steps: list[dict] = []
    answer = ""
    citations: list[dict] = []
    warnings: list[str] = []
    error_detail: Optional[str] = None

    billing = await _begin_v1_billing(raw_request, "autopilot", _key, request.session_id, request.connection_id)
    async with billing:
        try:
            # Resolve data sources from persisted state — client-sent flags
            # are fallback hints only (see api/utils/autopilot_sources.py).
            from api.utils.autopilot_sources import SourceFlags, resolve_autopilot_sources
            sources = await resolve_autopilot_sources(
                session_id=request.session_id,
                connection_id=request.connection_id,
                db_connections_table=_d["db_connections_table"],
                async_session_factory=_d["async_session_factory"],
                vector_stores=_d["vector_stores"],
                org_encryption_keys_table=_d["org_encryption_keys_table"],
                fallback=SourceFlags(
                    has_docs=request.has_docs,
                    has_db=request.has_db,
                    has_csv=request.has_csv,
                ),
            )

            async for chunk in run_autopilot(
                question=request.question,
                session_id=request.session_id,
                connection_id=request.connection_id or "",
                persona=request.persona,
                db_connections_table=_d["db_connections_table"],
                schema_cache_table=_d["schema_cache_table"],
                query_history_table=_d["query_history_table"],
                query_embeddings_table=_d["query_embeddings_table"],
                session_artifacts_table=_d["session_artifacts_table"],
                table_embeddings_table=_d["table_embeddings_table"],
                async_session_factory=_d["async_session_factory"],
                expert_personas=_d["expert_personas"],
                vector_stores=_d["vector_stores"],
                has_docs=sources.has_docs,
                has_db=sources.has_db,
                has_csv=sources.has_csv,
                chat_history=_autopilot_history if _autopilot_history else None,
            ):
                evt = _parse_sse_event(chunk)
                if not evt:
                    continue
                etype = evt.get("type")
                if etype == "plan":
                    plan_steps = evt.get("steps", [])
                elif etype == "step":
                    steps.append(evt)
                elif etype == "answer":
                    answer = evt.get("content", "")
                elif etype == "done":
                    citations = evt.get("citations", [])
                elif etype == "warning":
                    warnings.append(evt.get("content", ""))
                elif etype == "error":
                    error_detail = evt.get("content") or "An internal error occurred."
        except Exception as exc:
            logger.error("v1_autopilot_run error: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="An internal error occurred.") from exc

    if error_detail:
        raise HTTPException(status_code=502, detail=error_detail)

    return {
        "plan": plan_steps,
        "steps": steps,
        "answer": answer,
        "citations": citations,
        "warnings": warnings,
    }


@router.get(
    "/v1/lineage/{run_id}",
    response_model=Lineage,
    summary="Read answer lineage",
    description="Returns the persisted lineage for one answer, joined with its LLM calls. Requires `lineage:read`.",
)
async def v1_get_lineage(run_id: str, _key=_v1_lineage_read):
    from api.lineage_service import get_lineage
    # DOCBOT-1620: only runs on sessions/connections owned by users of the key's org.
    key_scope = CallerScope(user_id=None, org_id=getattr(_key, "org_id", None), anon_token=None)
    if not key_scope.org_id or not await _d["run_row_accessible"](_d["answer_lineage_table"], run_id, key_scope):
        raise HTTPException(status_code=404, detail="Lineage not found.")
    lineage = await get_lineage(run_id)
    if lineage is None:
        raise HTTPException(status_code=404, detail="Lineage not found.")
    return lineage.model_dump(mode="json")


@router.get(
    "/v1/connectors",
    response_model=V1ConnectorListResponse,
    summary="List marketplace connectors",
    description="List the API key's organization's active marketplace connectors (metadata only, no credentials). Requires `connectors:manage`.",
)
async def v1_list_connectors(_key=_v1_connectors_manage):
    from api.connector_store import list_active_connectors
    connectors = await list_active_connectors(org_id=_key.org_id)
    return {"connectors": connectors}


@router.post(
    "/v1/connectors",
    response_model=V1ConnectorRegisterResponse,
    summary="Register a marketplace connector",
    description=(
        "Registers and tests a commerce connector (Amazon SP-API, Shopify, ...). "
        "The connector is owned by the API key's organization. Requires `connectors:manage`."
    ),
)
async def v1_register_connector(body: V1ConnectorRegisterRequest, _key=_v1_connectors_manage):
    from api.connectors.registry import get_connector_class
    from api.connectors.base import ConnectorCredentials

    try:
        cls = get_connector_class(body.connector_type)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    creds = ConnectorCredentials(connector_type=body.connector_type, credentials=body.credentials)
    connector = cls(creds)

    try:
        ok = await connector.test_connection()
    except Exception as exc:
        logger.warning("v1 connector test_connection failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Connection test failed: {exc}")
    if not ok:
        raise HTTPException(status_code=502, detail="Connector test_connection returned False — check credentials.")

    connector_id = str(uuid.uuid4())
    app_state = _d["app"].state
    if not hasattr(app_state, "connectors"):
        app_state.connectors = {}
    app_state.connectors[connector_id] = connector

    from api.connector_store import save_connector
    await save_connector(connector_id, body.connector_type, body.credentials, org_id=_key.org_id)

    from api.connectors.sync_jobs import register_sync_schedules
    register_sync_schedules(connector_id)

    logger.info("v1 registered %s connector: %s (api_key=%s)", body.connector_type, connector_id, _key.id)
    return {"connector_id": connector_id, "connector_type": body.connector_type, "status": "connected"}


@router.post(
    "/v1/connectors/{connector_id}/sync",
    response_model=V1ConnectorSyncResponse,
    summary="Sync a marketplace connector",
    description="Fetches orders + financials and persists them to the commerce tables. Requires `connectors:manage`.",
)
async def v1_sync_connector(connector_id: str, body: ConnectorDateRangeRequest, _key=_v1_connectors_manage):
    from api.commerce_service import sync_connector_data
    from api.connector_store import connector_in_org, update_last_sync

    # DOCBOT-2591: 404 (not 403) for other-org / org-less connectors — don't leak existence.
    if not await connector_in_org(connector_id, _key.org_id):
        raise HTTPException(status_code=404, detail="Connector not found.")
    connector = _get_connector(connector_id)
    try:
        summary = await sync_connector_data(connector_id, connector, body.start_date, body.end_date)
    except Exception as exc:
        logger.error("v1 sync failed for %s: %s", connector_id, exc)
        raise HTTPException(status_code=502, detail=f"Sync failed: {exc}")

    await update_last_sync(connector_id)
    return summary
