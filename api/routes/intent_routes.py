"""Intent router — DOCBOT-2728.

``POST /api/route-intent`` — viewer+, same principal gate as the chat routes.
The source summary is built only from the caller's own session/connection
(``api.intent_service.build_source_summary``); a foreign id is ignored.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from api.intent_models import RouteIntentRequest, RouteIntentResponse
from api.intent_service import route_intent
from api.rbac_service import UserRole, require_role
from api.utils.caller_scope import CallerScope, require_principal
from api.utils.spend_guard import check_platform_capacity

router = APIRouter(tags=["intent"])

_state: dict[str, Any] = {}


def wire_intent_routes(
    sessions_table: Any, db_connections_table: Any, schema_cache_table: Any, async_session_factory: Any
) -> None:
    _state.update(
        sessions_table=sessions_table,
        db_connections_table=db_connections_table,
        schema_cache_table=schema_cache_table,
        async_session_factory=async_session_factory,
    )


@router.post(
    "/api/route-intent",
    dependencies=[Depends(require_role(UserRole.viewer)), Depends(check_platform_capacity)],
    response_model=RouteIntentResponse,
)
async def post_route_intent(
    body: RouteIntentRequest, scope: CallerScope = Depends(require_principal)
) -> RouteIntentResponse:
    return await route_intent(body, scope, **_state)
