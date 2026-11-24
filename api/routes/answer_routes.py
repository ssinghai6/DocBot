"""Dropped-answer recovery — DOCBOT-2636.

``GET /api/hybrid/answer/{run_id}`` — viewer+, and 404 unless the run's
session/connection is the caller's, exactly like ``GET /api/lineage/{run_id}``.
``status="pending"`` while the turn is still running (client polls).
The client passes ``session_id`` / ``connection_id`` as query params so an
anonymous demo caller passes ``require_principal`` in required-auth mode;
ownership is still checked against the run itself.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.answer_models import RecoveredAnswer
from api.answer_recovery_service import recover_answer
from api.rbac_service import UserRole, require_role
from api.utils.caller_scope import CallerScope, require_principal

router = APIRouter(tags=["hybrid"])

_state: dict[str, Any] = {}


def wire_answer_routes(
    agent_traces_table: Any, sessions_table: Any, db_connections_table: Any, async_session_factory: Any
) -> None:
    _state.update(
        traces=agent_traces_table,
        sessions=sessions_table,
        connections=db_connections_table,
        factory=async_session_factory,
    )


@router.get(
    "/api/hybrid/answer/{run_id}",
    dependencies=[Depends(require_role(UserRole.viewer))],
    response_model=RecoveredAnswer,
)
async def get_recovered_answer(run_id: str, scope: CallerScope = Depends(require_principal)) -> RecoveredAnswer:
    answer = await recover_answer(
        run_id, scope, _state["traces"], _state["sessions"], _state["connections"], _state["factory"]
    )
    if answer is None:
        raise HTTPException(status_code=404, detail="Answer not found.")
    return answer
