"""Tie-out workpaper download — DOCBOT-2014.

``GET /api/lineage/{run_id}/tieout.{xlsx|pdf}`` — viewer+, and (DOCBOT-1620)
404 unless the run's session/connection is the caller's, exactly like
``GET /api/lineage/{run_id}``. Rendering lives in ``api/tieout_export.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api import tieout_export
from api.lineage_service import get_lineage
from api.rbac_service import UserRole, require_role
from api.resource_scope_service import run_row_accessible
from api.utils.caller_scope import CallerScope, require_principal

router = APIRouter(tags=["lineage"])

_state: dict[str, Any] = {}


def wire_tieout_routes(
    answer_lineage_table: Any, sessions_table: Any, db_connections_table: Any, async_session_factory: Any
) -> None:
    _state.update(
        lineage=answer_lineage_table,
        sessions=sessions_table,
        connections=db_connections_table,
        factory=async_session_factory,
    )


@router.get(
    "/api/lineage/{run_id}/tieout.{fmt}",
    dependencies=[Depends(require_role(UserRole.viewer))],
    response_class=Response,
)
async def download_tieout(
    run_id: str, fmt: Literal["xlsx", "pdf"], scope: CallerScope = Depends(require_principal)
) -> Response:
    if not await run_row_accessible(
        _state["lineage"], run_id, scope, _state["sessions"], _state["connections"], _state["factory"]
    ):
        raise HTTPException(status_code=404, detail="Lineage not found.")
    lineage = await get_lineage(run_id)
    if lineage is None or not lineage.discrepancies:
        raise HTTPException(status_code=404, detail="No tie-out lines for this answer.")
    rendered = tieout_export.render(lineage, fmt)
    if rendered is None:
        raise HTTPException(status_code=404, detail="Unknown format.")
    body, media_type, name = rendered
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"},
    )
