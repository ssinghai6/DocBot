"""Email verification routes — DOCBOT-1713 (F-4: new routes live here).

``POST /api/auth/verify`` consumes the single-use token from the emailed
link and flips ``users.email_verified`` — deliberately the same column
DOCBOT-1617 reads before linking an OAuth/SAML identity, so the two tickets
share one security state.

The route is intentionally anonymous (the single-use token IS the
credential; listed in tests/unit/test_route_guards.py PUBLIC_ROUTES). All
ports live in ``api.email_verification_service``; handlers stay thin.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from api.email_verify_models import VerifyRequest, VerifyResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Wired at startup by api/index.py (this module has no access to the shared
# metadata before include time). Values are Table objects / session factory.
_state: dict[str, Any] = {}


def wire_email_verification(
    users_table: Any,
    tokens_table: Any,
    async_session_factory: Any,
) -> None:
    """Bind shared tables + the async session factory at include time."""
    _state["users_table"] = users_table
    _state["tokens_table"] = tokens_table
    _state["async_session_factory"] = async_session_factory


def _state_or_fail() -> dict[str, Any]:
    if not all(
        _state.get(key) for key in ("users_table", "tokens_table", "async_session_factory")
    ):
        raise RuntimeError("email_verify_routes: wire_email_verification() was not called")
    return _state


@router.post("/verify", response_model=VerifyResponse)
async def verify_email(
    body: Optional[VerifyRequest] = None,
    token: Optional[str] = Query(None, max_length=512),
) -> VerifyResponse:
    """Consume a single-use verification link token for DOCBOT-1713.

    The token is the credential: deliberately anonymous (PUBLIC_ROUTES in
    tests/unit/test_route_guards.py). Body or ``?token=`` query both work so
    the emailed link can pass the raw value however the frontend renders it.
    """
    raw = (body.token if body is not None else "") or (token or "")
    if not raw.strip():
        raise HTTPException(status_code=422, detail="Missing verification token.")

    state = _state_or_fail()
    from api import email_verification_service as evs

    try:
        email = await evs.consume_verification_token(
            raw, state["tokens_table"], state["users_table"], state["async_session_factory"]
        )
    except evs.VerificationTokenInvalid:
        # Generic on purpose: unknown/expired/reused stay indistinguishable.
        raise HTTPException(status_code=400, detail=evs.GENERIC_TOKEN_ERROR) from None
    return VerifyResponse(status="verified", email=email)
