"""SAML 2.0 SSO authentication service — DOCBOT-601.

Supports SP-initiated SSO with Okta and Azure AD (and any SAML 2.0 IdP).

Flow:
    1. User hits GET /api/auth/saml/login  → redirected to IdP login page
    2. IdP POSTs SAML assertion to POST /api/auth/saml/acs
    3. ACS validates assertion, JIT-provisions user, creates session token
    4. Browser receives HttpOnly session cookie, redirected to frontend

Public API:
    build_saml_auth(request_data)  → OneLogin_Saml2_Auth instance
    get_sp_metadata()              → XML string
    process_acs(request_data)      → (user_id, email, name, groups)
    create_session(user_id)        → session_token (str)
    get_user_from_session(token)   → user row or None
    delete_session(token)          → None
"""

from __future__ import annotations

import os
import uuid
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Session TTL
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "8"))


# ---------------------------------------------------------------------------
# SAML settings builder
# ---------------------------------------------------------------------------

def _saml_settings() -> dict:
    """Build python3-saml settings dict from environment variables.

    Required env vars:
        SAML_SP_ENTITY_ID       — e.g. https://docbot.example.com
        SAML_SP_ACS_URL         — e.g. https://docbot.example.com/api/auth/saml/acs
        SAML_IDP_ENTITY_ID      — from IdP metadata
        SAML_IDP_SSO_URL        — IdP SSO redirect URL
        SAML_IDP_X509_CERT      — IdP public certificate (base64, no headers)

    Optional:
        SAML_SP_X509_CERT       — SP certificate for signed requests
        SAML_SP_PRIVATE_KEY     — SP private key for signed requests
        SAML_WANT_NAME_ID_ENCRYPTED   — "true" / "false" (default false)
        SAML_WANT_ASSERTIONS_SIGNED   — "true" / "false" (default true)
    """
    sp_entity_id = os.getenv("SAML_SP_ENTITY_ID", "")
    sp_acs_url = os.getenv("SAML_SP_ACS_URL", "")
    idp_entity_id = os.getenv("SAML_IDP_ENTITY_ID", "")
    idp_sso_url = os.getenv("SAML_IDP_SSO_URL", "")
    idp_cert = os.getenv("SAML_IDP_X509_CERT", "")
    sp_cert = os.getenv("SAML_SP_X509_CERT", "")
    sp_key = os.getenv("SAML_SP_PRIVATE_KEY", "")

    settings: dict[str, Any] = {
        "strict": True,
        "debug": os.getenv("SAML_DEBUG", "false").lower() == "true",
        "sp": {
            "entityId": sp_entity_id,
            "assertionConsumerService": {
                "url": sp_acs_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
            },
            "singleLogoutService": {
                "url": sp_acs_url.replace("/acs", "/slo"),
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
            "x509cert": sp_cert,
            "privateKey": sp_key,
        },
        "idp": {
            "entityId": idp_entity_id,
            "singleSignOnService": {
                "url": idp_sso_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": idp_cert,
        },
        "security": {
            "nameIdEncrypted": os.getenv("SAML_WANT_NAME_ID_ENCRYPTED", "false").lower() == "true",
            "authnRequestsSigned": bool(sp_cert and sp_key),
            "logoutRequestSigned": False,
            "logoutResponseSigned": False,
            "signMetadata": bool(sp_cert and sp_key),
            "wantMessagesSigned": False,
            "wantAssertionsSigned": os.getenv("SAML_WANT_ASSERTIONS_SIGNED", "true").lower() == "true",
            "wantNameId": True,
            "wantNameIdEncrypted": False,
            "wantAssertionsEncrypted": False,
            "allowSingleLabelDomains": False,
            "rejectUnsolicitedResponsesWithInResponseTo": False,
        },
    }
    return settings


def is_saml_configured() -> bool:
    """Return True if minimum SAML env vars are set."""
    required = ["SAML_SP_ENTITY_ID", "SAML_SP_ACS_URL",
                "SAML_IDP_ENTITY_ID", "SAML_IDP_SSO_URL", "SAML_IDP_X509_CERT"]
    return all(os.getenv(v) for v in required)


def is_auth_configured() -> bool:
    """Return True when ANY auth method is available (email/password always qualifies)."""
    return True  # email/password is always available; OAuth providers are optional


def build_saml_auth(request_data: dict) -> Any:
    """Build a OneLogin_Saml2_Auth instance for a given request.

    request_data keys expected by python3-saml:
        https  — bool
        http_host  — str
        script_name  — str (path)
        get_data  — dict (query params)
        post_data  — dict (POST body)
    """
    from onelogin.saml2.auth import OneLogin_Saml2_Auth
    return OneLogin_Saml2_Auth(request_data, _saml_settings())


def get_sp_metadata() -> tuple[str, Optional[str]]:
    """Return (metadata_xml, error_string).  error_string is None on success."""
    from onelogin.saml2.settings import OneLogin_Saml2_Settings
    try:
        saml_settings = OneLogin_Saml2_Settings(_saml_settings(), sp_validation_only=True)
        metadata = saml_settings.get_sp_metadata()
        errors = saml_settings.validate_metadata(metadata)
        if errors:
            return "", ", ".join(errors)
        return metadata.decode("utf-8") if isinstance(metadata, bytes) else metadata, None
    except Exception as exc:
        return "", str(exc)


# ---------------------------------------------------------------------------
# ACS assertion processing
# ---------------------------------------------------------------------------

def process_acs(auth: Any) -> dict:
    """Process a validated SAML assertion and return user attributes.

    Returns dict with: email, name, first_name, last_name, groups, name_id
    Raises ValueError if authentication failed.
    """
    auth.process_response()
    errors = auth.get_errors()
    if errors:
        reason = auth.get_last_error_reason() or ", ".join(errors)
        raise ValueError(f"SAML authentication failed: {reason}")
    if not auth.is_authenticated():
        raise ValueError("SAML response is not authenticated.")

    attrs = auth.get_attributes()
    name_id = auth.get_nameid()

    def _attr(keys: list[str]) -> str:
        for k in keys:
            val = attrs.get(k)
            if val:
                return val[0] if isinstance(val, list) else val
        return ""

    email = _attr([
        "email", "mail",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn",
    ]) or name_id

    first_name = _attr([
        "firstName", "givenName",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
    ])
    last_name = _attr([
        "lastName", "sn", "surname",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
    ])
    display_name = _attr([
        "displayName", "cn", "name",
        "http://schemas.microsoft.com/identity/claims/displayname",
    ]) or f"{first_name} {last_name}".strip() or email.split("@")[0]

    groups_raw = attrs.get("groups", attrs.get(
        "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups", []
    ))
    groups = list(groups_raw) if groups_raw else []

    return {
        "email": email.lower().strip(),
        "name": display_name,
        "first_name": first_name,
        "last_name": last_name,
        "groups": groups,
        "name_id": name_id,
        "provider": _detect_provider(),
    }


def _detect_provider() -> str:
    """Guess the IdP provider from the SSO URL."""
    sso_url = os.getenv("SAML_IDP_SSO_URL", "")
    if "okta.com" in sso_url:
        return "okta"
    if "microsoftonline.com" in sso_url or "azure" in sso_url.lower():
        return "azure_ad"
    return "saml"


# ---------------------------------------------------------------------------
# Session management (stateless token stored in DB)
# ---------------------------------------------------------------------------

def generate_session_token() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


async def create_user_session(
    user_id: str,
    user_sessions_table: Any,
    async_session_factory: Any,
) -> str:
    """Persist a new session token for user_id. Returns the token."""
    from sqlalchemy import insert as sa_insert
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                sa_insert(user_sessions_table).values(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    token=token,
                    expires_at=expires_at,
                )
            )
    return token


async def get_user_from_session(
    token: str,
    user_sessions_table: Any,
    users_table: Any,
    async_session_factory: Any,
) -> Optional[Any]:
    """Return the user row for a valid, non-expired session token, or None."""
    from sqlalchemy import select as sa_select
    async with async_session_factory() as session:
        result = await session.execute(
            sa_select(user_sessions_table).where(
                user_sessions_table.c.token == token,
                user_sessions_table.c.expires_at > datetime.now(timezone.utc),
            )
        )
        sess_row = result.fetchone()
        if not sess_row:
            return None
        user_result = await session.execute(
            sa_select(users_table).where(users_table.c.id == sess_row.user_id)
        )
        return user_result.fetchone()


async def delete_session(
    token: str,
    user_sessions_table: Any,
    async_session_factory: Any,
) -> None:
    from sqlalchemy import delete as sa_delete
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                sa_delete(user_sessions_table).where(
                    user_sessions_table.c.token == token
                )
            )


# ---------------------------------------------------------------------------
# JIT user provisioning
# ---------------------------------------------------------------------------

async def jit_provision_user(
    attrs: dict,
    users_table: Any,
    async_session_factory: Any,
) -> str:
    """Create user on first login if not exists. Returns user_id.

    attrs keys: email, name, provider, provider_id (optional), password_hash (optional)
    Providers: saml | okta | azure_ad | github | google | email
    """
    from sqlalchemy import select as sa_select, insert as sa_insert, update as sa_update
    email = attrs["email"]

    async with async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                sa_select(users_table).where(users_table.c.email == email)
            )
            existing = result.fetchone()

            if existing:
                # Update name/provider on re-login in case they changed
                update_vals: dict[str, Any] = {
                    "name": attrs["name"],
                    "provider": attrs["provider"],
                    "last_login_at": datetime.now(timezone.utc),
                }
                if attrs.get("provider_id"):
                    update_vals["provider_id"] = attrs["provider_id"]
                await session.execute(
                    sa_update(users_table)
                    .where(users_table.c.email == email)
                    .values(**update_vals)
                )
                logger.info("Auth login: returning user %s (%s)", email, existing.id)
                return existing.id

            # First login — provision
            user_id = str(uuid.uuid4())
            insert_vals: dict[str, Any] = {
                "id": user_id,
                "email": email,
                "name": attrs["name"],
                "provider": attrs["provider"],
                "role": "analyst",  # default role
                "last_login_at": datetime.now(timezone.utc),
            }
            if attrs.get("provider_id"):
                insert_vals["provider_id"] = attrs["provider_id"]
            if attrs.get("password_hash"):
                insert_vals["password_hash"] = attrs["password_hash"]

            await session.execute(sa_insert(users_table).values(**insert_vals))
            logger.info("Auth JIT provision: new user %s (%s) via %s", email, user_id, attrs["provider"])
            return user_id
