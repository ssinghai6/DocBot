"""Consumer OAuth 2.0 + email/password authentication — DOCBOT-701.

Supported providers:
    github   — GitHub OAuth App (free)
    google   — Google OAuth 2.0 (free)
    email    — Email + bcrypt password (no external dependency)

Guest mode: no account needed; existing session_id-based flow is unchanged.

Environment variables required per provider:

    GitHub:
        GITHUB_CLIENT_ID
        GITHUB_CLIENT_SECRET

    Google:
        GOOGLE_CLIENT_ID
        GOOGLE_CLIENT_SECRET

    Email/password: always available (no env vars needed)

    Shared:
        APP_BASE_URL   — e.g. https://docbot-backend.railway.app
                         Used to build OAuth callback URLs.
        FRONTEND_URL   — e.g. https://doc-bot-nine.vercel.app
                         Where the user lands after successful login.
"""

from __future__ import annotations

import functools
import os
import logging
import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _app_base_url() -> str:
    return os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")

def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")[0]).rstrip("/")

def is_github_configured() -> bool:
    return bool(os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET"))

def is_google_configured() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))

def get_available_auth_methods() -> list[str]:
    """Return list of enabled auth methods: github, google, email, saml."""
    methods: list[str] = ["email"]  # always available
    if is_github_configured():
        methods.append("github")
    if is_google_configured():
        methods.append("google")
    # SAML check delegated to auth_service to avoid circular imports
    return methods


# ---------------------------------------------------------------------------
# GitHub OAuth
# ---------------------------------------------------------------------------

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def github_authorize_url(state: str) -> str:
    """Build the GitHub OAuth authorization URL to redirect the user to."""
    params = {
        "client_id": os.getenv("GITHUB_CLIENT_ID", ""),
        "redirect_uri": f"{_app_base_url()}/api/auth/github/callback",
        "scope": "read:user user:email",
        "state": state,
    }
    return f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}"


async def github_exchange_code(code: str) -> dict:
    """Exchange OAuth code for user info dict: {email, name, provider_id, avatar_url}.
    Raises ValueError on failure.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": os.getenv("GITHUB_CLIENT_ID"),
                "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
                "code": code,
                "redirect_uri": f"{_app_base_url()}/api/auth/github/callback",
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(f"GitHub token exchange failed: {token_data.get('error_description', token_data)}")

        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

        user_resp = await client.get(GITHUB_USER_URL, headers=headers)
        user_resp.raise_for_status()
        user = user_resp.json()

        # DOCBOT-1617: never trust the public profile `email` field — always
        # take the primary address GitHub itself marks verified.
        emails_resp = await client.get(GITHUB_EMAILS_URL, headers=headers)
        emails_resp.raise_for_status()
        email = select_github_verified_email(emails_resp.json())

        if not email:
            raise ValueError("GitHub account has no verified primary email. Please verify one in your GitHub settings.")

        name = user.get("name") or user.get("login") or email.split("@")[0]
        return {
            "email": email.lower().strip(),
            "name": name,
            "provider": "github",
            "provider_id": str(user["id"]),
            "email_verified": True,
            "avatar_url": user.get("avatar_url"),
        }


def select_github_verified_email(emails: object) -> Optional[str]:
    """Return the primary+verified address from GitHub's /user/emails, or None."""
    if not isinstance(emails, list):
        return None
    for entry in emails:
        if (
            isinstance(entry, dict)
            and entry.get("primary") is True
            and entry.get("verified") is True
            and isinstance(entry.get("email"), str)
            and entry["email"].strip()
        ):
            return entry["email"]
    return None


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def google_authorize_url(state: str) -> str:
    """Build the Google OAuth authorization URL to redirect the user to."""
    params = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "redirect_uri": f"{_app_base_url()}/api/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "state": state,
    }
    return f"{GOOGLE_AUTHORIZE_URL}?{urlencode(params)}"


async def google_exchange_code(code: str) -> dict:
    """Exchange OAuth code for user info dict: {email, name, provider_id}.
    Raises ValueError on failure.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{_app_base_url()}/api/auth/google/callback",
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(f"Google token exchange failed: {token_data.get('error_description', token_data)}")

        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo_resp.raise_for_status()
        info = userinfo_resp.json()

        email = info.get("email")
        if not email:
            raise ValueError("Google account returned no email.")

        name = info.get("name") or info.get("given_name") or email.split("@")[0]
        return {
            "email": email.lower().strip(),
            "name": name,
            "provider": "google",
            "provider_id": info.get("sub", ""),
            # DOCBOT-1617: identity linking requires Google's own assertion.
            "email_verified": is_truthy_claim(info.get("email_verified")),
            "avatar_url": info.get("picture"),
        }


def is_truthy_claim(value: object) -> bool:
    """Google returns email_verified as a bool (userinfo) or "true" (id_token)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


# ---------------------------------------------------------------------------
# Email + password
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    import bcrypt
    # bcrypt hard limit is 72 bytes; encode and truncate before hashing
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    import bcrypt
    pw_bytes = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))


@functools.lru_cache(maxsize=1)
def dummy_password_hash() -> str:
    """bcrypt hash checked when the account is missing, so login timing doesn't reveal it."""
    return hash_password(secrets.token_hex(16))


def validate_password_strength(password: str) -> Optional[str]:
    """Return an error string if weak, None if acceptable."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None


HIBP_RANGE_URL = "https://api.pwnedpasswords.com/range/"
BREACHED_PASSWORD_MESSAGE = (
    "This password has appeared in a known data breach. Please choose a different password."
)


async def is_password_breached(password: str) -> bool:
    """HIBP k-anonymity check — DOCBOT-1622. Only the first 5 SHA-1 hex chars leave the server.

    Fails OPEN (returns False) on any network/HTTP error so an HIBP outage
    never blocks sign-up.
    """
    import hashlib
    digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(HIBP_RANGE_URL + prefix, headers={"Add-Padding": "true"})
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("HIBP range check unavailable (%s) — allowing password", type(exc).__name__)
        return False
    for line in resp.text.splitlines():
        hash_suffix, _, count = line.partition(":")
        if hash_suffix.strip().upper() == suffix:
            # Padding rows carry count 0 and must not count as a breach.
            return count.strip().isdigit() and int(count) > 0
    return False


# ---------------------------------------------------------------------------
# OAuth state — simple in-memory store (short-lived, single-server)
# ---------------------------------------------------------------------------
# CSRF state tokens are short-lived (5 min) and used once.
# For multi-instance deployments, swap for Redis or signed JWT.

import secrets
import time

_pending_states: dict[str, float] = {}  # state → expiry timestamp

_STATE_TTL = 300  # 5 minutes


def generate_oauth_state() -> str:
    state = secrets.token_urlsafe(32)
    # Prune expired entries
    now = time.time()
    expired = [k for k, v in _pending_states.items() if v < now]
    for k in expired:
        del _pending_states[k]
    _pending_states[state] = now + _STATE_TTL
    return state


def validate_oauth_state(state: str) -> bool:
    expiry = _pending_states.pop(state, None)
    if expiry is None:
        return False
    return time.time() < expiry


def oauth_success_redirect(token: str) -> str:
    """URL to redirect to after successful OAuth login."""
    return f"{_frontend_url()}/chat?auth_success=1"
