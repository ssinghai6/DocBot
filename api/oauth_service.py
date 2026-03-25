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

import os
import logging
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

        email = user.get("email")
        if not email:
            emails_resp = await client.get(GITHUB_EMAILS_URL, headers=headers)
            emails_resp.raise_for_status()
            emails = emails_resp.json()
            primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
            email = primary["email"] if primary else None

        if not email:
            raise ValueError("GitHub account has no verified email. Please add one in your GitHub settings.")

        name = user.get("name") or user.get("login") or email.split("@")[0]
        return {
            "email": email.lower().strip(),
            "name": name,
            "provider": "github",
            "provider_id": str(user["id"]),
            "avatar_url": user.get("avatar_url"),
        }


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
            "avatar_url": info.get("picture"),
        }


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


def validate_password_strength(password: str) -> Optional[str]:
    """Return an error string if weak, None if acceptable."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None


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
    return f"{_frontend_url()}/?auth_success=1"
