"""Sentry init with secret/PII scrubbing (DOCBOT-1401, DOCBOT-1618).

Error tracking only (traces_sample_rate=0.0). Every event and breadcrumb is
passed through ``scrub`` so API keys, passwords and DB credentials never
leave the process, even when they end up in an exception message.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

FILTERED = "[Filtered]"

_SENSITIVE_WORDS = r"password|passwd|secret|token|api[_-]?key|authorization|cookie|owner[_-]?key"

# Dict keys whose whole value is dropped (headers, extra, cookies, form fields).
_SENSITIVE_KEY = re.compile(rf"(?i)(?:{_SENSITIVE_WORDS})")

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # DB/URL credentials: scheme://user:pass@host -> scheme://user:***@host
    (re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]+@"), r"\1***@"),
    # key=value / "key": "value" / Authorization: Bearer xyz inside free text
    (
        re.compile(
            rf"(?i)([\w-]*(?:{_SENSITIVE_WORDS})[\w-]*)"
            r"""(["']?\s*[:=]\s*["']?)(?:(?:Bearer|Basic|Token)\s+)?[^\s"',;&}]+"""
        ),
        rf"\1\2{FILTERED}",
    ),
    (re.compile(r"\bgsk_[A-Za-z0-9]{16,}"), FILTERED),  # Groq
    (re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{8,}"), FILTERED),  # Stripe secret/restricted
    (re.compile(r"\bwhsec_[A-Za-z0-9+/=]{8,}"), FILTERED),  # Stripe webhook secret
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), FILTERED),  # OpenAI-style
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), FILTERED),  # Google
    (re.compile(r"\be2b_[A-Za-z0-9]{16,}"), FILTERED),  # E2B
    (re.compile(r"\bre_(?=[A-Za-z0-9_]*\d)[A-Za-z0-9_]{16,}"), FILTERED),  # Resend
    (re.compile(r"\bgAAAAA[A-Za-z0-9_\-]{20,}={0,2}"), FILTERED),  # Fernet token
    (re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]{43}="), FILTERED),  # Fernet key (32B urlsafe b64)
]


def scrub_text(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def scrub(value: Any) -> Any:
    """Recursively redact secrets in dicts/lists/strings. Returns a new object."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE_KEY.search(k) and v not in (None, ""):
                out[k] = FILTERED
            else:
                out[k] = scrub(v)
        return out
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def before_send(event: dict, hint: Optional[dict]) -> dict:
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("cookies", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            request["headers"] = {
                k: v for k, v in headers.items()
                if k.lower() not in ("authorization", "cookie", "x-owner-key")
            }
    return scrub(event)


def before_breadcrumb(crumb: dict, hint: Optional[dict]) -> dict:
    return scrub(crumb)


def init_sentry() -> bool:
    """Initialise Sentry if SENTRY_DSN is set. Returns True when enabled."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return False
    import sentry_sdk

    environment = os.getenv("RAILWAY_ENVIRONMENT", "local")
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.0,
        environment=environment,
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        before_send=before_send,
        before_breadcrumb=before_breadcrumb,
    )
    logger.info("Sentry error tracking enabled (environment=%s)", environment)
    return True
