"""CORS origin parsing + boot guard — DOCBOT-1624 (SEC-22).

``ALLOWED_ORIGINS`` is a comma-separated env var. Browsers send the Origin
header without a trailing slash, so ``"https://app.example.com/ "`` would
never match; entries are stripped of whitespace and trailing slashes and
empty entries dropped. A ``*`` origin combined with credentialed CORS lets
any site make authenticated requests with the user's cookies, so the app
refuses to boot in that configuration.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000"


class UnsafeCorsConfigError(RuntimeError):
    """Raised at startup when the CORS configuration is unsafe."""


def parse_allowed_origins(raw: str | None = None) -> list[str]:
    """Normalize ``ALLOWED_ORIGINS`` (env when ``raw`` is None) into an origin list."""
    if raw is None:
        raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [o for o in (part.strip().rstrip("/") for part in raw.split(",")) if o]


def assert_safe_cors_config(origins: list[str], allow_credentials: bool) -> None:
    """Refuse ``*`` with credentials; log the final origin list once."""
    if allow_credentials and "*" in origins:
        raise UnsafeCorsConfigError(
            "Refusing to start: ALLOWED_ORIGINS contains '*' while CORS "
            "allow_credentials=True. List the exact frontend origins instead."
        )
    logger.info("CORS allowed origins: %s", origins)
