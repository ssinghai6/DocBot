"""LangSmith env pin — DOCBOT-1509.

LangChain/LangGraph automatic tracing ships full node inputs and outputs
(prompts, responses, document text, DB rows). DocBot's metadata-only rule
forbids that under any configuration, so this module forces the SDK's
automatic-tracing flags off.

Ordering matters in two ways:

1. The pin must run before anything reads the flags. langsmith caches
   ``get_env_var`` with ``functools.lru_cache``, so an early read sticks.
   ``pin_langsmith_env()`` therefore clears that cache after writing the env.
   ``api/index.py`` calls it before any other DocBot import, and
   ``langsmith_tracing`` calls it again at import as a backstop.
2. DocBot's own tracing switch (``LANGSMITH_TRACING``) is read from the user's
   values. The pin overwrites that variable, so the user's values are captured
   first and exposed through ``docbot_env()``.

This module imports only stdlib, so it is safe to import first thing.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_DOCBOT_KEYS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY")

# Snapshot of DocBot's own LangSmith settings, taken before the pin overwrites
# LANGSMITH_TRACING. None until pin_langsmith_env() has run.
_user_values: Optional[dict[str, Optional[str]]] = None


def _clear_langsmith_env_cache() -> None:
    """Drop langsmith's cached env lookups so the pinned values take effect.

    Only touches langsmith when it is already imported. If it is not, nothing
    can have cached a stale value yet.
    """
    utils = sys.modules.get("langsmith.utils")
    cached = getattr(utils, "get_env_var", None)
    clear = getattr(cached, "cache_clear", None)
    if callable(clear):
        clear()


def pin_langsmith_env() -> None:
    """Force LangChain/LangSmith automatic tracing off. Idempotent.

    Captures DocBot's own settings first. Overrides a truthy
    ``LANGCHAIN_TRACING_V2`` and logs a warning, because a user-set value would
    otherwise ship content, which violates the metadata-only rule. Falsy or
    unset values are left as they are, apart from being pinned to "false".
    """
    global _user_values
    if _user_values is not None:
        return
    _user_values = {key: os.environ.get(key) for key in _DOCBOT_KEYS}

    if (os.environ.get("LANGCHAIN_TRACING_V2") or "").strip().lower() in _TRUTHY:
        logger.warning(
            "LANGCHAIN_TRACING_V2 is enabled but DocBot forces it off: "
            "LangChain auto-tracing would send prompts and responses to LangSmith."
        )
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    _clear_langsmith_env_cache()


def docbot_env() -> Mapping[str, Optional[str]]:
    """DocBot's view of the LangSmith settings the user configured.

    Returns the pre-pin snapshot once pinned. Before that, falls back to the
    live environment.
    """
    if _user_values is not None:
        return _user_values
    return os.environ
