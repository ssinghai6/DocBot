"""LangSmith env pin — DOCBOT-1509.

LangChain/LangGraph automatic tracing ships full node inputs and outputs
(prompts, responses, document text, DB rows). DocBot's metadata-only rule
forbids that under any configuration, so this module forces every SDK
automatic-tracing flag off.

Which variables are pinned
--------------------------
langsmith reads ``TRACING_V2`` and ``TRACING`` through ``get_env_var``, which
checks the ``LANGSMITH_`` namespace before ``LANGCHAIN_``. So a truthy
``LANGSMITH_TRACING_V2`` wins even when ``LANGCHAIN_TRACING_V2`` is false.
Pinned to "false":

* ``LANGCHAIN_TRACING_V2`` and ``LANGSMITH_TRACING_V2`` (v2 auto-tracing)
* ``LANGSMITH_TRACING`` (also DocBot's own switch; see ``docbot_env``)

Popped (v1 flags, removed rather than set): ``LANGCHAIN_TRACING`` and
``LANGCHAIN_HANDLER``. langchain_core checks these by presence. A v1 flag
makes LLM calls raise RuntimeError, so removing it is safer than pinning it.

Fail-closed check
-----------------
After pinning, ``langsmith.utils.tracing_is_enabled()`` must be False. If it
is not, a WARNING is logged and DocBot's own sending path is forced off
through ``sending_forced_off()``. Nothing is raised into request handling.

Ordering matters in two ways:

1. The pin must run before anything reads the flags. langsmith caches
   ``get_env_var`` with ``functools.lru_cache``, so an early read sticks.
   ``pin_langsmith_env()`` therefore clears that cache after writing the env.
   ``api/index.py`` calls it before any other DocBot import, and
   ``langsmith_tracing`` calls it again at import as a backstop.
2. DocBot's own tracing switch (``LANGSMITH_TRACING``) is read from the user's
   values. The pin overwrites that variable, so the user's values are captured
   first and exposed through ``docbot_env()``.

This module imports only stdlib at load time. langsmith is imported inside the
fail-closed check, after the env is pinned.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_DOCBOT_KEYS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY")

# v2 automatic-tracing flags. Each one is pinned to "false".
_AUTO_TRACE_FLAGS = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING_V2",
    "LANGSMITH_TRACING",
)
# v1 flags. Removed from the environment, never set.
_V1_VARS = ("LANGCHAIN_TRACING", "LANGCHAIN_HANDLER")
# Flags whose truthy value is an override worth a warning. LANGSMITH_TRACING
# is DocBot's own switch, so it is not named here.
_WARN_ON_TRUTHY = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_HANDLER")

# Snapshot of DocBot's own LangSmith settings, taken before the pin overwrites
# LANGSMITH_TRACING. None until pin_langsmith_env() has run.
_user_values: Optional[dict[str, Optional[str]]] = None
# Set when the fail-closed check finds langsmith still enabled after the pin.
_force_disabled: bool = False


def _is_truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUTHY


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


def _verify_pinned() -> None:
    """Fail closed: if langsmith still reports tracing on, force DocBot off."""
    global _force_disabled
    _clear_langsmith_env_cache()
    try:
        from langsmith.utils import tracing_is_enabled
    except ImportError:
        return  # langsmith not installed, so nothing can auto-trace
    try:
        still_enabled = bool(tracing_is_enabled())
    except Exception as exc:  # the check itself must never break startup
        logger.warning(
            "LangSmith tracing check failed (%s); forcing DocBot LangSmith sending off",
            type(exc).__name__,
        )
        _force_disabled = True
        return
    if still_enabled:
        logger.warning(
            "LangSmith auto-tracing is still enabled after the env pin; "
            "forcing DocBot LangSmith sending off"
        )
        _force_disabled = True
    _clear_langsmith_env_cache()


def pin_langsmith_env() -> None:
    """Force LangChain/LangSmith automatic tracing off. Idempotent.

    Captures DocBot's own settings first. Warns when a truthy override of an
    auto-trace variable is found, because that value would otherwise ship
    prompts and responses, which violates the metadata-only rule.
    """
    global _user_values
    if _user_values is not None:
        return
    _user_values = {key: os.environ.get(key) for key in _DOCBOT_KEYS}

    for name in _WARN_ON_TRUTHY:
        if _is_truthy(os.environ.get(name)):
            logger.warning(
                "%s is enabled but DocBot forces it off: LangChain auto-tracing "
                "would send prompts and responses to LangSmith.",
                name,
            )
    for name in _AUTO_TRACE_FLAGS:
        os.environ[name] = "false"
    for name in _V1_VARS:
        os.environ.pop(name, None)
    _verify_pinned()


def sending_forced_off() -> bool:
    """True when the fail-closed check disabled DocBot's own LangSmith sending."""
    return _force_disabled


def docbot_env() -> Mapping[str, Optional[str]]:
    """DocBot's view of the LangSmith settings the user configured.

    Returns the pre-pin snapshot once pinned. Before that, falls back to the
    live environment.
    """
    if _user_values is not None:
        return _user_values
    return os.environ
