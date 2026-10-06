"""LangSmith tracing — DOCBOT-1509.

METADATA ONLY. Nothing sent to LangSmith contains prompt text, response text,
document content, DB rows, or connection strings. Every LLM call becomes a
LangSmith ``llm`` run whose ``inputs`` and ``outputs`` are empty (token usage
only), and whose ``extra.metadata`` is built from an explicit allowlist
(``_LLM_METADATA_KEYS``). Error details are the exception *class name* only.

Config (both env vars are read once, at import time):
  LANGSMITH_TRACING   bool — master switch
  LANGSMITH_API_KEY   str  — tracing is fully OFF unless this is also set

When tracing is off, every public function here returns immediately: no
client is constructed and no network call is made.

Design notes
------------
* The DOCBOT run_id (DOCBOT-1501 ContextVar, see llm_provider.run_trace) is
  reused as the LangSmith trace id. A ``run_trace(run_id, name=...)`` scope
  opens one root run whose LangSmith id is ``UUID(run_id)``. Each LLM call
  inside that scope is created as a child of the current run tree, so there
  is a single run_id scheme, not two.
* LangChain/LangGraph *automatic* tracing is deliberately disabled
  (LANGSMITH_TRACING and LANGCHAIN_TRACING_V2 are pinned to "false" in the
  process env after config is read; the investigation root is entered with
  ``tracing_context(enabled=False)``). Automatic tracing ships full node
  inputs/outputs, i.e. prompt and response text, which violates the
  metadata-only rule. Grouping is done explicitly by this module instead.
* Sending happens on a small background thread pool. A LangSmith outage,
  timeout or 4xx is logged at WARNING with the exception class name only and
  never reaches the caller.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Keys copied from the llm_call payload into LangSmith metadata. This
# allowlist is the security boundary: anything not listed here is never sent.
_LLM_METADATA_KEYS = (
    "run_id",
    "provider",
    "model",
    "caller",
    "latency_ms",
    "estimated_cost_usd",
    "success",
    "fallback_triggered",
)

_MAX_PENDING = 500  # bound the background queue; drop (not block) past this


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool
    api_key: Optional[str]


def load_config(env: Any = None) -> TracingConfig:
    """Resolve tracing config from an env mapping. Enabled only if both the
    LANGSMITH_TRACING flag is truthy AND LANGSMITH_API_KEY is non-empty."""
    source = os.environ if env is None else env
    key = (source.get("LANGSMITH_API_KEY") or "").strip()
    flag = (source.get("LANGSMITH_TRACING") or "").strip().lower() in _TRUTHY
    return TracingConfig(enabled=bool(flag and key), api_key=key or None)


_config: TracingConfig = load_config()

# Stop LangChain/LangSmith automatic tracing from shipping content. Our own
# explicit tracing above does not depend on these flags. Must run after
# load_config() reads the user's values. langsmith caches env lookups, so this
# only takes effect if nothing read these vars earlier in the process.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

_client: Any = None
_client_lock = threading.Lock()

_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_pending: "set[concurrent.futures.Future]" = set()
_pending_lock = threading.Lock()


def is_enabled() -> bool:
    return _config.enabled


def _get_client() -> Any:
    """Lazily build the LangSmith client. Only called when tracing is enabled."""
    global _client
    with _client_lock:
        if _client is None:
            from langsmith import Client

            _client = Client(api_key=_config.api_key)
        return _client


def _safe_run(fn: Any, *args: Any) -> None:
    """Background wrapper: any failure is reduced to a log line."""
    try:
        fn(*args)
    except Exception as exc:  # tracing must never surface to a user request
        logger.warning("langsmith trace send failed (%s)", type(exc).__name__)


def _submit(fn: Any, *args: Any) -> bool:
    global _executor
    with _pending_lock:
        if len(_pending) >= _MAX_PENDING:
            logger.debug("langsmith trace queue full, dropping run")
            return False
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="langsmith-trace"
            )
        future = _executor.submit(_safe_run, fn, *args)
        _pending.add(future)

    def _done(f: "concurrent.futures.Future") -> None:
        with _pending_lock:
            _pending.discard(f)

    future.add_done_callback(_done)
    return True


def flush(timeout: float = 5.0) -> None:
    """Wait for queued sends to finish. Intended for tests and shutdown."""
    with _pending_lock:
        futures = list(_pending)
    if futures:
        concurrent.futures.wait(futures, timeout=timeout)


def build_llm_run_spec(payload: dict) -> dict:
    """Project a DOCBOT llm_call payload onto the metadata-only LangSmith spec.

    Takes only allowlisted keys. ``error_class`` is the exception class name;
    error messages are never read here.
    """
    spec = {
        "run_id": payload.get("run_id"),
        "provider": payload.get("llm_provider"),
        "model": payload.get("llm_model"),
        "caller": payload.get("llm_caller"),
        "latency_ms": payload.get("llm_latency_ms"),
        "estimated_cost_usd": payload.get("llm_estimated_cost_usd"),
        "success": payload.get("llm_success"),
        "fallback_triggered": payload.get("llm_fallback_triggered"),
        "input_tokens": payload.get("llm_input_tokens"),
        "output_tokens": payload.get("llm_output_tokens"),
        "error_class": payload.get("error_class"),
    }
    return spec


def _send_llm_run(spec: dict, parent: Any) -> None:
    from langsmith import run_trees

    latency_ms = float(spec.get("latency_ms") or 0.0)
    end = datetime.now(timezone.utc)
    start = datetime.fromtimestamp(end.timestamp() - latency_ms / 1000.0, tz=timezone.utc)

    metadata = {k: spec[k] for k in _LLM_METADATA_KEYS if spec.get(k) is not None}
    metadata["ls_provider"] = spec.get("provider")
    metadata["ls_model_name"] = spec.get("model")

    in_tok, out_tok = spec.get("input_tokens"), spec.get("output_tokens")
    outputs: dict[str, Any] = {}
    if in_tok is not None and out_tok is not None:
        outputs["usage_metadata"] = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }

    name = f"llm:{spec.get('caller') or 'unknown'}"
    error = spec.get("error_class")
    if parent is not None:
        run = parent.create_child(
            name,
            run_type="llm",
            inputs={},
            outputs=outputs,
            error=error,
            start_time=start,
            end_time=end,
            extra={"metadata": metadata},
        )
    else:
        run = run_trees.RunTree(
            name=name,
            run_type="llm",
            inputs={},
            outputs=outputs,
            error=error,
            start_time=start,
            end_time=end,
            extra={"metadata": metadata},
            client=_get_client(),
        )
    run.post()


def emit_llm_run(payload: dict) -> None:
    """Record one LLM call as a LangSmith run. Non-blocking; never raises."""
    if not _config.enabled:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        spec = build_llm_run_spec(payload)
        _submit(_send_llm_run, spec, get_current_run_tree())
    except Exception as exc:  # tracing must never break the LLM call path
        logger.debug("langsmith emit skipped (%s)", type(exc).__name__)


def _post_root(root: Any) -> None:
    root.end(outputs={})
    root.post()


@contextmanager
def investigation_scope(run_id: str, name: str) -> Iterator[Optional[Any]]:
    """Open one LangSmith root run for a multi-step investigation.

    The root's id is ``UUID(run_id)`` so LangSmith and the DOCBOT run_id are the
    same identifier. Nested scopes (e.g. deep_retrieve inside Autopilot) reuse
    the existing parent and open no new root. The root is sent when the scope
    exits. When tracing is off this is a no-op that yields None.
    """
    if not _config.enabled:
        yield None
        return

    from langsmith.run_helpers import get_current_run_tree, tracing_context

    if get_current_run_tree() is not None:
        yield None
        return

    try:
        from langsmith import run_trees

        root = run_trees.RunTree(
            id=uuid.UUID(run_id),
            name=name,
            run_type="chain",
            inputs={},
            extra={"metadata": {"run_id": run_id}},
            client=_get_client(),
        )
    except Exception as exc:
        logger.warning("langsmith root run skipped (%s)", type(exc).__name__)
        yield None
        return

    # enabled=False stops LangChain automatic tracing inside the scope while
    # keeping `root` visible via get_current_run_tree() for child linking.
    with tracing_context(parent=root, enabled=False):
        try:
            yield root
        finally:
            _submit(_post_root, root)
