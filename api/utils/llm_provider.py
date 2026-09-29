"""LLM Provider with automatic fallback — Investor Readiness Sprint.

Provides a unified LLM interface with Groq (openai/gpt-oss-20b) as primary
and Gemini 2.5 Flash as fallback. On Groq failure (rate limit, 5xx,
timeout), automatically retries with Gemini.

DOCBOT-1403: GROQ_MODEL was "llama-3.3-70b-versatile", which Groq removed
from its catalog entirely (confirmed via /v1/models — not an access issue,
the model no longer exists). Every call defaulting to GROQ_MODEL (SQL gen,
autopilot, query expansion, hybrid synthesis, intent classification) was
silently 404ing and falling back to Gemini, or failing outright if
GEMINI_API_KEY wasn't set. Migrated to openai/gpt-oss-20b — same model
family as GROQ_CODE_MODEL (openai/gpt-oss-120b, already proven working),
confirmed free-tier, deliberately a distinct model from GROQ_CODE_MODEL so
sandbox_service's two-model retry ladder (`if _model != GROQ_CODE_MODEL`)
still means something.

Usage:
    from api.utils.llm_provider import get_llm, call_llm

    # Get a LangChain LLM instance (tries Groq first, Gemini on failure)
    llm = get_llm()

    # Or call directly with a prompt string
    response = await call_llm("Summarize this document...")

Usage:
    from api.utils.llm_provider import get_llm, call_llm, chat_completion, chat_completion_stream

    # LangChain: get a ChatModel with fallback
    llm = get_llm()

    # Raw SDK style: non-streaming
    text = chat_completion(messages, model="openai/gpt-oss-20b")

    # Raw SDK style: streaming
    for token in chat_completion_stream(messages, model="openai/gpt-oss-20b"):
        print(token, end="")
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run/trace ID propagation — DOCBOT-1501
#
# A ``run_id`` groups every LLM call belonging to one multi-step investigation
# (Autopilot, Deep Research, hybrid chat) so they can be reconstructed after
# the fact from the persisted call log. Propagated via a ContextVar rather
# than threading an explicit parameter through every nested function call:
# asyncio.Task/create_task (including LangGraph's internal node dispatch)
# copies the current context at task-creation time, so setting this once at
# the top of a pipeline entrypoint automatically covers every LLM call made
# anywhere in that call tree — including concurrently-dispatched steps.
# ---------------------------------------------------------------------------

_current_run_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "llm_run_id", default=None
)


def new_run_id() -> str:
    """Generate a fresh run/trace id."""
    return uuid.uuid4().hex


def current_run_id() -> Optional[str]:
    """Return the run_id active in the current context, if any."""
    return _current_run_id.get()


@contextmanager
def run_trace(run_id: Optional[str] = None):
    """Bind ``run_id`` as the active trace for every LLM call made within
    this context (and in any asyncio task spawned from within it).

    Usage at the top of a pipeline entrypoint::

        run_id = current_run_id() or new_run_id()
        with run_trace(run_id):
            ...  # every LLM call in here shares run_id

    Re-entrant/idempotent by convention: nested pipeline functions should
    check ``current_run_id()`` first and reuse it rather than minting a new
    one, so a nested call (e.g. Autopilot's doc_search step calling
    ``deep_retrieve``) stays under the parent investigation's trace.
    """
    token = _current_run_id.set(run_id or new_run_id())
    try:
        yield _current_run_id.get()
    finally:
        _current_run_id.reset(token)


# Optional persistence hook, injected by api/index.py at startup via
# set_trace_sink(). None in unit tests / any context where the DB-backed
# call log isn't wired — _log_llm_call still emits its stdout JSON line.
_trace_sink: Optional[Callable[[dict], None]] = None


def set_trace_sink(sink: Optional[Callable[[dict], None]]) -> None:
    """Register a callback invoked with the llm_call payload after each call.

    The callback must be synchronous and non-blocking (e.g. a queue.put_nowait
    wrapper) — it runs inline on whatever thread/loop made the LLM call.
    """
    global _trace_sink
    _trace_sink = sink

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-2.5-flash"

# Groq error types that trigger fallback
_FALLBACK_STATUS_CODES = {429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Thin LLM telemetry — DOCBOT-1401
#
# One structured log line per call, emitted from this module (the choke point
# for chat_completion/chat_completion_stream/call_llm) rather than at each
# external callsite. Deliberately not Langfuse/LangSmith/OTel — single
# container, no fan-out to trace across; a grep-able JSON log line into
# Railway's log viewer is enough signal for a solo-founder deploy.
# ---------------------------------------------------------------------------

# $ per 1K tokens (input, output). Verified against Groq's published rate
# card 2026-09 (not billing-grade — rates can change without notice).
# Unlisted models log a null estimated_cost_usd rather than guessing.
_COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    GROQ_MODEL: (0.000075, 0.0003),       # openai/gpt-oss-20b: $0.075 / $0.30 per 1M
    "openai/gpt-oss-120b": (0.00015, 0.0006),  # GROQ_CODE_MODEL: $0.15 / $0.60 per 1M
    GEMINI_MODEL: (0.000075, 0.0003),
}


def _estimate_cost_usd(
    model: str, input_tokens: Optional[int], output_tokens: Optional[int]
) -> Optional[float]:
    if input_tokens is None or output_tokens is None:
        return None
    rates = _COST_PER_1K_TOKENS.get(model)
    if not rates:
        return None
    in_rate, out_rate = rates
    return round((input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate, 6)


def _log_llm_call(
    *,
    provider: str,
    model: str,
    latency_ms: float,
    success: bool,
    fallback_triggered: bool,
    caller: Optional[str],
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    run_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Emit one structured log line for an LLM call as a JSON string in the
    message body — not via logging's `extra=` mechanism, which silently
    drops unrecognized fields under the default formatter (`logging.basicConfig()`
    in api/index.py has no custom Formatter, so `extra=` fields never reached
    stdout/Railway logs; only unit tests using `caplog`, which reads LogRecord
    attributes directly and bypasses formatting, could see them). Grep for
    '"event": "llm_call"' in Railway logs, or `jq` for a structured pull.

    DOCBOT-1501: also resolves a run_id (explicit override → active
    ContextVar → freshly minted) and hands the payload to the optional
    persistence sink (see set_trace_sink) so multi-step investigations are
    queryable after the fact, not just grep-able from stdout.
    """
    resolved_run_id = run_id or current_run_id() or new_run_id()
    payload = {
        "event": "llm_call",
        "run_id": resolved_run_id,
        "llm_provider": provider,
        "llm_model": model,
        "llm_latency_ms": round(latency_ms),
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_estimated_cost_usd": _estimate_cost_usd(model, input_tokens, output_tokens),
        "llm_success": success,
        "llm_fallback_triggered": fallback_triggered,
        "llm_caller": caller,
        "error_message": error_message,
    }
    logger.info(json.dumps(payload), extra=payload)

    if _trace_sink is not None:
        try:
            _trace_sink(payload)
        except Exception as exc:  # tracing must never break the LLM call path
            logger.debug("llm_provider: trace sink failed (%s)", exc)


def log_external_llm_call(
    *,
    provider: str,
    model: str,
    latency_ms: float,
    success: bool,
    caller: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    fallback_triggered: bool = False,
    run_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Public logging hook for call sites that build their own LLM client
    instead of going through call_llm/chat_completion/chat_completion_stream
    (e.g. hybrid_service.classify_intent, which takes an injected groq_client
    for test-compat reasons and can't be routed through this module's own
    fallback wrappers without a signature/behavior change).

    Kept for observability parity with the wrapped functions above; adding
    Groq→Gemini fallback at these call sites is a separate, riskier change
    tracked outside this ticket.
    """
    _log_llm_call(
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        success=success,
        fallback_triggered=fallback_triggered,
        caller=caller,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        run_id=run_id,
        error_message=error_message,
    )


def log_malformed_llm_output(*, caller: str, error: str) -> None:
    """DOCBOT-1504: structured, grep-able signal for "the LLM API call
    succeeded but the response didn't parse/validate as the structured
    output we asked for" — a distinct failure mode from the API-level
    success/failure already tracked by _log_llm_call (that call still
    counts as success=True; this is a content-shape failure on top of it).

    Emitted once per malformed-output event (i.e. once per failed parse
    attempt, including the final failure after a retry) so the event rate
    is countable — the initial/only intended consumer is grepping Railway
    logs for '"event": "malformed_llm_output"', with DOCBOT-1501/1502's
    persisted call log providing correlated latency/cost via the matching
    `caller` tag in the same time window. A first-class column in the
    llm_calls table is a natural follow-up once there's a concrete need to
    query malformed-output rate rather than grep it.
    """
    payload = {"event": "malformed_llm_output", "caller": caller, "error": str(error)[:200]}
    logger.warning(json.dumps(payload), extra=payload)


def _get_groq_llm(
    api_key: Optional[str] = None,
    temperature: float = 0,
    streaming: bool = False,
) -> BaseChatModel:
    """Create a Groq ChatModel instance."""
    from langchain_groq import ChatGroq

    key = api_key or os.getenv("groq_api_key", "")
    if not key:
        raise ValueError("Groq API key not available (groq_api_key env var)")

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=key,
        temperature=temperature,
        streaming=streaming,
    )


def _get_gemini_llm(
    api_key: Optional[str] = None,
    temperature: float = 0,
    streaming: bool = False,
) -> BaseChatModel:
    """Create a Gemini ChatModel instance.

    Uses google-generativeai SDK directly wrapped in a LangChain-compatible
    interface, avoiding the broken langchain-google-genai pydantic_v1 dependency.
    """
    from api.utils._gemini_wrapper import GeminiChatModel

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Gemini API key not available (GEMINI_API_KEY env var)")

    return GeminiChatModel(
        model=GEMINI_MODEL,
        api_key=key,
        temperature=temperature,
    )


def get_llm(
    temperature: float = 0,
    streaming: bool = False,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> BaseChatModel:
    """Return a LangChain ChatModel, preferring Groq with Gemini fallback.

    Parameters
    ----------
    temperature : float
        Sampling temperature (default 0 for deterministic).
    streaming : bool
        Whether the model should stream tokens.
    groq_api_key : str, optional
        Override Groq API key (defaults to env var).
    gemini_api_key : str, optional
        Override Gemini API key (defaults to env var).
    provider : str, optional
        Force a specific provider ("groq" or "gemini"). If None, tries
        Groq first and falls back to Gemini on instantiation failure.

    Returns
    -------
    BaseChatModel
        A LangChain-compatible chat model.
    """
    if provider == "gemini":
        logger.info("LLM provider: Gemini (forced)")
        return _get_gemini_llm(gemini_api_key, temperature, streaming)

    if provider == "groq":
        logger.info("LLM provider: Groq (forced)")
        return _get_groq_llm(groq_api_key, temperature, streaming)

    # Default: try Groq first
    try:
        llm = _get_groq_llm(groq_api_key, temperature, streaming)
        logger.info("LLM provider: Groq (primary)")
        return llm
    except (ValueError, ImportError) as exc:
        logger.warning("Groq LLM unavailable (%s), falling back to Gemini", exc)
        return _get_gemini_llm(gemini_api_key, temperature, streaming)


def _is_retriable_error(exc: Exception) -> bool:
    """Check if an exception is a retriable Groq API error."""
    exc_str = str(exc).lower()

    # Check for HTTP status codes in the error message
    for code in _FALLBACK_STATUS_CODES:
        if str(code) in exc_str:
            return True

    # Check for common error patterns
    retriable_patterns = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "server error",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "timeout",
        "timed out",
        "connection error",
    ]
    return any(pattern in exc_str for pattern in retriable_patterns)


def safe_int(value) -> Optional[int]:
    """Coerce to int or None — never leak a non-JSON-serializable object
    (e.g. an unconfigured MagicMock attribute in tests) into a log payload."""
    return value if isinstance(value, int) else None


def _token_usage_from_response(response) -> tuple[Optional[int], Optional[int]]:
    """Best-effort extraction of (input_tokens, output_tokens) from a
    LangChain ChatModel response. Returns (None, None) if unavailable —
    providers/wrappers don't consistently populate response_metadata, and a
    non-dict response_metadata (e.g. an unconfigured test mock) must not leak
    a non-JSON-serializable value into the telemetry payload."""
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None, None
    usage = metadata.get("token_usage") or metadata.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        return None, None
    return safe_int(usage.get("prompt_tokens")), safe_int(usage.get("completion_tokens"))


async def call_llm(
    prompt: str,
    *,
    temperature: float = 0,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    caller: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Call the LLM with automatic fallback from Groq to Gemini.

    This is a convenience function for simple prompt-in/string-out usage.
    For more complex chains, use get_llm() and compose your own pipeline.

    Parameters
    ----------
    prompt : str
        The user prompt to send.
    temperature : float
        Sampling temperature.
    groq_api_key : str, optional
        Override Groq API key.
    gemini_api_key : str, optional
        Override Gemini API key.
    caller : str, optional
        Short tag identifying the calling code path (e.g. "sql_gen",
        "autopilot_planner") — carried into the structured llm_call log line
        for per-path cost/latency breakdown. Purely observational.

    Returns
    -------
    str
        The model's response text.
    """
    from langchain_core.messages import HumanMessage

    # Try Groq first
    try:
        groq_llm = _get_groq_llm(groq_api_key, temperature, streaming=False)
        start = time.monotonic()
        response = await groq_llm.ainvoke([HumanMessage(content=prompt)])
        elapsed = time.monotonic() - start
        logger.info("LLM call completed via Groq in %.2fs", elapsed)
        in_tok, out_tok = _token_usage_from_response(response)
        _log_llm_call(
            provider="groq", model=GROQ_MODEL, latency_ms=elapsed * 1000,
            success=True, fallback_triggered=False, caller=caller,
            input_tokens=in_tok, output_tokens=out_tok, run_id=run_id,
        )
        return response.content
    except ValueError:
        # Groq key not available — go straight to Gemini
        logger.warning("Groq unavailable (no API key), using Gemini")
    except Exception as exc:
        if _is_retriable_error(exc):
            logger.warning(
                "Groq call failed with retriable error (%s: %s), falling back to Gemini",
                type(exc).__name__,
                str(exc)[:200],
            )
        else:
            # Non-retriable error — still try Gemini but log as error
            logger.error(
                "Groq call failed with non-retriable error (%s: %s), attempting Gemini fallback",
                type(exc).__name__,
                str(exc)[:200],
            )

    # Fallback to Gemini
    gemini_llm = _get_gemini_llm(gemini_api_key, temperature, streaming=False)
    start = time.monotonic()
    response = await gemini_llm.ainvoke([HumanMessage(content=prompt)])
    elapsed = time.monotonic() - start
    logger.info("LLM call completed via Gemini (fallback) in %.2fs", elapsed)
    in_tok, out_tok = _token_usage_from_response(response)
    _log_llm_call(
        provider="gemini", model=GEMINI_MODEL, latency_ms=elapsed * 1000,
        success=True, fallback_triggered=True, caller=caller,
        input_tokens=in_tok, output_tokens=out_tok, run_id=run_id,
    )
    return response.content


# ---------------------------------------------------------------------------
# Raw SDK-style completions with Groq → Gemini fallback
# ---------------------------------------------------------------------------

# Groq code-generation model. Migrated qwen/qwen3-32b → openai/gpt-oss-120b
# (Qwen3 32B decommissioned on Groq 2026-07-17). gpt-oss-120b returns reasoning
# in a separate field rather than inline <think> tags, so the <think>-stripping
# in sandbox_service is a harmless no-op for this model.
GROQ_CODE_MODEL = "openai/gpt-oss-120b"


def _gemini_completion(
    messages: List[dict],
    temperature: float = 0,
    max_tokens: int = 800,
) -> str:
    """Non-streaming Gemini completion via google-generativeai SDK."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    system_instruction = None
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [content]})
        else:
            contents.append({"role": "user", "parts": [content]})

    gen_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config=gen_config,
    )
    response = model.generate_content(contents)
    return response.text or ""


def _gemini_completion_stream(
    messages: List[dict],
    temperature: float = 0,
    max_tokens: int = 800,
) -> Iterator[str]:
    """Streaming Gemini completion via google-generativeai SDK."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    system_instruction = None
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [content]})
        else:
            contents.append({"role": "user", "parts": [content]})

    gen_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config=gen_config,
    )
    response = model.generate_content(contents, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def chat_completion(
    messages: List[dict],
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0,
    max_tokens: int = 800,
    caller: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Non-streaming chat completion with Groq → Gemini fallback.

    Drop-in replacement for `groq.Groq().chat.completions.create()`.
    Returns the response text string directly.

    caller : str, optional
        Short tag identifying the calling code path (e.g. "sql_gen",
        "hybrid_synthesis") — carried into the structured llm_call log line.
    run_id : str, optional
        Explicit trace id override. Usually left unset — resolved from the
        active run_trace() ContextVar instead (see llm_provider module docs).
    """
    # Try Groq first
    try:
        from groq import Groq
        api_key = os.getenv("groq_api_key", "")
        if not api_key:
            raise ValueError("groq_api_key not set")
        client = Groq(api_key=api_key)
        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.monotonic() - start
        logger.info("chat_completion via Groq (%s) in %.2fs", model, elapsed)
        usage = getattr(response, "usage", None)
        _log_llm_call(
            provider="groq", model=model, latency_ms=elapsed * 1000,
            success=True, fallback_triggered=False, caller=caller,
            input_tokens=safe_int(getattr(usage, "prompt_tokens", None)),
            output_tokens=safe_int(getattr(usage, "completion_tokens", None)),
            run_id=run_id,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        if isinstance(exc, ValueError) or _is_retriable_error(exc):
            logger.warning("Groq chat_completion failed (%s), falling back to Gemini", str(exc)[:200])
        else:
            logger.error("Groq chat_completion error (%s), attempting Gemini", str(exc)[:200])

    # Fallback to Gemini
    start = time.monotonic()
    result = _gemini_completion(messages, temperature, max_tokens)
    elapsed = time.monotonic() - start
    logger.info("chat_completion via Gemini (fallback) in %.2fs", elapsed)
    _log_llm_call(
        provider="gemini", model=GEMINI_MODEL, latency_ms=elapsed * 1000,
        success=True, fallback_triggered=True, caller=caller, run_id=run_id,
    )
    return result


def chat_completion_stream(
    messages: List[dict],
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 800,
    caller: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Iterator[str]:
    """Streaming chat completion with Groq → Gemini fallback.

    Yields content tokens as strings. Drop-in replacement for the
    streaming pattern used in db_service and hybrid_service.

    caller : str, optional
        Short tag identifying the calling code path — carried into the
        structured llm_call log line. Token counts aren't logged for
        streaming calls (not consistently available per-chunk); latency,
        provider, and fallback status still are.
    """
    # Try Groq first
    start = time.monotonic()
    try:
        from groq import Groq
        api_key = os.getenv("groq_api_key", "")
        if not api_key:
            raise ValueError("groq_api_key not set")
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        _log_llm_call(
            provider="groq", model=model, latency_ms=(time.monotonic() - start) * 1000,
            success=True, fallback_triggered=False, caller=caller, run_id=run_id,
        )
        return  # success — don't fall through
    except Exception as exc:
        if isinstance(exc, ValueError) or _is_retriable_error(exc):
            logger.warning("Groq streaming failed (%s), falling back to Gemini", str(exc)[:200])
        else:
            logger.error("Groq streaming error (%s), attempting Gemini", str(exc)[:200])

    # Fallback to Gemini streaming
    start = time.monotonic()
    try:
        yield from _gemini_completion_stream(messages, temperature, max_tokens)
        _log_llm_call(
            provider="gemini", model=GEMINI_MODEL, latency_ms=(time.monotonic() - start) * 1000,
            success=True, fallback_triggered=True, caller=caller, run_id=run_id,
        )
    except Exception as exc:
        _log_llm_call(
            provider="gemini", model=GEMINI_MODEL, latency_ms=(time.monotonic() - start) * 1000,
            success=False, fallback_triggered=True, caller=caller, run_id=run_id,
            error_message=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        raise
