"""LLM Provider with automatic fallback — Investor Readiness Sprint.

Provides a unified LLM interface with Groq (Llama 3.3-70b) as primary
and Gemini 2.5 Flash as fallback. On Groq failure (rate limit, 5xx,
timeout), automatically retries with Gemini.

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
    text = chat_completion(messages, model="llama-3.3-70b-versatile")

    # Raw SDK style: streaming
    for token in chat_completion_stream(messages, model="llama-3.3-70b-versatile"):
        print(token, end="")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator, List, Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
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

# Approximate $ per 1K tokens (input, output). For cost visibility only, not
# billing-grade. Unlisted models log a null estimated_cost_usd rather than
# guessing.
_COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    GROQ_MODEL: (0.00059, 0.00079),
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
) -> None:
    """Emit one structured log line for an LLM call. Grep for "llm_call" in
    Railway logs, or filter on the llm_* extra fields if JSON-formatted."""
    logger.info(
        "llm_call",
        extra={
            "llm_provider": provider,
            "llm_model": model,
            "llm_latency_ms": round(latency_ms),
            "llm_input_tokens": input_tokens,
            "llm_output_tokens": output_tokens,
            "llm_estimated_cost_usd": _estimate_cost_usd(model, input_tokens, output_tokens),
            "llm_success": success,
            "llm_fallback_triggered": fallback_triggered,
            "llm_caller": caller,
        },
    )


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
    )


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


def _token_usage_from_response(response) -> tuple[Optional[int], Optional[int]]:
    """Best-effort extraction of (input_tokens, output_tokens) from a
    LangChain ChatModel response. Returns (None, None) if unavailable —
    providers/wrappers don't consistently populate response_metadata."""
    metadata = getattr(response, "response_metadata", None) or {}
    usage = metadata.get("token_usage") or metadata.get("usage_metadata") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


async def call_llm(
    prompt: str,
    *,
    temperature: float = 0,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    caller: Optional[str] = None,
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
            input_tokens=in_tok, output_tokens=out_tok,
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
        input_tokens=in_tok, output_tokens=out_tok,
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
) -> str:
    """Non-streaming chat completion with Groq → Gemini fallback.

    Drop-in replacement for `groq.Groq().chat.completions.create()`.
    Returns the response text string directly.

    caller : str, optional
        Short tag identifying the calling code path (e.g. "sql_gen",
        "hybrid_synthesis") — carried into the structured llm_call log line.
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
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
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
        success=True, fallback_triggered=True, caller=caller,
    )
    return result


def chat_completion_stream(
    messages: List[dict],
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 800,
    caller: Optional[str] = None,
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
            success=True, fallback_triggered=False, caller=caller,
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
            success=True, fallback_triggered=True, caller=caller,
        )
    except Exception:
        _log_llm_call(
            provider="gemini", model=GEMINI_MODEL, latency_ms=(time.monotonic() - start) * 1000,
            success=False, fallback_triggered=True, caller=caller,
        )
        raise
