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


async def call_llm(
    prompt: str,
    *,
    temperature: float = 0,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
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
    return response.content


# ---------------------------------------------------------------------------
# Raw SDK-style completions with Groq → Gemini fallback
# ---------------------------------------------------------------------------

GROQ_CODE_MODEL = "qwen/qwen3-32b"


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
) -> str:
    """Non-streaming chat completion with Groq → Gemini fallback.

    Drop-in replacement for `groq.Groq().chat.completions.create()`.
    Returns the response text string directly.
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
    return result


def chat_completion_stream(
    messages: List[dict],
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Iterator[str]:
    """Streaming chat completion with Groq → Gemini fallback.

    Yields content tokens as strings. Drop-in replacement for the
    streaming pattern used in db_service and hybrid_service.
    """
    # Try Groq first
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
        return  # success — don't fall through
    except Exception as exc:
        if isinstance(exc, ValueError) or _is_retriable_error(exc):
            logger.warning("Groq streaming failed (%s), falling back to Gemini", str(exc)[:200])
        else:
            logger.error("Groq streaming error (%s), attempting Gemini", str(exc)[:200])

    # Fallback to Gemini streaming
    yield from _gemini_completion_stream(messages, temperature, max_tokens)
