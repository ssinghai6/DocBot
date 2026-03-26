"""Unit tests for api.utils.llm_provider — LLM fallback provider.

All LLM calls are mocked. No network access required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    """Set API keys in env for all tests."""
    monkeypatch.setenv("groq_api_key", "test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


# ---------------------------------------------------------------------------
# get_llm tests
# ---------------------------------------------------------------------------


def test_get_llm_returns_groq_by_default():
    """get_llm() returns a Groq model when key is available."""
    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_instance = MagicMock()
        mock_groq.return_value = mock_instance

        from api.utils.llm_provider import get_llm
        result = get_llm(groq_api_key="test-key")

        mock_groq.assert_called_once()
        assert result is mock_instance


def test_get_llm_forced_gemini():
    """get_llm(provider='gemini') returns Gemini model."""
    with patch("api.utils._gemini_wrapper.GeminiChatModel") as mock_gemini:
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        from api.utils.llm_provider import get_llm
        result = get_llm(provider="gemini", gemini_api_key="test-key")

        mock_gemini.assert_called_once()
        assert result is mock_instance


def test_get_llm_forced_groq():
    """get_llm(provider='groq') returns Groq model."""
    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_instance = MagicMock()
        mock_groq.return_value = mock_instance

        from api.utils.llm_provider import get_llm
        result = get_llm(provider="groq", groq_api_key="test-key")

        mock_groq.assert_called_once()
        assert result is mock_instance


def test_get_llm_falls_back_to_gemini_on_groq_failure():
    """When Groq instantiation fails, get_llm() falls back to Gemini."""
    with patch("langchain_groq.ChatGroq", side_effect=ValueError("no key")), \
         patch("api.utils._gemini_wrapper.GeminiChatModel") as mock_gemini:
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        from api.utils.llm_provider import get_llm
        result = get_llm(groq_api_key="", gemini_api_key="test-key")

        mock_gemini.assert_called_once()
        assert result is mock_instance


# ---------------------------------------------------------------------------
# _is_retriable_error tests
# ---------------------------------------------------------------------------


def test_retriable_error_rate_limit():
    """Rate limit errors should be retriable."""
    from api.utils.llm_provider import _is_retriable_error
    assert _is_retriable_error(Exception("429 Too Many Requests"))
    assert _is_retriable_error(Exception("rate limit exceeded"))
    assert _is_retriable_error(Exception("Rate_Limit_Error"))


def test_retriable_error_server_errors():
    """5xx errors should be retriable."""
    from api.utils.llm_provider import _is_retriable_error
    assert _is_retriable_error(Exception("500 Internal Server Error"))
    assert _is_retriable_error(Exception("502 Bad Gateway"))
    assert _is_retriable_error(Exception("503 Service Unavailable"))
    assert _is_retriable_error(Exception("504 Gateway Timeout"))


def test_retriable_error_timeout():
    """Timeout errors should be retriable."""
    from api.utils.llm_provider import _is_retriable_error
    assert _is_retriable_error(Exception("Request timed out"))
    assert _is_retriable_error(Exception("connection timeout"))


def test_non_retriable_error():
    """Normal errors should not be retriable."""
    from api.utils.llm_provider import _is_retriable_error
    assert not _is_retriable_error(Exception("Invalid API key"))
    assert not _is_retriable_error(Exception("Model not found"))
    assert not _is_retriable_error(ValueError("Bad input"))


# ---------------------------------------------------------------------------
# call_llm tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_success_with_groq():
    """call_llm() uses Groq on success."""
    mock_response = MagicMock()
    mock_response.content = "test response"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("api.utils.llm_provider._get_groq_llm", return_value=mock_llm):
        from api.utils.llm_provider import call_llm
        result = await call_llm("test prompt")

    assert result == "test response"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_call_llm_falls_back_on_rate_limit():
    """call_llm() falls back to Gemini when Groq hits rate limit."""
    groq_llm = AsyncMock()
    groq_llm.ainvoke = AsyncMock(side_effect=Exception("429 Too Many Requests"))

    gemini_response = MagicMock()
    gemini_response.content = "gemini response"
    gemini_llm = AsyncMock()
    gemini_llm.ainvoke = AsyncMock(return_value=gemini_response)

    with patch("api.utils.llm_provider._get_groq_llm", return_value=groq_llm), \
         patch("api.utils.llm_provider._get_gemini_llm", return_value=gemini_llm):
        from api.utils.llm_provider import call_llm
        result = await call_llm("test prompt")

    assert result == "gemini response"


@pytest.mark.asyncio
async def test_call_llm_falls_back_on_timeout():
    """call_llm() falls back to Gemini when Groq times out."""
    groq_llm = AsyncMock()
    groq_llm.ainvoke = AsyncMock(side_effect=Exception("Request timed out"))

    gemini_response = MagicMock()
    gemini_response.content = "gemini fallback"
    gemini_llm = AsyncMock()
    gemini_llm.ainvoke = AsyncMock(return_value=gemini_response)

    with patch("api.utils.llm_provider._get_groq_llm", return_value=groq_llm), \
         patch("api.utils.llm_provider._get_gemini_llm", return_value=gemini_llm):
        from api.utils.llm_provider import call_llm
        result = await call_llm("test prompt")

    assert result == "gemini fallback"


@pytest.mark.asyncio
async def test_call_llm_falls_back_when_no_groq_key():
    """call_llm() falls back to Gemini when Groq key is missing."""
    gemini_response = MagicMock()
    gemini_response.content = "gemini only"
    gemini_llm = AsyncMock()
    gemini_llm.ainvoke = AsyncMock(return_value=gemini_response)

    with patch("api.utils.llm_provider._get_groq_llm", side_effect=ValueError("no key")), \
         patch("api.utils.llm_provider._get_gemini_llm", return_value=gemini_llm):
        from api.utils.llm_provider import call_llm
        result = await call_llm("test prompt")

    assert result == "gemini only"
