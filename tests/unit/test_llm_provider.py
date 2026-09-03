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


# ---------------------------------------------------------------------------
# Thin LLM telemetry — DOCBOT-1401
# ---------------------------------------------------------------------------


class TestEstimateCostUsd:
    def test_known_model_computes_cost(self):
        from api.utils.llm_provider import GROQ_MODEL, _estimate_cost_usd
        cost = _estimate_cost_usd(GROQ_MODEL, 1000, 1000)
        assert cost is not None
        assert cost > 0

    def test_unknown_model_returns_none(self):
        from api.utils.llm_provider import _estimate_cost_usd
        assert _estimate_cost_usd("some-unlisted-model", 1000, 1000) is None

    def test_missing_token_counts_returns_none(self):
        from api.utils.llm_provider import GROQ_MODEL, _estimate_cost_usd
        assert _estimate_cost_usd(GROQ_MODEL, None, 1000) is None
        assert _estimate_cost_usd(GROQ_MODEL, 1000, None) is None


class TestLogLlmCallSurvivesDefaultFormatter:
    """DOCBOT-1402 regression: extra={...} fields are silently dropped by
    Python's default logging formatter (logging.basicConfig() in api/index.py
    sets no custom Formatter). caplog reads LogRecord attributes directly and
    bypasses formatting entirely, so it could not have caught this — these
    tests render through an actual logging.Formatter to prove the payload
    survives the real Railway/stdout path, not just the test harness.
    """

    def test_message_is_valid_json_through_default_formatter(self, caplog):
        import json
        import logging
        from api.utils.llm_provider import _log_llm_call

        with caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            _log_llm_call(
                provider="groq", model="openai/gpt-oss-20b", latency_ms=123.4,
                success=True, fallback_triggered=False, caller="sql_gen",
                input_tokens=50, output_tokens=20,
            )

        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1

        # Render through the SAME formatter shape logging.basicConfig() uses —
        # no custom Formatter, no extra field access. If the payload only
        # lived in `extra=`, this would render as the bare word "llm_call"
        # with none of the fields, which is exactly what DOCBOT-1402 found.
        formatted = logging.Formatter("%(message)s").format(records[0])
        parsed = json.loads(formatted)  # raises if it's not real JSON in the message body
        assert parsed["event"] == "llm_call"
        assert parsed["llm_provider"] == "groq"
        assert parsed["llm_model"] == "openai/gpt-oss-20b"
        assert parsed["llm_latency_ms"] == 123
        assert parsed["llm_caller"] == "sql_gen"
        assert parsed["llm_input_tokens"] == 50
        assert parsed["llm_output_tokens"] == 20
        assert parsed["llm_success"] is True
        assert parsed["llm_fallback_triggered"] is False


class TestLogLlmCall:
    def test_emits_structured_log_record(self, caplog):
        import logging
        from api.utils.llm_provider import _log_llm_call

        with caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            _log_llm_call(
                provider="groq", model="openai/gpt-oss-20b", latency_ms=123.4,
                success=True, fallback_triggered=False, caller="sql_gen",
                input_tokens=50, output_tokens=20,
            )

        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1
        record = records[0]
        assert record.llm_provider == "groq"
        assert record.llm_model == "openai/gpt-oss-20b"
        assert record.llm_latency_ms == 123
        assert record.llm_success is True
        assert record.llm_fallback_triggered is False
        assert record.llm_caller == "sql_gen"
        assert record.llm_input_tokens == 50
        assert record.llm_output_tokens == 20

    def test_log_external_llm_call_delegates(self, caplog):
        import logging
        from api.utils.llm_provider import log_external_llm_call

        with caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            log_external_llm_call(
                provider="groq", model="openai/gpt-oss-20b", latency_ms=50.0,
                success=True, caller="intent_classification",
            )

        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1
        assert records[0].llm_caller == "intent_classification"
        assert records[0].llm_fallback_triggered is False


class TestChatCompletionTelemetry:
    def test_success_logs_provider_groq(self, caplog):
        import logging

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("groq.Groq", return_value=mock_client), \
             caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            from api.utils.llm_provider import chat_completion
            result = chat_completion([{"role": "user", "content": "hi"}], caller="sql_gen")

        assert result == "hello"
        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1
        assert records[0].llm_provider == "groq"
        assert records[0].llm_caller == "sql_gen"
        assert records[0].llm_input_tokens == 10
        assert records[0].llm_output_tokens == 5

    def test_fallback_logs_provider_gemini(self, caplog):
        import logging

        with patch("groq.Groq", side_effect=Exception("503 Service Unavailable")), \
             patch("api.utils.llm_provider._gemini_completion", return_value="gemini says hi"), \
             caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            from api.utils.llm_provider import chat_completion
            result = chat_completion([{"role": "user", "content": "hi"}], caller="hybrid_synthesis")

        assert result == "gemini says hi"
        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1
        assert records[0].llm_provider == "gemini"
        assert records[0].llm_fallback_triggered is True
        assert records[0].llm_caller == "hybrid_synthesis"


class TestChatCompletionStreamTelemetry:
    def test_success_logs_provider_groq(self, caplog):
        import logging

        def _make_chunk(content):
            chunk = MagicMock()
            chunk.choices[0].delta.content = content
            return chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [_make_chunk("hel"), _make_chunk("lo")]

        with patch("groq.Groq", return_value=mock_client), \
             caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            from api.utils.llm_provider import chat_completion_stream
            tokens = list(chat_completion_stream([{"role": "user", "content": "hi"}], caller="autopilot_synth"))

        assert "".join(tokens) == "hello"
        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1
        assert records[0].llm_provider == "groq"
        assert records[0].llm_fallback_triggered is False
        assert records[0].llm_caller == "autopilot_synth"

    def test_fallback_logs_provider_gemini(self, caplog):
        import logging

        with patch("groq.Groq", side_effect=Exception("timeout")), \
             patch("api.utils.llm_provider._gemini_completion_stream", return_value=iter(["gem", "ini"])), \
             caplog.at_level(logging.INFO, logger="api.utils.llm_provider"):
            from api.utils.llm_provider import chat_completion_stream
            tokens = list(chat_completion_stream([{"role": "user", "content": "hi"}], caller="autopilot_synth"))

        assert "".join(tokens) == "gemini"
        records = [r for r in caplog.records if getattr(r, "event", None) == "llm_call"]
        assert len(records) == 1
        assert records[0].llm_provider == "gemini"
        assert records[0].llm_fallback_triggered is True
        assert records[0].llm_success is True


# ---------------------------------------------------------------------------
# Static coverage check — DOCBOT-1402
#
# DOCBOT-1401 added a `caller=` kwarg to chat_completion/chat_completion_stream/
# call_llm for per-path telemetry attribution, but only wired it at 3 of 12
# production call sites — the other 9 (SQL gen, autopilot, sandbox code-gen,
# hybrid synthesis) silently logged `llm_caller: null`. No runtime test could
# catch this (every individual call site's own unit tests mock the response
# and never assert on the caller kwarg). This is a static AST check instead:
# every call to these three functions anywhere under api/ must pass `caller=`.
# ---------------------------------------------------------------------------


class TestCallerKwargCoverage:
    def test_every_llm_provider_call_passes_caller(self):
        import ast
        from pathlib import Path

        api_dir = Path(__file__).resolve().parents[2] / "api"
        target_functions = {"chat_completion", "chat_completion_stream", "call_llm"}
        missing: list[str] = []

        for py_file in api_dir.rglob("*.py"):
            if py_file.name == "llm_provider.py":
                continue  # the definitions themselves, not call sites
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None
                )
                if name not in target_functions:
                    continue
                has_caller = any(kw.arg == "caller" for kw in node.keywords)
                if not has_caller:
                    missing.append(f"{py_file.relative_to(api_dir.parent)}:{node.lineno} {name}()")

        assert not missing, (
            "These calls to chat_completion/chat_completion_stream/call_llm are "
            "missing caller= — per-path cost/latency telemetry will show "
            "llm_caller: null for them:\n" + "\n".join(missing)
        )


class TestModelConstants:
    """DOCBOT-1403: GROQ_MODEL was 'llama-3.3-70b-versatile', a model Groq
    removed from its catalog entirely — every default-model call was
    silently 404ing. Guard against picking a model that isn't in the cost
    table (a maintenance foot-gun: _COST_PER_1K_TOKENS duplicates
    GROQ_CODE_MODEL's value as a literal since GROQ_CODE_MODEL is defined
    later in the module) and against accidentally reintroducing the dead
    model string anywhere.
    """

    def test_groq_model_is_not_the_decommissioned_model(self):
        from api.utils.llm_provider import GROQ_MODEL
        assert GROQ_MODEL != "llama-3.3-70b-versatile"

    def test_groq_model_distinct_from_code_model(self):
        # sandbox_service's retry ladder branches on `_model != GROQ_CODE_MODEL`;
        # if these collide, that branch silently stops meaning anything.
        from api.utils.llm_provider import GROQ_CODE_MODEL, GROQ_MODEL
        assert GROQ_MODEL != GROQ_CODE_MODEL

    def test_code_model_cost_entry_matches_constant(self):
        # _COST_PER_1K_TOKENS keys GROQ_CODE_MODEL's rate by a literal string
        # (GROQ_CODE_MODEL is defined after this dict in the module). Catch
        # drift if one changes without the other.
        from api.utils.llm_provider import GROQ_CODE_MODEL, _COST_PER_1K_TOKENS
        assert GROQ_CODE_MODEL in _COST_PER_1K_TOKENS

    def test_groq_model_has_a_cost_entry(self):
        from api.utils.llm_provider import GROQ_MODEL, _COST_PER_1K_TOKENS
        assert GROQ_MODEL in _COST_PER_1K_TOKENS
