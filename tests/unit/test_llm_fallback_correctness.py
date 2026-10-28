"""DOCBOT-2307: LLM fallback correctness.

Streaming falls back to Gemini only before the first token; the intent
classifier runs on the async provider path; doc chat goes through the
provider; the "LLM configured" gate accepts any provider key. No network.
"""

from __future__ import annotations

import ast
import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest

from api.utils import llm_provider
from api.utils.exceptions import StreamInterruptedError, safe_error_message

MSGS = [{"role": "user", "content": "q"}]


def _chunk(content):
    c = MagicMock()
    c.choices[0].delta.content = content
    c.choices[0].finish_reason = None
    return c


def _groq_stream_then_fail(*tokens):
    def _gen():
        for t in tokens:
            yield _chunk(t)
        raise RuntimeError("503 service unavailable")
    client = MagicMock()
    client.chat.completions.create.return_value = _gen()
    return client


def test_pre_token_failure_falls_back_to_gemini_single_copy(monkeypatch):
    monkeypatch.setenv("groq_api_key", "k")
    with patch("groq.Groq", return_value=_groq_stream_then_fail()), \
         patch.object(llm_provider, "_gemini_completion_stream", return_value=iter(["Gem", "ini"])) as gem:
        out = list(llm_provider.chat_completion_stream(MSGS, caller="t"))
    assert out == ["Gem", "ini"]
    gem.assert_called_once()


def test_post_token_failure_raises_interrupt_without_gemini(monkeypatch):
    monkeypatch.setenv("groq_api_key", "k")
    out: list[str] = []
    with patch("groq.Groq", return_value=_groq_stream_then_fail("Hel", "lo")), \
         patch.object(llm_provider, "_gemini_completion_stream") as gem:
        with pytest.raises(StreamInterruptedError):
            for tok in llm_provider.chat_completion_stream(MSGS, caller="t"):
                out.append(tok)
    assert out == ["Hel", "lo"]  # no duplicated / restarted text
    gem.assert_not_called()


def test_interrupt_message_is_safe_for_clients():
    assert safe_error_message(StreamInterruptedError()) == "Answer interrupted, please retry."
    assert "interrupted" not in safe_error_message(RuntimeError("secret internals"))


@pytest.mark.asyncio
async def test_async_stream_propagates_interrupt(monkeypatch):
    monkeypatch.setenv("groq_api_key", "k")
    out: list[str] = []
    with patch("groq.Groq", return_value=_groq_stream_then_fail("a")), \
         patch.object(llm_provider, "_gemini_completion_stream") as gem:
        with pytest.raises(StreamInterruptedError):
            async for tok in llm_provider.achat_completion_stream(MSGS, caller="t"):
                out.append(tok)
    assert out == ["a"]
    gem.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_path_is_metered_through_async_stream(monkeypatch):
    """Gemini fallback still records usage, and the ContextVar-bound run_id
    reaches the worker thread."""
    monkeypatch.setenv("groq_api_key", "k")
    with patch("groq.Groq", side_effect=RuntimeError("timeout")), \
         patch.object(llm_provider, "_gemini_completion_stream", return_value=iter(["ok"])), \
         patch.object(llm_provider, "meter_llm_call") as meter, \
         llm_provider.run_trace("run-2307"):
        out = [t async for t in llm_provider.achat_completion_stream(MSGS, caller="t")]
    assert out == ["ok"]
    meter.assert_called_once()
    assert meter.call_args.kwargs["provider"] == "gemini"
    assert meter.call_args.kwargs["run_id"] == "run-2307"


@pytest.mark.asyncio
async def test_intent_classifier_async_path_and_token_budget():
    from api.hybrid_service import classify_intent

    loop_thread = threading.get_ident()
    seen: dict = {}

    def _fake(messages, **kw):
        seen["thread"] = threading.get_ident()
        seen.update(kw)
        return "sql"

    with patch.object(llm_provider, "chat_completion", side_effect=_fake), \
         patch("api.utils.llm_cache.get_cached_response", return_value=None), \
         patch("api.utils.llm_cache.set_cached_response"):
        result = await classify_intent(
            question="how many orders?", has_db=True, has_docs=True,
            session_id="s", async_session_factory=MagicMock(),
        )
    assert result == "sql"
    assert seen["thread"] != loop_thread  # not on the event loop
    assert seen["max_tokens"] >= 200
    assert seen["caller"] == "intent_classification"


def test_doc_chat_goes_through_provider():
    import api.index as idx

    tree = ast.parse(inspect.getsource(idx.chat).lstrip())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert "achat_completion_stream" in names
    assert "get_llm" not in names and "astream" not in names
    assert "llm_configured" in names


@pytest.mark.parametrize("groq,gemini,expected", [
    ("g", None, True), (None, "m", True), ("g", "m", True), (None, None, False),
])
def test_llm_configured_accepts_any_provider_key(monkeypatch, groq, gemini, expected):
    for name, val in (("groq_api_key", groq), ("GEMINI_API_KEY", gemini)):
        if val:
            monkeypatch.setenv(name, val)
        else:
            monkeypatch.delenv(name, raising=False)
    assert llm_provider.llm_configured() is expected
