"""DOCBOT-2584: LLM_PRIMARY=bedrock routes gpt-oss calls to Amazon Bedrock."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.utils import llm_provider as lp


def _bedrock_env(monkeypatch):
    monkeypatch.setenv("LLM_PRIMARY", "bedrock")
    monkeypatch.setenv("BEDROCK_API_KEY", "test-bedrock-key")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")


def test_default_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PRIMARY", raising=False)
    monkeypatch.setenv("groq_api_key", "g")
    assert lp._primary_target(lp.GROQ_MODEL) == ("groq", "g", None, lp.GROQ_MODEL)


def test_bedrock_target_maps_models(monkeypatch):
    _bedrock_env(monkeypatch)
    provider, key, base_url, model_id = lp._primary_target(lp.GROQ_CODE_MODEL)
    assert (provider, key, model_id) == ("bedrock", "test-bedrock-key", "openai.gpt-oss-120b-1:0")
    assert base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert lp._primary_target(lp.GROQ_MODEL)[3] == "openai.gpt-oss-20b-1:0"


def test_bedrock_without_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PRIMARY", "bedrock")
    monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        lp._primary_target(lp.GROQ_MODEL)


def test_bedrock_models_are_costed():
    # spend caps (PLATFORM_DAILY_USD_CAP etc.) only count calls with a known rate
    for model_id in lp._BEDROCK_MODEL_IDS.values():
        assert lp._estimate_cost_usd(model_id, 1000, 1000) > 0


@pytest.mark.parametrize("chunks", [
    ["<reasoning>think</reasoning>Answer"],
    ["<reas", "oning>th", "ink</reas", "oning>Ans", "wer"],
    ["<", "r", "easoning>x", "<", "/reasoning>", "Answer"],
])
def test_strip_reasoning_stream(chunks):
    assert "".join(lp._strip_reasoning_stream(iter(chunks))) == "Answer"


def test_strip_reasoning_keeps_plain_text():
    assert lp._strip_reasoning("a < b and <b>bold</b>") == "a < b and <b>bold</b>"


def test_chat_completion_uses_bedrock(monkeypatch):
    _bedrock_env(monkeypatch)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="<reasoning>x</reasoning> 42"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    with patch("groq.Groq", return_value=client) as groq_cls, patch.object(lp, "_log_llm_call") as log:
        assert lp.chat_completion([{"role": "user", "content": "q"}]) == "42"
    assert groq_cls.call_args.kwargs["base_url"] == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert client.chat.completions.create.call_args.kwargs["model"] == "openai.gpt-oss-20b-1:0"
    assert log.call_args.kwargs["provider"] == "bedrock"


def test_stream_logs_estimated_tokens(monkeypatch):
    _bedrock_env(monkeypatch)
    chunk = SimpleNamespace(choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="abcdefgh"))])
    client = MagicMock()
    client.chat.completions.create.return_value = iter([chunk])
    with patch("groq.Groq", return_value=client), patch.object(lp, "_log_llm_call") as log:
        assert "".join(lp.chat_completion_stream([{"role": "user", "content": "x" * 400}])) == "abcdefgh"
    kw = log.call_args.kwargs
    assert (kw["input_tokens"], kw["output_tokens"]) == (100, 2)
    assert lp._estimate_cost_usd(kw["model"], kw["input_tokens"], kw["output_tokens"]) > 0


class _Sent(Exception):
    pass


async def _request_kwargs(llm) -> dict:
    """Run ainvoke against a fake client and return the kwargs ChatGroq would send."""
    seen = {}

    async def create(**kw):
        seen.update(kw)
        raise _Sent

    llm.async_client = SimpleNamespace(create=create)
    with pytest.raises(_Sent):
        await llm.ainvoke("hi")
    return seen


@pytest.mark.asyncio
async def test_bedrock_langchain_client_omits_service_tier(monkeypatch):
    # DOCBOT-2604: Bedrock 400s on service_tier, so call_llm/get_llm callers failed silently
    _bedrock_env(monkeypatch)
    monkeypatch.setenv("BEDROCK_FALLBACK_MODEL", "openai.gpt-oss-20b-1:0")
    for llm in (lp._get_groq_llm(), lp._get_fallback_llm()):
        sent = await _request_kwargs(llm)
        assert "service_tier" not in sent and sent["model"].startswith("openai.gpt-oss")


@pytest.mark.asyncio
async def test_groq_langchain_client_keeps_service_tier(monkeypatch):
    monkeypatch.delenv("LLM_PRIMARY", raising=False)
    monkeypatch.setenv("groq_api_key", "g")
    assert (await _request_kwargs(lp._get_groq_llm()))["service_tier"] == "on_demand"
