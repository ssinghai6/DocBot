"""DOCBOT-2585: every LLM path (models, fallback, LangExtract) can run on Bedrock."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.utils import llm_provider as lp


def _bedrock_env(monkeypatch, **extra):
    monkeypatch.setenv("LLM_PRIMARY", "bedrock")
    monkeypatch.setenv("BEDROCK_API_KEY", "test-bedrock-key")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def _response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def test_env_overrides_bedrock_models(monkeypatch):
    _bedrock_env(monkeypatch, BEDROCK_MODEL="zai.glm-4.7-flash", BEDROCK_CODE_MODEL="nvidia.nemotron-super-3-120b")
    assert lp._primary_target(lp.GROQ_MODEL)[3] == "zai.glm-4.7-flash"
    assert lp._primary_target(lp.GROQ_CODE_MODEL)[3] == "nvidia.nemotron-super-3-120b"


def test_candidate_models_are_costed():
    for model_id in ("zai.glm-4.7-flash", "nvidia.nemotron-nano-3-30b", "nvidia.nemotron-super-3-120b"):
        assert lp._estimate_cost_usd(model_id, 1000, 1000) > 0


def test_fallback_is_gemini_unless_bedrock_fallback_set(monkeypatch):
    _bedrock_env(monkeypatch)
    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL", raising=False)
    assert lp._fallback_label() == ("gemini", lp.GEMINI_MODEL)
    monkeypatch.setenv("BEDROCK_FALLBACK_MODEL", "nvidia.nemotron-super-3-120b")
    assert lp._fallback_label() == ("bedrock", "nvidia.nemotron-super-3-120b")


def test_groq_primary_ignores_bedrock_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PRIMARY", "groq")
    monkeypatch.setenv("BEDROCK_FALLBACK_MODEL", "nvidia.nemotron-super-3-120b")
    assert lp._fallback_label()[0] == "gemini"


def test_chat_completion_falls_back_to_second_bedrock_model(monkeypatch):
    _bedrock_env(monkeypatch, BEDROCK_FALLBACK_MODEL="nvidia.nemotron-super-3-120b")
    client = MagicMock()
    client.chat.completions.create.side_effect = [Exception("429 Too Many Requests"), _response("<think>hm</think> ok")]
    with patch("groq.Groq", return_value=client), patch.object(lp, "_gemini_completion") as gemini, \
            patch.object(lp, "_log_llm_call") as log:
        assert lp.chat_completion([{"role": "user", "content": "q"}]) == "ok"
    gemini.assert_not_called()
    assert client.chat.completions.create.call_args.kwargs["model"] == "nvidia.nemotron-super-3-120b"
    assert (log.call_args.kwargs["provider"], log.call_args.kwargs["fallback_triggered"]) == ("bedrock", True)


def test_stream_falls_back_to_second_bedrock_model(monkeypatch):
    _bedrock_env(monkeypatch, BEDROCK_FALLBACK_MODEL="nvidia.nemotron-super-3-120b")
    chunk = SimpleNamespace(choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="fine"))])
    client = MagicMock()
    client.chat.completions.create.side_effect = [Exception("503 Service Unavailable"), iter([chunk])]
    with patch("groq.Groq", return_value=client), patch.object(lp, "_gemini_completion_stream") as gemini, \
            patch.object(lp, "_log_llm_call"):
        assert "".join(lp.chat_completion_stream([{"role": "user", "content": "q"}])) == "fine"
    gemini.assert_not_called()


def test_call_llm_falls_back_to_second_bedrock_model(monkeypatch):
    _bedrock_env(monkeypatch, BEDROCK_FALLBACK_MODEL="nvidia.nemotron-super-3-120b")
    primary, fallback = MagicMock(), MagicMock()
    primary.ainvoke.side_effect = Exception("500 Internal Server Error")

    async def _ok(_):
        return SimpleNamespace(content="<think>x</think>done", usage_metadata=None, response_metadata={})

    fallback.ainvoke.side_effect = _ok
    with patch.object(lp, "_groq_client", side_effect=[primary, fallback]) as client, \
            patch.object(lp, "_get_gemini_llm") as gemini, patch.object(lp, "_log_llm_call"):
        assert asyncio.run(lp.call_llm("q")) == "done"
    gemini.assert_not_called()
    assert client.call_args.args[:2] == ("bedrock", "nvidia.nemotron-super-3-120b")


@pytest.mark.parametrize("chunks", [
    ["<think>plan</think>Answer"],
    ["<thi", "nk>pl", "an</th", "ink>Answer"],
    ["<reasoning>a</reasoning><think>b</think>Answer"],
])
def test_strip_think_tags(chunks):
    assert "".join(lp._strip_reasoning_stream(iter(chunks))) == "Answer"


def test_extraction_runs_on_bedrock(monkeypatch):
    from api import document_extractor as de

    _bedrock_env(monkeypatch)
    model_cls = MagicMock()
    monkeypatch.setattr(de, "_bedrock_language_model", lambda: model_cls)
    provider, model_id, kwargs = de._extraction_target("")
    assert (provider, model_id) == ("bedrock", "openai.gpt-oss-120b-1:0")
    assert model_cls.call_args.kwargs["base_url"] == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
    assert kwargs["model"] is model_cls.return_value
    assert kwargs["fence_output"] is False and "api_key" not in kwargs


@pytest.mark.parametrize("raw", [
    # shapes gpt-oss returned on Bedrock (after reasoning is stripped)
    '```json{"extractions": [{"metric": "$330 million"}]}```',
    '{\n{\n  "extractions": [{"metric": "$330 million"}]\n}',
    '{"extractions": [{"metric": "$330 million"}]}',
])
def test_first_json_object(raw):
    from api import document_extractor as de

    assert de._first_json_object(raw) == '{"extractions": [{"metric": "$330 million"}]}'


def test_first_json_object_leaves_non_json():
    from api import document_extractor as de

    assert de._first_json_object("no json {here") == "no json {here"


def test_extraction_gemini_when_bedrock_off(monkeypatch):
    from api import document_extractor as de

    monkeypatch.setenv("LLM_PRIMARY", "groq")
    assert de._extraction_target("") is None
    assert de._extraction_target("gk")[:2] == ("gemini", "gemini-2.5-flash")


def test_bedrock_extraction_survives_long_filings(monkeypatch):
    """A 10-K is 100+ chunk calls: one throttled call must not sink the whole extraction."""
    pytest.importorskip("langextract")
    from api import document_extractor as de

    _bedrock_env(monkeypatch)
    _, _, kwargs = de._extraction_target("")
    assert kwargs["model"]._client.max_retries == de._BEDROCK_EXTRACT_RETRIES > 2
    assert kwargs["max_char_buffer"] == de._BEDROCK_CHUNK_CHARS > 1000


def test_llm_configured_with_bedrock_key_only(monkeypatch):
    # Autopilot planner/synthesizer and sandbox code gen gate on this; no Groq key needed on Bedrock
    _bedrock_env(monkeypatch)
    monkeypatch.delenv("groq_api_key", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert lp.llm_configured() is True
    monkeypatch.setenv("LLM_PRIMARY", "groq")
    assert lp.llm_configured() is False


def test_autopilot_planner_not_degraded_without_groq_key(monkeypatch):
    from api import autopilot_service as ap

    _bedrock_env(monkeypatch)
    monkeypatch.delenv("groq_api_key", raising=False)
    state = {"question": "Compare revenue to the books", "has_db": True, "has_docs": True}
    with patch("api.utils.llm_provider.chat_completion", return_value='["sql_query: revenue by year"]'):
        out = ap._planner_node(state)
        out = asyncio.run(out) if asyncio.iscoroutine(out) else out
    assert ap._DEGR_PLANNER_FALLBACK not in (out.get("degradation_reasons") or [])
