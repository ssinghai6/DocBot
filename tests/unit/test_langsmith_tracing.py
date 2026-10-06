"""DOCBOT-1509 — LangSmith tracing (metadata only). No network: the LangSmith
client is always a MagicMock. ``client.create_run`` is the wire boundary, so
its kwargs are the exact payload that would be sent.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import api.utils.langsmith_tracing as lst
from api.utils.llm_provider import _log_llm_call, chat_completion, run_trace

SECRET_PROMPT = "SECRET-PROMPT-7f3a9c"
SECRET_RESPONSE = "SECRET-RESPONSE-b81d2e"
SECRET_USER_DATA = "SECRET-ROWDATA-4c0e11"


@pytest.fixture(autouse=True)
def _groq_key(monkeypatch) -> None:
    """chat_completion checks this env var before calling the (mocked) Groq SDK."""
    monkeypatch.setenv("groq_api_key", "test-groq-key")


@pytest.fixture
def client(monkeypatch) -> MagicMock:
    """Tracing ON with a mocked LangSmith client. Never touches the network."""
    mock_client = MagicMock(name="langsmith_client")
    monkeypatch.setattr(lst, "_config", lst.TracingConfig(enabled=True, api_key="test-key"))
    monkeypatch.setattr(lst, "_get_client", lambda: mock_client)
    yield mock_client
    lst.flush()


@pytest.fixture
def tracing_off(monkeypatch) -> MagicMock:
    mock_get = MagicMock(name="_get_client")
    monkeypatch.setattr(lst, "_config", lst.TracingConfig(enabled=False, api_key=None))
    monkeypatch.setattr(lst, "_get_client", mock_get)
    return mock_get


def _llm_payload(**overrides: Any) -> dict:
    base = dict(
        provider="groq", model="openai/gpt-oss-20b", latency_ms=120.0,
        success=True, fallback_triggered=False, caller="sql_gen",
        input_tokens=50, output_tokens=20,
    )
    base.update(overrides)
    _log_llm_call(**base)


def _create_run_calls(client: MagicMock) -> list[dict]:
    lst.flush()
    return [c.kwargs for c in client.create_run.call_args_list]


def _mock_groq(response_text: str = SECRET_RESPONSE):
    """Patch the groq SDK used inside chat_completion."""
    groq_instance = MagicMock()
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = 11
    usage.completion_tokens = 7
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    groq_instance.chat.completions.create.return_value = completion
    return patch("groq.Groq", return_value=groq_instance)


# ---------------------------------------------------------------------------
# (a) Tracing OFF: zero client calls, no network
# ---------------------------------------------------------------------------


def test_tracing_off_makes_zero_client_calls(tracing_off):
    _llm_payload()
    with run_trace("a" * 32, name="autopilot"):
        _llm_payload(caller="inside_investigation")
    lst.flush()
    tracing_off.assert_not_called()


def test_load_config_requires_both_flag_and_key():
    assert lst.load_config({}).enabled is False
    assert lst.load_config({"LANGSMITH_TRACING": "true"}).enabled is False
    assert lst.load_config({"LANGSMITH_API_KEY": "k"}).enabled is False
    assert lst.load_config({"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": "k"}).enabled is False
    assert lst.load_config({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "  "}).enabled is False
    cfg = lst.load_config({"LANGSMITH_TRACING": "True", "LANGSMITH_API_KEY": "k"})
    assert cfg.enabled is True and cfg.api_key == "k"


def test_sdk_auto_tracing_is_pinned_off_in_subprocess():
    """LangChain/LangSmith automatic tracing ships full inputs and outputs, so
    it must stay off even when the user sets LANGSMITH_TRACING=true. Checked in
    a fresh interpreter because langsmith caches env lookups per process."""
    code = (
        "import api.utils.langsmith_tracing as t\n"
        "from langsmith.utils import tracing_is_enabled\n"
        "print(t.is_enabled(), bool(tracing_is_enabled()))\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "dummy-not-real",
        "PYTHONPATH": ".",
    }
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env=env, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True False"


# ---------------------------------------------------------------------------
# (b) Security: the payload the client receives contains no prompt/response
# ---------------------------------------------------------------------------


def test_chat_completion_payload_contains_no_prompt_or_response(client):
    messages = [{"role": "user", "content": SECRET_PROMPT}]
    with _mock_groq(SECRET_RESPONSE):
        result = chat_completion(messages, caller="hybrid_synthesis")
    assert result == SECRET_RESPONSE

    calls = _create_run_calls(client)
    assert len(calls) == 1
    run = calls[0]

    # Hard guarantee: the serialized payload never contains prompt or response text.
    wire = json.dumps(run, default=str)
    assert SECRET_PROMPT not in wire
    assert SECRET_RESPONSE not in wire

    assert run["inputs"] == {}
    assert set(run["outputs"].keys()) <= {"usage_metadata"}
    assert run["outputs"]["usage_metadata"] == {
        "input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
    }
    allowed = set(lst._LLM_METADATA_KEYS) | {"ls_provider", "ls_model_name"}
    assert set(run["extra"]["metadata"].keys()) <= allowed
    assert run["extra"]["metadata"]["caller"] == "hybrid_synthesis"
    assert run["run_type"] == "llm"


def test_error_message_text_never_reaches_payload(client):
    """Streaming failure path passes error_message containing user-derived text.
    Only the exception class name may be sent."""
    class UpstreamBoom(Exception):
        pass

    def _failing_stream(*_a, **_k):
        raise UpstreamBoom(f"bad row {SECRET_USER_DATA}")

    from api.utils import llm_provider

    with patch.object(llm_provider, "_gemini_completion_stream", side_effect=UpstreamBoom(f"bad row {SECRET_USER_DATA}")):
        with patch("groq.Groq", side_effect=ValueError("groq_api_key not set")):
            with pytest.raises(UpstreamBoom):
                list(llm_provider.chat_completion_stream([{"role": "user", "content": "q"}], caller="rag"))

    calls = _create_run_calls(client)
    assert len(calls) == 1
    wire = json.dumps(calls[0], default=str)
    assert SECRET_USER_DATA not in wire
    assert calls[0]["error"] == "UpstreamBoom"
    assert calls[0]["extra"]["metadata"]["success"] is False


def test_build_spec_drops_unknown_and_error_message_keys():
    spec = lst.build_llm_run_spec({
        "run_id": "r", "llm_provider": "groq", "llm_model": "m",
        "error_message": f"leak {SECRET_USER_DATA}", "messages": [SECRET_PROMPT],
        "response": SECRET_RESPONSE,
    })
    wire = json.dumps(spec, default=str)
    assert SECRET_USER_DATA not in wire
    assert SECRET_PROMPT not in wire
    assert SECRET_RESPONSE not in wire


# ---------------------------------------------------------------------------
# (c) Failure isolation: LangSmith failing never fails or slows the LLM call
# ---------------------------------------------------------------------------


def test_client_construction_failure_does_not_propagate(monkeypatch):
    monkeypatch.setattr(lst, "_config", lst.TracingConfig(enabled=True, api_key="k"))
    monkeypatch.setattr(lst, "_get_client", MagicMock(side_effect=RuntimeError("client boom")))
    with _mock_groq(SECRET_RESPONSE):
        assert chat_completion([{"role": "user", "content": "hi"}]) == SECRET_RESPONSE
    lst.flush()


def test_client_post_raising_does_not_propagate(monkeypatch):
    failing = MagicMock(name="client")
    failing.create_run.side_effect = ConnectionError("langsmith is down")
    monkeypatch.setattr(lst, "_config", lst.TracingConfig(enabled=True, api_key="k"))
    monkeypatch.setattr(lst, "_get_client", lambda: failing)
    with _mock_groq(SECRET_RESPONSE):
        assert chat_completion([{"role": "user", "content": "hi"}]) == SECRET_RESPONSE
    lst.flush()
    failing.create_run.assert_called()


def test_emit_raising_synchronously_does_not_propagate(client):
    with patch.object(lst, "_submit", side_effect=RuntimeError("emit boom")):
        _llm_payload()  # must not raise


def test_slow_langsmith_does_not_block_the_caller(monkeypatch):
    release = threading.Event()
    slow = MagicMock(name="slow_client")
    slow.create_run.side_effect = lambda **_k: release.wait(timeout=10)
    monkeypatch.setattr(lst, "_config", lst.TracingConfig(enabled=True, api_key="k"))
    monkeypatch.setattr(lst, "_get_client", lambda: slow)
    try:
        start = time.monotonic()
        _llm_payload()
        assert time.monotonic() - start < 0.5
    finally:
        release.set()
        lst.flush()


# ---------------------------------------------------------------------------
# (d) DOCBOT-1501 run_id is attached; parent/child linking under one trace
# ---------------------------------------------------------------------------


def test_run_id_from_run_trace_attached_as_metadata(client):
    rid = uuid.uuid4().hex
    with run_trace(rid):
        _llm_payload(caller="sql_gen")
    calls = _create_run_calls(client)
    assert len(calls) == 1
    assert calls[0]["extra"]["metadata"]["run_id"] == rid


def test_investigation_root_uses_docbot_run_id_as_trace_id(client):
    rid = uuid.uuid4().hex
    with run_trace(rid, name="autopilot"):
        _llm_payload(caller="planner")
    calls = _create_run_calls(client)
    root = next(c for c in calls if c["run_type"] == "chain")
    child = next(c for c in calls if c["run_type"] == "llm")
    assert str(root["id"]) == str(uuid.UUID(rid))
    assert str(child["parent_run_id"]) == str(root["id"])
    assert str(child["trace_id"]) == str(root["trace_id"])


def test_nested_run_trace_reuses_outer_root(client):
    rid = uuid.uuid4().hex
    with run_trace(rid, name="autopilot"):
        with run_trace(rid, name="deep_retrieve"):
            _llm_payload(caller="deep_retrieve_planner")
    calls = _create_run_calls(client)
    roots = [c for c in calls if c["run_type"] == "chain"]
    assert len(roots) == 1
    assert roots[0]["name"] == "autopilot"


def test_child_llm_runs_group_under_langgraph_node_dispatch(client):
    """Verifies LangGraph node dispatch carries the investigation parent into
    LLM runs made inside nodes (explicit grouping, not LangChain auto-tracing)."""
    from typing import TypedDict as _TD

    from langgraph.graph import END, START, StateGraph

    class State(_TD):
        rid: str

    async def planner(state: State) -> dict:
        _llm_payload(caller="planner")
        return {}

    async def executor(state: State) -> dict:
        _llm_payload(caller="executor")
        return {}

    graph = StateGraph(State)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", END)
    app = graph.compile()

    import asyncio

    rid = uuid.uuid4().hex

    async def _go():
        with run_trace(rid, name="autopilot"):
            await app.ainvoke({"rid": rid})

    asyncio.run(_go())

    calls = _create_run_calls(client)
    root = next(c for c in calls if c["run_type"] == "chain")
    llm_runs = [c for c in calls if c["run_type"] == "llm"]
    assert {c["name"] for c in llm_runs} == {"llm:planner", "llm:executor"}
    for child in llm_runs:
        assert str(child["parent_run_id"]) == str(root["id"])
        assert str(child["trace_id"]) == str(uuid.UUID(rid))


def test_root_scope_disables_langchain_auto_tracing(client):
    from langsmith.utils import tracing_is_enabled

    with run_trace(uuid.uuid4().hex, name="autopilot"):
        assert not tracing_is_enabled()
