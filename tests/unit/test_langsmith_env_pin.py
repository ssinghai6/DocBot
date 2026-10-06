"""DOCBOT-1509 regression: no LangSmith auto-trace env var may ship content.

Every case runs in a FRESH interpreter. langsmith caches env lookups and
langchain reads its v1 variables at configure time, so an in-process test
would see state left behind by other tests. Each subprocess sets one
auto-trace variable to a truthy value, imports ``api.index`` (which pins the
environment), then either invokes a LangGraph StateGraph with a mocked
LangSmith client or checks ``tracing_is_enabled()``.

The mocked client is the wire boundary: any create_run/update_run/ingest call
is recorded with its payload. A violation is any call that carries non-empty
inputs or outputs, or any payload containing the sentinel prompt text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Every variable the langsmith or langchain_core SDKs read to decide that
# automatic tracing is on.
AUTO_TRACE_VARS = (
    "LANGSMITH_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_HANDLER",
)

# Variables where langsmith's own tracing_is_enabled() reads a truthy value.
# LANGCHAIN_TRACING and LANGCHAIN_HANDLER are v1 flags, not read by that check.
_V2_READ_VARS = ("LANGSMITH_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")

ENV_CASES = [
    {"LANGSMITH_TRACING_V2": "true"},
    {"LANGSMITH_TRACING": "true"},
    {"LANGSMITH_TRACING_V2": "true", "LANGSMITH_TRACING": "true"},
    {"LANGCHAIN_TRACING_V2": "true"},
    {"LANGCHAIN_TRACING": "true"},
    {"LANGCHAIN_HANDLER": "1"},
]


def _case_id(case: dict) -> str:
    return "+".join(sorted(case))


_CONTENT_SCRIPT = r'''
import json
from typing import TypedDict
from unittest.mock import patch

import api.index  # noqa: F401  (pins the LangSmith environment)
import langsmith
from langgraph.graph import END, StateGraph

SENTINEL = "PROMPT_SECRET"
violations = []
calls = []


def _has_content(payload):
    if isinstance(payload, dict):
        return bool(payload.get("inputs")) or bool(payload.get("outputs"))
    return False


def _recorder(method_name):
    def fake(self, *args, **kwargs):
        record = {"method": method_name, "args": repr(args), "kwargs": repr(kwargs)}
        calls.append(record)
        content = _has_content(kwargs) or any(_has_content(a) for a in args)
        if content or SENTINEL in record["args"] or SENTINEL in record["kwargs"]:
            violations.append(method_name)
        return None
    return fake


patches = []
for _name in ("create_run", "update_run", "multipart_ingest", "batch_ingest_runs"):
    if hasattr(langsmith.Client, _name):
        patches.append(patch.object(langsmith.Client, _name, _recorder(_name)))
for _p in patches:
    _p.start()


class State(TypedDict):
    q: str
    a: str


def answer(state):
    return {"a": "RESPONSE for " + state["q"]}


graph = StateGraph(State)
graph.add_node("answer", answer)
graph.set_entry_point("answer")
graph.add_edge("answer", END)
app = graph.compile()
app.invoke({"q": SENTINEL + " quarterly revenue by region", "a": ""})

try:
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
except ImportError:
    pass

print(json.dumps({"patched": len(patches), "calls": len(calls), "violations": violations}))
'''

_ENABLED_SCRIPT = r'''
import json
from langsmith.utils import tracing_is_enabled

before = bool(tracing_is_enabled())
import api.index  # noqa: F401  (pins the LangSmith environment)
after = bool(tracing_is_enabled())
print(json.dumps({"before": before, "after": after}))
'''


def _run(script: str, env_extra: dict) -> dict:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO),
        # api.index requires DATABASE_URL at import. No connection is opened.
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/d",
        "LANGSMITH_API_KEY": "dummy-not-real",
        "groq_api_key": "dummy-not-real",
    }
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=170, env=env, cwd=str(REPO),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", ENV_CASES, ids=_case_id)
def test_graph_invoke_ships_no_content_to_langsmith(case):
    result = _run(_CONTENT_SCRIPT, case)
    assert result["patched"] > 0, "precondition: the mocked client must be wired"
    assert result["violations"] == [], (
        f"content-bearing LangSmith calls under {case}: {result['violations']}"
    )


@pytest.mark.parametrize("var", AUTO_TRACE_VARS)
def test_tracing_disabled_after_importing_api_index(var):
    result = _run(_ENABLED_SCRIPT, {var: "true"})
    if var in _V2_READ_VARS:
        assert result["before"] is True, "precondition: the SDK must see tracing on"
    assert result["after"] is False, f"auto-tracing still on after import with {var}=true"


# ---------------------------------------------------------------------------
# In-process checks: warning text and the fail-closed path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"])
def test_override_warning_names_the_variable(monkeypatch, caplog, name):
    import logging

    import api.utils.langsmith_pin as pin

    monkeypatch.setattr(pin, "_user_values", None)
    monkeypatch.setattr(pin, "_force_disabled", False)
    monkeypatch.setenv(name, "true")
    with caplog.at_level(logging.WARNING, logger="api.utils.langsmith_pin"):
        pin.pin_langsmith_env()
    assert any(name in r.getMessage() for r in caplog.records)


def test_fail_closed_forces_docbot_sending_off(monkeypatch, caplog):
    """If langsmith still reports tracing on after the pin, DocBot's own
    sending path must be disabled, and the event is logged, not raised."""
    import logging
    from unittest.mock import patch

    import api.utils.langsmith_pin as pin
    import api.utils.langsmith_tracing as lst

    monkeypatch.setattr(pin, "_user_values", None)
    monkeypatch.setattr(pin, "_force_disabled", False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy-not-real")
    with patch("langsmith.utils.tracing_is_enabled", return_value=True):
        with caplog.at_level(logging.WARNING, logger="api.utils.langsmith_pin"):
            pin.pin_langsmith_env()
    assert pin.sending_forced_off() is True
    assert lst.load_config(pin.docbot_env()).enabled is False
    assert any("still enabled" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(pin, "_force_disabled", False)
