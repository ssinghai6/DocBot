"""DOCBOT-1802 — Gemini fallback runs its sync SDK call in a worker thread;
the billing ContextVars must follow it there so the call is attributed and
metered on the request's RequestMeter. No network: ``_generate`` is stubbed.
"""

from __future__ import annotations

import threading

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from api.utils._gemini_wrapper import GeminiChatModel
from api.utils.billing_context import (
    Principal,
    RequestMeter,
    bind_principal,
    get_meter,
    get_principal,
    metered_request,
)


async def test_agenerate_propagates_billing_context(monkeypatch):
    seen: dict = {}

    def fake_generate(self, messages, stop=None, **kwargs):
        seen["thread"] = threading.get_ident()
        seen["meter"] = get_meter()
        seen["principal"] = get_principal()
        get_meter().record_llm(10, 5, "0.0001")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(GeminiChatModel, "_generate", fake_generate)
    model = GeminiChatModel(api_key="test-not-a-real-key")
    principal = Principal(user_id="u1", org_id="org-1", auth_method="session")
    meter = RequestMeter()

    with bind_principal(principal), metered_request(meter):
        result = await model._agenerate([HumanMessage(content="hi")])

    assert result.generations[0].message.content == "ok"
    assert seen["thread"] != threading.get_ident()  # really ran off-loop
    assert seen["meter"] is meter
    assert seen["principal"] is principal
    snap = meter.snapshot()
    assert snap.llm_calls == 1 and snap.cost_micro_usd == 100
