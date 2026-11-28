"""DOCBOT-2730: intent-router eval — gold-set consistency + regex baseline (CI),
live LLM route accuracy (external).

CI:    pytest tests/eval/test_intent_router_eval.py
Live:  pytest tests/eval/test_intent_router_eval.py -m external -s   (needs groq_api_key)
"""

from __future__ import annotations

import asyncio
import os
import typing
from collections import Counter

import pytest

import api.intent_service as svc
from api.intent_models import IntentTask
from tests.eval.intent_router_gold import GOLD_INTENTS, GoldIntent, score


def _regex(g: GoldIntent):
    return svc.regex_route(g.question, g.src[1], g.src[2])


def test_gold_set_is_well_formed() -> None:
    assert len(GOLD_INTENTS) == 60
    assert len({g.question for g in GOLD_INTENTS}) == 60
    tasks = set(typing.get_args(IntentTask))
    for g in GOLD_INTENTS:
        assert g.route in ("fast", "autopilot") and g.task in tasks
        assert g.data_ready or g.task in ("forecast", "what_if", "diagnose")
    counts = Counter(g.task for g in GOLD_INTENTS)
    assert counts["lookup"] >= 10 and counts["viz"] >= 6 and counts["forecast"] >= 14
    assert sum(not g.data_ready for g in GOLD_INTENTS) >= 5


def test_regex_baseline_misses_paraphrases() -> None:
    """The regex baseline is what the router must beat; pin its known blind spot."""
    base = score(_regex)
    print(f"\nregex baseline: {base}")
    assert base.route_accuracy < 0.9  # keyword-free forecasts/what-ifs go to the fast path
    assert base.data_ready_accuracy < 1.0  # it never flags missing data


@pytest.mark.external
@pytest.mark.skipif(not os.environ.get("groq_api_key"), reason="needs groq_api_key")
def test_llm_router_beats_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Measure model quality, not network jitter: the prod 1.5s timeout would
    # silently score regex fallbacks as LLM answers.
    monkeypatch.setattr(svc, "TIMEOUT_S", 15.0)
    svc._cache.clear()
    fallbacks = 0

    def predict(g: GoldIntent):
        nonlocal fallbacks
        r = asyncio.run(svc.classify(g.question, [], g.src[0], g.src[1], g.src[2]))
        fallbacks += r.confidence == 0.0
        return r

    llm = score(predict)
    base = score(_regex)
    print(f"\nLLM router: {llm}\nregex baseline: {base}\nregex fallbacks during run: {fallbacks}")
    assert llm.route_accuracy >= 0.90
    assert llm.false_autopilot_simple <= 0.05
