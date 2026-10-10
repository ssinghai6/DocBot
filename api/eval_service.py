"""LLM-judge offline eval harness for agent traces — DOCBOT-1519.

DOCBOT-1512 (merged) added ``api/trace_service.py`` + an ``agent_traces``
table that logs ``{question, intent_classified, tool_chosen, plan_steps_json,
retrieved_refs_json, final_answer, latency_ms, cost_usd, error,
user_feedback}`` for every autopilot/hybrid/chat run. Nothing evaluated those
traces until now. This module is a read-only offline consumer of that table:
it samples recent traces, asks an LLM judge whether the tool/intent choice
was plausible and whether the final answer looks supported by the retrieved
evidence, and persists the verdicts to a new ``eval_judgments`` table for
later reference (the eventual Tier 3 fine-tuning data labeling work reads
from exactly this table).

Scope is strictly read-judge-report: no training, no model changes, and no
changes to any live pipeline (autopilot/hybrid/chat routes, SQL validation,
auth, the tools registry are all untouched).

Public API
----------
register_eval_judgments_table(metadata) -> Table
wire_eval_store(table, async_session_factory) -> None
sample_recent_traces(pipeline=None, limit=50, since_hours=168) -> list[dict]
judge_trace(trace) -> dict
    Never raises — falls back to "uncertain" verdicts with a
    "judge call failed: <error>" reason on any failure.
run_eval_batch(pipeline=None, limit=50) -> dict
    Orchestrates sample + bounded-concurrency judge + persist + summarize.

Design notes
------------
This mirrors the ``register_agent_traces_table`` / ``wire_trace_store``
pattern in ``api/trace_service.py`` (itself modeled on
``api/lineage_service.py`` / ``api/llm_trace_service.py``): a table defined
once at import time from ``api/index.py``, wired with a live table +
session factory from the FastAPI lifespan, with persistence writes wrapped
in try/except so a DB hiccup never breaks a judge run.

The judge call goes through ``api/utils/llm_provider.py``'s ``call_llm``
(Groq primary, Gemini 2.5 Flash fallback) — the same general-purpose model
every other prod callsite uses (SQL gen, hybrid synthesis, intent
classification). There is no existing "strong model as judge" convention in
this codebase to follow (document_extractor.py's Gemini usage is a direct
LangExtract integration for financial extraction, not a role precedent), so
this defaults to the general fallback-wrapped model as the ticket specifies.

Concurrency for a batch of judge calls is bounded via an ``asyncio.Semaphore``
(default 5 concurrent judge calls), mirroring
``deep_research_service.py``'s ``ThreadPoolExecutor(max_workers=min(len(expanded), 6))``
pattern for bounding concurrent LLM-backed work — a ``limit=50`` batch does
not fire 50 unbounded parallel calls at Groq/Gemini.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Table, Text, func, select
from sqlalchemy import insert as sa_insert

logger = logging.getLogger(__name__)

VALID_TOOL_SELECTION_VERDICTS = {"correct", "incorrect", "uncertain"}
VALID_EVIDENCE_SUPPORT_VERDICTS = {"supported", "unsupported", "uncertain"}

# DOCBOT-1507 prompt versioning convention — bump manually if the judge
# prompt's wording/instructions change in a way that could shift scores.
PROMPT_VERSION_EVAL_JUDGE = "v1"

JUDGE_CALLER = "eval_judge"

# Bounded concurrency for a batch of judge calls — see module docstring.
DEFAULT_JUDGE_CONCURRENCY = 5

# Length caps on reasons persisted to the DB — these are judge verdicts, not
# a second copy of the full trace.
_MAX_REASON_LEN = 500


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------


def register_eval_judgments_table(metadata: Any) -> Table:
    """Define the eval_judgments table on shared metadata.

    Called once at import time from ``api/index.py``, mirroring
    ``register_agent_traces_table``.
    """
    return Table(
        "eval_judgments",
        metadata,
        Column("id", String, primary_key=True),  # UUID, minted here
        Column("trace_id", String, nullable=False, index=True),
        Column("tool_selection_verdict", String, nullable=False),
        Column("tool_selection_reason", Text),
        Column("evidence_support_verdict", String, nullable=False),
        Column("evidence_support_reason", Text),
        Column(
            "judged_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            index=True,
        ),
    )


# ---------------------------------------------------------------------------
# Module-level wiring
# ---------------------------------------------------------------------------

_eval_judgments_table: Optional[Table] = None
_async_session_factory: Any = None


def wire_eval_store(table: Table, async_session_factory: Any) -> None:
    """Inject table + session factory references. Called once from index.py lifespan."""
    global _eval_judgments_table, _async_session_factory
    _eval_judgments_table = table
    _async_session_factory = async_session_factory


# ---------------------------------------------------------------------------
# Sampling — read-only query against agent_traces
# ---------------------------------------------------------------------------


async def sample_recent_traces(
    pipeline: Optional[str] = None,
    limit: int = 50,
    since_hours: int = 168,
) -> list[dict]:
    """Return up to `limit` most recent agent_traces rows within the last
    `since_hours`, optionally filtered by `pipeline`, newest first.

    Never raises. Returns [] if the agent_traces store isn't wired (no DB
    configured) or the query fails for any reason — an eval run with no
    data to judge is a valid outcome, not an error.
    """
    from api.trace_service import get_agent_traces_table

    table = get_agent_traces_table()
    session_factory = _async_session_factory
    if table is None or session_factory is None:
        logger.warning("eval_service: agent_traces store not wired, returning no traces")
        return []

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        query = select(table).where(table.c.created_at >= cutoff)
        if pipeline:
            query = query.where(table.c.pipeline == pipeline)
        query = query.order_by(table.c.created_at.desc()).limit(limit)

        async with session_factory() as session:
            result = await session.execute(query)
            rows = result.mappings().all()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("eval_service: sample_recent_traces failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Judging — one LLM call per trace, never raises
# ---------------------------------------------------------------------------


def _build_judge_prompt(trace: dict) -> str:
    question = trace.get("question") or ""
    intent = trace.get("intent_classified") or "unknown"
    tool = trace.get("tool_chosen") or "unknown"
    answer = (trace.get("final_answer") or "")[:3000]

    refs_raw = trace.get("retrieved_refs_json")
    try:
        refs = json.loads(refs_raw) if isinstance(refs_raw, str) else (refs_raw or [])
    except (json.JSONDecodeError, TypeError):
        refs = []
    refs_text = json.dumps(refs, default=str)[:3000]

    return f"""You are an offline QA judge reviewing one turn of an AI agent's work. \
You are given the question, the tool/intent the agent chose, the evidence it \
retrieved, and the final answer it gave. Score two things:

1. tool_selection: was choosing tool "{tool}" (classified intent "{intent}") a \
plausible choice given the question? One of: correct, incorrect, uncertain.
2. evidence_support: does the final answer appear supported by the retrieved \
evidence (not fabricated or unsupported by it)? One of: supported, unsupported, \
uncertain.

Question: {question}

Retrieved evidence (citations/sources the agent used):
{refs_text}

Final answer given to the user:
{answer}

Respond with ONLY a single JSON object, no markdown code fences, no extra text:
{{"tool_selection_verdict": "...", "tool_selection_reason": "<one sentence>", \
"evidence_support_verdict": "...", "evidence_support_reason": "<one sentence>"}}
"""


def _parse_judge_response(raw: str) -> dict:
    """Parse the judge's JSON response, clamping any out-of-vocabulary verdict
    to "uncertain" rather than raising. Raises on unparseable JSON — the
    caller (judge_trace) catches that and falls back to a full-uncertain
    verdict."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)

    tool_verdict = str(data.get("tool_selection_verdict", "uncertain")).strip().lower()
    if tool_verdict not in VALID_TOOL_SELECTION_VERDICTS:
        tool_verdict = "uncertain"

    evidence_verdict = str(data.get("evidence_support_verdict", "uncertain")).strip().lower()
    if evidence_verdict not in VALID_EVIDENCE_SUPPORT_VERDICTS:
        evidence_verdict = "uncertain"

    return {
        "tool_selection_verdict": tool_verdict,
        "tool_selection_reason": str(data.get("tool_selection_reason", ""))[:_MAX_REASON_LEN],
        "evidence_support_verdict": evidence_verdict,
        "evidence_support_reason": str(data.get("evidence_support_reason", ""))[:_MAX_REASON_LEN],
    }


async def judge_trace(trace: dict) -> dict:
    """Make one LLM call judging a single agent trace.

    Returns {trace_id, tool_selection_verdict, tool_selection_reason,
    evidence_support_verdict, evidence_support_reason}. Never raises — any
    failure (LLM call error, malformed/unparseable response) falls back to
    an "uncertain" verdict on both axes with reason "judge call failed: <error>".
    """
    trace_id = trace.get("id") or trace.get("trace_id") or "unknown"
    try:
        from api.utils.llm_provider import call_llm

        prompt = _build_judge_prompt(trace)
        raw = await call_llm(
            prompt,
            caller=JUDGE_CALLER,
            prompt_version=PROMPT_VERSION_EVAL_JUDGE,
        )
        parsed = _parse_judge_response(raw)
        return {"trace_id": trace_id, **parsed}
    except Exception as exc:  # judging must never break an eval batch
        logger.warning("eval_service: judge_trace failed for trace %s: %s", trace_id, exc)
        reason = f"judge call failed: {exc}"
        return {
            "trace_id": trace_id,
            "tool_selection_verdict": "uncertain",
            "tool_selection_reason": reason,
            "evidence_support_verdict": "uncertain",
            "evidence_support_reason": reason,
        }


async def _persist_judgment(judgment: dict) -> None:
    """Fire-and-persist write of one judgment row. Never raises."""
    if _eval_judgments_table is None or _async_session_factory is None:
        return
    try:
        async with _async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_insert(_eval_judgments_table).values(
                        id=str(uuid.uuid4()),
                        trace_id=judgment["trace_id"],
                        tool_selection_verdict=judgment["tool_selection_verdict"],
                        tool_selection_reason=judgment["tool_selection_reason"],
                        evidence_support_verdict=judgment["evidence_support_verdict"],
                        evidence_support_reason=judgment["evidence_support_reason"],
                    )
                )
    except Exception as exc:  # persistence must never break an eval batch
        logger.warning(
            "eval_service: failed to persist judgment for trace %s: %s",
            judgment.get("trace_id"), exc,
        )


# ---------------------------------------------------------------------------
# Orchestration — sample + bounded-concurrency judge + persist + summarize
# ---------------------------------------------------------------------------


async def run_eval_batch(pipeline: Optional[str] = None, limit: int = 50) -> dict:
    """Sample recent traces, judge each one (bounded concurrency), persist
    every judgment, and return an aggregate summary.

    Returns {pipeline, sampled, tool_selection_counts, evidence_support_counts,
    flagged, estimated_cost_usd, judgments}. `flagged` lists every judgment
    that was anything other than a clean correct/supported pair — the
    disagreement / low-confidence traces worth a human look.
    """
    from api.utils.llm_provider import get_session_cost_usd, new_run_id, run_trace

    traces = await sample_recent_traces(pipeline=pipeline, limit=limit)
    if not traces:
        return {
            "pipeline": pipeline,
            "sampled": 0,
            "tool_selection_counts": {},
            "evidence_support_counts": {},
            "flagged": [],
            "estimated_cost_usd": 0.0,
            "judgments": [],
        }

    # Every judge call in this batch shares one run_id purely so their
    # estimated costs accumulate under one key in llm_provider's per-run_id
    # cost ledger (see get_session_cost_usd) — this batch is not itself a
    # multi-step "investigation" in the Autopilot/Deep Research sense.
    batch_run_id = new_run_id()
    semaphore = asyncio.Semaphore(DEFAULT_JUDGE_CONCURRENCY)

    async def _bounded_judge(trace: dict) -> dict:
        async with semaphore:
            with run_trace(batch_run_id):
                return await judge_trace(trace)

    judgments = await asyncio.gather(*[_bounded_judge(t) for t in traces])

    # Persist every judgment. A persistence failure for one row must not
    # drop the others — gather with return_exceptions, and _persist_judgment
    # itself never raises anyway (belt and suspenders).
    await asyncio.gather(*[_persist_judgment(j) for j in judgments], return_exceptions=True)

    tool_selection_counts: dict[str, int] = {}
    evidence_support_counts: dict[str, int] = {}
    flagged: list[dict] = []
    for judgment in judgments:
        tool_selection_counts[judgment["tool_selection_verdict"]] = (
            tool_selection_counts.get(judgment["tool_selection_verdict"], 0) + 1
        )
        evidence_support_counts[judgment["evidence_support_verdict"]] = (
            evidence_support_counts.get(judgment["evidence_support_verdict"], 0) + 1
        )
        if (
            judgment["tool_selection_verdict"] != "correct"
            or judgment["evidence_support_verdict"] != "supported"
        ):
            flagged.append(judgment)

    return {
        "pipeline": pipeline,
        "sampled": len(traces),
        "tool_selection_counts": tool_selection_counts,
        "evidence_support_counts": evidence_support_counts,
        "flagged": flagged,
        "estimated_cost_usd": round(get_session_cost_usd(batch_run_id), 6),
        "judgments": judgments,
    }
