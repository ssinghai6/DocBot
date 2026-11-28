"""Intent router — DOCBOT-2728.

One small LLM call decides how a chat turn should run:

* ``route``: ``fast`` (single pipeline) or ``autopilot`` (multi-step).
* ``task``: lookup / compare / forecast / what_if / diagnose / viz.
* ``data_ready`` + ``missing``: whether the attached sources hold the data the
  task needs (e.g. a forecast needs a historical numeric series).

The model sees a *source summary* the server builds itself: document file
names for the session and table / column names from the cached schema of the
connection (a CSV's schema is its column list). Both are read only after the
same ownership check the other routes use; credentials, connection strings
and row values never reach the prompt.

Any failure (timeout, provider error, malformed JSON, schema mismatch) returns
``regex_route()`` — a Python port of the frontend's old regex trigger — with
``confidence = 0`` so callers can tell the two apart.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Optional

from sqlalchemy import Table, select

from api.db_service import connection_access_clause
from api.intent_models import IntentSource, IntentTask, RouteIntentRequest, RouteIntentResponse
from api.utils.billing_context import copy_ctx_run
from api.utils.caller_scope import CallerScope
from api.utils.llm_provider import GROQ_MODEL, chat_completion

logger = logging.getLogger(__name__)

PROMPT_VERSION_INTENT_ROUTER = "v1"
TIMEOUT_S = 1.5
_CACHE_MAX = 512
_HISTORY_MESSAGES = 6  # last 3 turns (user + assistant), same window as the chat routes
_SUMMARY_MAX_CHARS = 2000

# ── Deterministic fallback: port of src/hooks/useChatSubmit.ts (pre-2729) ────
SIMPLE_VIZ = re.compile(
    r"^\s*(plot|show|graph|chart|draw|visualize|visualise|display|histogram|heatmap|render|what is|tell me)\b",
    re.I,
)
MULTI_STEP_TRIGGER = re.compile(
    r"\b(predict|forecast|what.?if|simulat|root.?cause|deep.?dive|break.?down|investigat|diagnos"
    r"|scenario|correlat|regress|cluster|anomal|outli)",
    re.I,
)
COMPLEX_COMPARE = re.compile(r"\bcompar\w*\s+.+\s+(vs|versus|against|with|to)\b", re.I)

_TASK_PATTERNS: list[tuple[IntentTask, re.Pattern[str]]] = [
    ("forecast", re.compile(r"\b(predict|forecast|regress)", re.I)),
    ("what_if", re.compile(r"\b(what.?if|simulat|scenario)", re.I)),
    ("diagnose", re.compile(r"\b(root.?cause|deep.?dive|break.?down|investigat|diagnos|anomal|outli|cluster)", re.I)),
    ("compare", re.compile(r"\b(compar|correlat)", re.I)),
]


def _source(has_doc: bool, has_db: bool) -> IntentSource:
    if has_doc and has_db:
        return "hybrid"
    return "sql" if has_db else "doc"


def regex_route(question: str, has_doc: bool, has_db: bool) -> RouteIntentResponse:
    """Today's frontend regex logic, ported. ``data_ready`` is True: no check is made."""
    autopilot = False
    task: IntentTask = "lookup"
    if SIMPLE_VIZ.search(question):
        task = "viz"
    elif MULTI_STEP_TRIGGER.search(question) or COMPLEX_COMPARE.search(question):
        autopilot = has_doc or has_db
        task = next((t for t, p in _TASK_PATTERNS if p.search(question)), "compare")
    return RouteIntentResponse(
        route="autopilot" if autopilot else "fast",
        source=_source(has_doc, has_db),
        task=task,
        data_ready=True,
        missing=None,
        confidence=0.0,
    )


# ── Source summary (names only, ownership-checked) ───────────────────────────


async def build_source_summary(
    session_id: Optional[str],
    connection_id: Optional[str],
    scope: CallerScope,
    sessions_table: Table,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory: Any,
) -> tuple[str, bool, bool]:
    """``(summary, has_doc, has_db)`` for the caller's own session / connection.

    A session or connection the caller cannot access is treated as absent.
    Only file names, the dialect and cached table/column names are read.
    """
    lines: list[str] = []
    has_doc = has_db = False
    async with async_session_factory() as session:
        if session_id:
            row = (
                await session.execute(
                    select(sessions_table.c.files_info, sessions_table.c.source).where(
                        sessions_table.c.session_id == session_id,
                        connection_access_clause(sessions_table, scope),
                    )
                )
            ).first()
            if row is not None:
                names = _file_names(row.files_info)
                has_doc = bool(names) or row.source == "edgar"
                if has_doc:
                    label = "SEC filing" if row.source == "edgar" else "Documents"
                    lines.append(f"{label}: " + (", ".join(names) or "(names unavailable)"))
        if connection_id:
            row = (
                await session.execute(
                    select(db_connections_table.c.dialect, schema_cache_table.c.schema_json)
                    .select_from(
                        db_connections_table.outerjoin(
                            schema_cache_table, schema_cache_table.c.connection_id == db_connections_table.c.id
                        )
                    )
                    .where(
                        db_connections_table.c.id == connection_id,
                        connection_access_clause(db_connections_table, scope),
                    )
                )
            ).first()
            if row is not None:
                has_db = True
                kind = "CSV file" if row.dialect == "csv" else f"{row.dialect} database"
                lines.append(f"{kind}: " + (_schema_names(row.schema_json) or "(schema not loaded yet)"))
    if not lines:
        lines.append("No sources attached.")
    return "\n".join(lines)[:_SUMMARY_MAX_CHARS], has_doc, has_db


def _file_names(files_info: Optional[str]) -> list[str]:
    try:
        items = json.loads(files_info or "[]")
    except json.JSONDecodeError:
        return []
    return [str(f.get("filename")) for f in items if isinstance(f, dict) and f.get("filename")][:20]


def _schema_names(schema_json: Optional[str]) -> str:
    try:
        tables = json.loads(schema_json or "[]")
    except json.JSONDecodeError:
        return ""
    parts = []
    for t in tables[:30] if isinstance(tables, list) else []:
        if not isinstance(t, dict):
            continue
        cols = [f"{c.get('name')} {c.get('type', '')}".strip() for c in (t.get("columns") or [])[:25] if isinstance(c, dict)]
        parts.append(f"{t.get('name')}({', '.join(cols)})")
    return "; ".join(parts)


# ── LLM call ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You route questions for a financial analysis assistant. Reply with ONE JSON object only, no prose:
{"route": "fast"|"autopilot", "source": "doc"|"sql"|"hybrid", "task": "lookup"|"compare"|"forecast"|"what_if"|"diagnose"|"viz", "data_ready": true|false, "missing": string|null, "confidence": 0..1}

task:
- lookup: fetch or explain a fact, number, definition or summary.
- viz: draw a chart/plot of existing data, nothing projected.
- compare: two or more entities/periods side by side, or a correlation.
- forecast: any future value, projection, trend extrapolation or prediction, even without the words "forecast"/"predict" (e.g. "where will X be in 2030", "next year's revenue").
- what_if: a hypothetical change of an input (e.g. "if CAC rises 20%", "suppose we cut prices").
- diagnose: explain WHY something changed, root cause, anomalies, drivers.

route: "autopilot" for forecast, what_if, diagnose, and for compare that needs several queries or sources. "fast" for lookup, viz and simple compare. A plain chart request is always "fast".
source: "doc" if only documents answer it, "sql" for the database/CSV, "hybrid" when both are needed. Pick from attached sources only.
data_ready: false ONLY when the task is forecast, what_if or diagnose AND the attached sources clearly lack what it needs (e.g. a forecast needs a historical numeric series: a table/CSV with a date or year column and the metric, or a financial filing / statement with multi-year figures). Documents with unrelated titles, or no sources at all, are not ready. Otherwise true.
missing: when data_ready is false, a short phrase naming what is missing (e.g. "no historical diesel price series in your sources"); else null.
confidence: your confidence in route and task."""


def _build_messages(question: str, history: list[dict[str, str]], summary: str) -> list[dict[str, str]]:
    convo = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in history) or "(none)"
    user = f"Attached sources:\n{summary}\n\nRecent conversation:\n{convo}\n\nQuestion: {question[:2000]}"
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user}]


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_llm_output(text: str) -> RouteIntentResponse:
    """Parse the model's reply. Raises ``ValueError`` / ``ValidationError`` on garbage."""
    match = _JSON_OBJECT.search(text or "")
    if not match:
        raise ValueError("no JSON object in router output")
    result = RouteIntentResponse.model_validate_json(match.group(0))
    if result.data_ready:
        result.missing = None
    return result


def _call_llm(messages: list[dict[str, str]]) -> str:
    return chat_completion(
        messages,
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=300,
        caller="intent_router",
        prompt_version=PROMPT_VERSION_INTENT_ROUTER,
    )


# shortcut: per-process LRU; with several workers each keeps its own copy, fine for a cache.
_cache: "OrderedDict[str, RouteIntentResponse]" = OrderedDict()


def _cache_key(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_put(key: str, value: RouteIntentResponse) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


async def classify(question: str, history: list[dict[str, str]], summary: str, has_doc: bool, has_db: bool) -> RouteIntentResponse:
    """LLM route for ``question``, or ``regex_route`` on any failure."""
    messages = _build_messages(question, history[-_HISTORY_MESSAGES:], summary)
    key = _cache_key(messages)
    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        return cached.model_copy()
    loop = asyncio.get_running_loop()
    try:
        text = await asyncio.wait_for(loop.run_in_executor(None, copy_ctx_run(_call_llm, messages)), TIMEOUT_S)
        result = parse_llm_output(text)
    # Any failure (timeout, provider SDK errors of many types, bad JSON) must
    # degrade to the regex route, never fail the chat turn.
    except Exception as exc:  # noqa: BLE001
        logger.info("intent_router: fallback to regex (%s)", type(exc).__name__)
        return regex_route(question, has_doc, has_db)
    if not (has_doc or has_db):
        result.route = "fast"  # autopilot needs a source; nothing to investigate
    _cache_put(key, result)
    return result.model_copy()


async def route_intent(
    req: RouteIntentRequest,
    scope: CallerScope,
    *,
    sessions_table: Table,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory: Any,
) -> RouteIntentResponse:
    summary, has_doc, has_db = await build_source_summary(
        req.session_id, req.connection_id, scope,
        sessions_table, db_connections_table, schema_cache_table, async_session_factory,
    )
    history = [{"role": t.role, "content": t.content} for t in req.history]
    return await classify(req.question, history, summary, has_doc, has_db)
