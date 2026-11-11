"""DOCBOT-2541: tie-out accuracy eval -- score hybrid Docs+DB answers against
GOLD_TIEOUT, N runs per question, pass rate per question.

A run is one hybrid answer reduced to a ``RunRecord`` (the same fields the
``lineage`` SSE event carries). ``score_run`` is pure code; see
tests/eval/README.md section 5 for the method.

Modes
  live     in-process ``hybrid_chat`` against the real demo data with real
           Groq/Gemini calls (needs groq_api_key). The exact-match LLM cache is
           not wired in-process, so every run is an independent LLM sample.
  offline  no API keys: real retrieval (local ONNX embeddings), the gold
           ``books_sql`` validated + executed on the demo SQLite, the real
           discrepancy detector, the real scorer. The three LLM stages (intent,
           SQL generation, synthesis) are replaced by the gold route/query, so
           an offline pass rate is NOT a live accuracy number.

Run:
    python -m tests.eval.tieout_eval --mode offline --runs 10
    python -m tests.eval.tieout_eval --mode live --runs 10            # needs groq_api_key
    python -m tests.eval.tieout_eval --mode live --runs 10 --only segment_note12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from api.utils.discrepancy_detector import _label_tokens
from tests.eval.gold_tieout_set import GOLD_TIEOUT, PLANTED, GoldFigure, TieoutQuestion

# A figure "ties" to a value when they agree within 0.1% relative, after
# snapping a power-of-1000 unit gap (books in $M vs report in $). The demo's
# planted gaps are 0.55%-2.1%, so 0.1% can never mistake a gap for a tie.
TOLERANCE = 0.001


@dataclass
class RunRecord:
    intent: str | None = None
    answer: str = ""
    discrepancies: list[dict[str, Any]] = field(default_factory=list)  # lineage.discrepancies[]
    sql_rows: list[dict[str, Any]] = field(default_factory=list)        # lineage.sql.result_preview
    doc_pages: set[int] = field(default_factory=set)                     # pages of retrieved/cited chunks
    error: str | None = None


def close(a: float | None, b: float | None, tol: float = TOLERANCE) -> bool:
    if a is None or b is None:
        return False
    a, b = abs(float(a)), abs(float(b))
    if a == 0 or b == 0:
        return a == b
    hi, lo = max(a, b), min(a, b)
    lo *= 1000 ** max(0, round(math.log(hi / lo, 1000)))
    return abs(hi - lo) / hi <= tol


def _same_metric(metric: str, label: str) -> bool:
    return _label_tokens(metric) == _label_tokens(label)


def _rows_contain(rows: list[dict[str, Any]], value: float) -> bool:
    for row in rows:
        for v in row.values():
            try:
                if close(float(str(v).replace(",", "").replace("$", "")), value):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def score_run(q: TieoutQuestion, run: RunRecord) -> tuple[bool, list[str]]:
    """(passed, reasons it failed). A run passes only when every check holds."""
    if run.error:
        return False, [f"error: {run.error}"]
    fails: list[str] = []
    planted = PLANTED[q.dataset]
    for d in run.discrepancies:
        if not any(_same_metric(d.get("metric", ""), p) for p in planted):
            fails.append(f"false positive: {d.get('metric')!r}")
    for fig in q.figures:
        fails.extend(_check_figure(fig, run))
    return not fails, fails


def _check_figure(fig: GoldFigure, run: RunRecord) -> list[str]:
    hits = [
        d for d in run.discrepancies
        if _same_metric(d.get("metric", ""), fig.label)
        and (fig.period is None or d.get("period") in (None, fig.period))
    ]
    if not fig.flag:
        out = [f"{fig.label}: flagged but ties out"] if hits else []
        if not _rows_contain(run.sql_rows, fig.books):
            out.append(f"{fig.label}: books value {fig.books:,.0f} not in SQL result")
        if fig.page not in run.doc_pages:
            out.append(f"{fig.label}: doc page {fig.page} not cited")
        return out
    if len(hits) != 1:
        return [f"{fig.label}: expected 1 discrepancy, got {len(hits)}"]
    d = hits[0]
    out = []
    if not (close(d.get("report_value"), fig.report) and close(d.get("books_value"), fig.books)):
        out.append(
            f"{fig.label}: values {d.get('report_value')} vs {d.get('books_value')}, "
            f"want {fig.report:,.0f} vs {fig.books:,.0f}"
        )
    doc = d.get("doc_citation") or {}
    db = d.get("db_citation") or {}
    if not doc.get("file") or doc.get("page") != fig.page:
        out.append(f"{fig.label}: doc citation {doc} != page {fig.page}")
    if not (db.get("table") or db.get("query")):
        out.append(f"{fig.label}: no db citation")
    return out


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _sqlite_for(dataset: str):
    from api import demo_service as d

    return d._create_tieout_sqlite() if dataset == "tieout" else d._create_demo_sqlite()


def _chunks_for(dataset: str):
    from api import demo_service as d

    return d.TIEOUT_DOCUMENT_CHUNKS if dataset == "tieout" else d.DEMO_DOCUMENT_CHUNKS


async def offline_run(q: TieoutQuestion, store: Any, db_path: str) -> RunRecord:
    """Deterministic stages only (see module docstring)."""
    from api.hybrid_service import rag_retrieve
    from api.utils.discrepancy_detector import detect_discrepancies
    from api.utils.lineage import LineageCollector
    from api.utils.sql_validator import validate_and_sanitize_sql
    from sqlglot import exp, parse_one

    lineage = LineageCollector("offline", "hybrid", question=q.question)
    doc_context, citations = await rag_retrieve(q.question, "s", {"s": store}, lineage=lineage)
    sql = validate_and_sanitize_sql(q.books_sql, "sqlite")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()
    tables = sorted({t.name for t in parse_one(sql, read="sqlite").find_all(exp.Table)})
    meta = {"result_preview": rows, "sources": tables, "sql_query": sql}
    report = detect_discrepancies(doc_context, meta)
    for item in report.discrepancies:
        lineage.add_discrepancy(item)
    built = lineage.build().model_dump(mode="json")
    return RunRecord(
        intent="hybrid",
        discrepancies=built["discrepancies"],
        sql_rows=rows,
        doc_pages={int(c["page"]) for c in citations},
    )


async def _live_env():
    """In-process app DB (aiosqlite) with every table hybrid_chat touches."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    # api.index refuses to import without DATABASE_URL; its engine is lazy and
    # never used here (same fake URL as CI).
    os.environ.setdefault("DATABASE_URL", "postgresql://eval:eval@localhost:5432/unused")
    import api.index as idx

    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(idx.metadata.create_all)
    return idx, async_sessionmaker(engine, expire_on_commit=False)


async def live_run(q: TieoutQuestion, idx: Any, factory: Any, demo: dict, stores: dict) -> RunRecord:
    from api.hybrid_service import hybrid_chat

    run = RunRecord()
    try:
        async for chunk in hybrid_chat(
            question=q.question,
            session_id=demo["session_id"],
            connection_id=demo["connection_id"],
            persona="Finance Expert",
            has_docs=True,
            messages_table=idx.messages_table,
            sessions_table=idx.sessions_table,
            db_connections_table=idx.db_connections_table,
            schema_cache_table=idx.schema_cache_table,
            query_history_table=idx.query_history_table,
            query_embeddings_table=idx.query_embeddings_table,
            async_session_factory=factory,
            expert_personas=idx.EXPERT_PERSONAS,
            vector_stores=stores,
        ):
            if not chunk.startswith("data: "):
                continue
            evt = json.loads(chunk[6:])
            kind = evt.get("type")
            if kind == "metadata" and evt.get("intent"):
                run.intent = evt["intent"]
            elif kind == "token":
                run.answer += evt.get("content", "")
            elif kind == "lineage":
                run.discrepancies = evt.get("discrepancies") or []
                run.sql_rows = (evt.get("sql") or {}).get("result_preview") or []
                run.doc_pages |= {int(s["page"]) for s in evt.get("sources") or [] if s.get("page") is not None}
            elif kind == "done":
                run.doc_pages |= {int(c["page"]) for c in evt.get("citations") or [] if c.get("page") is not None}
    except Exception as exc:  # noqa: BLE001 -- one failed run is a failed sample, not a crashed eval
        run.error = f"{type(exc).__name__}: {exc}"
    return run


async def run_eval(mode: str, runs: int, only: set[str] | None = None) -> dict[str, list[tuple[bool, list[str]]]]:
    from api.utils.embeddings_provider import get_embeddings
    from api.utils.vector_store import create_store

    questions = [q for q in GOLD_TIEOUT if not only or q.id in only]
    results: dict[str, list[tuple[bool, list[str]]]] = {q.id: [] for q in questions}
    if mode == "offline":
        for dataset in sorted({q.dataset for q in questions}):
            store = create_store(f"eval-{dataset}-{uuid.uuid4().hex[:8]}", _chunks_for(dataset), get_embeddings())
            path = _sqlite_for(dataset)
            try:
                for q in (q for q in questions if q.dataset == dataset):
                    for _ in range(runs):
                        results[q.id].append(score_run(q, await offline_run(q, store, str(path))))
            finally:
                path.unlink(missing_ok=True)
        return results

    from api.demo_service import init_demo_session

    idx, factory = await _live_env()
    stores: dict = {}
    for q in questions:
        for _ in range(runs):
            # Fresh demo session per run: no chat history, no few-shot carry-over.
            demo = await init_demo_session(
                vector_stores=stores,
                get_embeddings_fn=get_embeddings,
                sessions_table=idx.sessions_table,
                db_connections_table=idx.db_connections_table,
                schema_cache_table=idx.schema_cache_table,
                async_session_factory=factory,
                dataset=q.dataset,
                owner={"user_id": None, "org_id": None, "anon_owner": "eval"},
            )
            results[q.id].append(score_run(q, await live_run(q, idx, factory, demo, stores)))
    return results


def report(results: dict[str, list[tuple[bool, list[str]]]], mode: str) -> str:
    lines = [f"=== Tie-out accuracy eval (mode={mode}, tolerance={TOLERANCE:.1%}) ==="]
    total = passed = 0
    for qid, outcomes in results.items():
        ok = sum(p for p, _ in outcomes)
        total += len(outcomes)
        passed += ok
        lines.append(f"  {ok:>3}/{len(outcomes):<3} {qid}")
        first_fail = next((r for p, r in outcomes if not p), None)
        if first_fail:
            lines.append(f"          first failure: {'; '.join(first_fail)[:300]}")
    lines.append(f"  overall {passed}/{total} runs passed ({passed / total:.0%})" if total else "  no runs")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCBOT-2541 tie-out accuracy eval")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--only", nargs="*", help="question ids from gold_tieout_set.py")
    args = parser.parse_args()
    results = asyncio.run(run_eval(args.mode, args.runs, set(args.only or []) or None))
    print(report(results, args.mode))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
