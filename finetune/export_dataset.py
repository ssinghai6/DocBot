"""Export a QLoRA SFT dataset for the Autopilot tool router — DOCBOT-1523.

What this distills
-------------------
``api/autopilot_service.py``'s ``_select_tool_llm`` (DOCBOT-1513) makes one
Groq call per investigation step to pick a tool (``sql_query`` / ``doc_search``
/ ``python_analysis`` / ``unsupported``). ``api/trace_service.py``
(DOCBOT-1512) logs every autopilot run's aggregate routing decision and final
answer to ``agent_traces``. ``api/eval_service.py`` (DOCBOT-1519) judges those
traces with an LLM and persists verdicts to ``eval_judgments``.

This script joins the two tables and emits hindsight-relabeled SFT examples:

  * ``tool_selection_verdict == "correct"``  -> positive example, label =
    the tool(s) the live router actually chose.
  * ``tool_selection_verdict == "incorrect"`` -> the judge's free-text
    ``tool_selection_reason`` is scanned for a mention of a *different*
    valid tool name. If exactly one is found, that becomes the corrected
    label (hindsight relabeling). If none/multiple are found, the row is
    skipped and logged — we do not guess.
  * ``tool_selection_verdict in ("uncertain", None)`` -> skipped (ungraded
    or judge call failed).

Known limitations (read before trusting this at scale)
--------------------------------------------------------
``agent_traces`` logs one row per *entire autopilot run*, not one row per
*step* — but ``_select_tool_llm`` is a per-step decision. The trace schema
(DOCBOT-1512) does not carry per-step text, nor the ``has_db``/``has_docs``/
``has_csv`` flags or persona hint that were live at call time. This script
therefore reconstructs an *approximation* of the router's input, not the
exact prompt it saw:

  * ``question`` is used as a stand-in for the per-step ``step`` text
    (the real router sees a decomposed sub-step, not the user's whole
    question).
  * ``has_db`` / ``has_docs`` / ``has_csv`` are inferred from which tools
    appear anywhere in the run's ``tool_chosen`` (a comma-joined list of
    every tool used across all steps) — e.g. ``sql_query`` present implies
    ``has_db=True``. ``has_csv`` is only inferred True when
    ``python_analysis`` is the *only* tool used in the whole run (the
    strongest signal of a CSV-only session) — this is a heuristic, not a
    stored fact.
  * Persona tool-preference is NOT stored on the trace and is always
    omitted from the reconstructed prompt.
  * A run with multiple distinct tools in ``tool_chosen`` emits one
    example per distinct tool (same reconstructed instruction, different
    label) rather than guessing which step produced which tool. This is a
    known source of label noise for multi-tool runs; it is accepted here
    because the alternative (dropping all multi-tool runs) would throw
    away most of the signal in a small seed dataset. A future
    improvement — logging the per-step tool decision directly in
    ``trace_service`` — would remove this approximation entirely; see
    ``finetune/README.md``.

Given these caveats, this pipeline is built to be correct and reusable as
*more* real per-step-faithful data accumulates, not tuned to extract maximum
signal from today's 9-trace seed batch.

Output format
--------------
One JSON object per line, in the standard chat-messages shape TRL's
``SFTTrainer`` consumes natively via ``apply_chat_template``:

    {"messages": [
        {"role": "system", "content": "<router system prompt>"},
        {"role": "user", "content": "Step: <question>"},
        {"role": "assistant", "content": "{\"tool\": \"sql_query\"}"}
    ]}

The assistant turn is the exact JSON shape ``_select_tool_llm`` itself emits
and parses (see its docstring / ``json.loads(raw[start:end])``), so a model
fine-tuned on this dataset is a drop-in replacement for that call site's
output contract.

CLI
---
    python -m finetune.export_dataset --min-examples 1 \
        --output finetune/data/routing_sft.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("finetune.export_dataset")

# Below this many examples, a production fine-tune is not meaningful — this
# is a warning threshold, not a hard failure (re-runnable as traces accrue).
SMALL_DATASET_WARNING_THRESHOLD = 50

# DOCBOT-1513's valid tool vocabulary — imported read-only from
# api/autopilot_service.py (never modified by this script).
_VALID_TOOLS = {"sql_query", "doc_search", "python_analysis", "unsupported"}

# PROMPT_VERSION_AUTOPILOT_TOOL_ROUTER = "v1" in api/autopilot_service.py as
# of this writing. This system prompt is a deliberate DUPLICATE of the
# prompt-building logic inlined in `_select_tool_llm` (that function is a
# read-only reference per DOCBOT-1523's scope — it is not refactored to
# expose a shared helper). If that prompt's wording changes, update this
# duplicate too so training data matches what the live router actually sees.
_ROUTER_PROMPT_VERSION_REFERENCE = "v1"


def _build_router_system_prompt(has_db: bool, has_docs: bool, has_csv: bool) -> str:
    """Rebuild (a duplicate of) `_select_tool_llm`'s system prompt.

    Mirrors api/autopilot_service.py::_select_tool_llm as of
    PROMPT_VERSION_AUTOPILOT_TOOL_ROUTER = "v1". Persona tool-preference is
    omitted — it is not stored on agent_traces (see module docstring).
    """
    available: list[str] = []
    if has_db and not has_csv:
        available.append("sql_query — run a SQL query against the connected database")
    if has_docs:
        available.append("doc_search — search uploaded PDF documents")
    if has_csv or has_db or has_docs:
        available.append(
            "python_analysis — run Python/pandas (charts, forecasting, stats) on data "
            "already fetched by a prior step, or directly on an uploaded CSV"
        )
    if not available:
        available.append("python_analysis — run Python/pandas analysis")

    tools_str = "\n".join(f"  - {a}" for a in available)

    return (
        "You are a tool router for a multi-step AI investigation agent. Given ONE "
        "investigation step, choose exactly one tool to execute it, from:\n"
        f"{tools_str}\n\n"
        "RULES:\n"
        "- A step that fetches/queries/retrieves/counts/aggregates data should use "
        "sql_query (when a database is available) even if it also mentions a chart — "
        "the chart itself is a later step.\n"
        "- Use python_analysis only for a step that itself asks to visualise, plot, "
        "forecast, model, or compute on data a prior step already fetched (or any step "
        "at all when the only source is an uploaded CSV).\n"
        "- Use doc_search for steps about documents, reports, PDFs, contracts, filings, "
        "or policies.\n\n"
        'Respond with ONLY a JSON object, e.g. {"tool": "sql_query"}. No explanation, '
        "no markdown fences."
    )


def _infer_source_flags(tools_used: list[str]) -> tuple[bool, bool, bool]:
    """Infer (has_db, has_docs, has_csv) from the set of tools used in a run.

    Heuristic, not a stored fact — see module docstring's "Known
    limitations" section.
    """
    tool_set = set(tools_used)
    has_db = "sql_query" in tool_set
    has_docs = "doc_search" in tool_set
    # Only infer CSV-only when python_analysis is the sole tool used in the
    # entire run — the strongest available signal of a CSV-only session.
    has_csv = tool_set == {"python_analysis"}
    return has_db, has_docs, has_csv


def _extract_corrected_tool(reason: str, wrong_tool: Optional[str]) -> Optional[str]:
    """Keyword heuristic: find a single valid tool name (other than
    ``wrong_tool``) mentioned in the judge's free-text reason.

    Design choice: a keyword scan over the closed 4-item tool vocabulary is
    simple, deterministic, free, and auditable — an LLM call to "ask the
    judge what it meant" would cost a request per incorrect row and add a
    second source of hallucination risk for a vocabulary this small. If a
    future dataset shows judges phrasing corrections in ways this scan
    misses often, an LLM-based extraction call is the natural upgrade (see
    finetune/README.md).

    Returns the corrected tool name if exactly one *other* valid tool is
    mentioned, else None (no confident correction derivable).
    """
    text = (reason or "").lower()
    mentioned = {tool for tool in _VALID_TOOLS if tool in text and tool != wrong_tool}
    if len(mentioned) == 1:
        return next(iter(mentioned))
    return None


def _parse_tools_used(tool_chosen_raw: Optional[str]) -> list[str]:
    """Parse the comma-joined `tool_chosen` column into a deduped, ordered
    list of valid tool names. Unknown tokens are dropped (defensive — the
    column is free text set by log_trace callers, not an enum column)."""
    if not tool_chosen_raw:
        return []
    seen: list[str] = []
    for token in tool_chosen_raw.split(","):
        name = token.strip()
        if name in _VALID_TOOLS and name not in seen:
            seen.append(name)
    return seen


def build_examples_for_trace(trace: dict, judgment: Optional[dict]) -> tuple[list[dict], list[str]]:
    """Return (sft_examples, skip_reasons) for one agent_traces row.

    Pure function — no DB/network access — so it's directly unit-testable.
    """
    trace_id = trace.get("id", "unknown")
    question = (trace.get("question") or "").strip()
    tools_used = _parse_tools_used(trace.get("tool_chosen"))

    if judgment is None:
        return [], [f"trace {trace_id}: no judgment row (ungraded) — skipped"]

    verdict = judgment.get("tool_selection_verdict")

    if verdict not in ("correct", "incorrect"):
        return [], [f"trace {trace_id}: verdict={verdict!r} (uncertain/null) — skipped"]

    if not question:
        return [], [f"trace {trace_id}: empty question — skipped"]

    if verdict == "correct":
        if not tools_used:
            return [], [f"trace {trace_id}: verdict=correct but no parseable tool_chosen — skipped"]
        labels = tools_used
        reason_note = "correct verdict"
    else:  # incorrect
        # wrong_tool: best-effort single "what was chosen" for the keyword
        # scan to exclude; if multiple tools were used we exclude all of
        # them from the candidate match set.
        corrected = None
        for wrong in (tools_used or [None]):
            corrected = _extract_corrected_tool(
                judgment.get("tool_selection_reason") or "", wrong
            )
            if corrected:
                break
        if not corrected:
            return [], [
                f"trace {trace_id}: verdict=incorrect, no confident correction "
                f"derivable from reason={judgment.get('tool_selection_reason')!r} — skipped"
            ]
        labels = [corrected]
        reason_note = f"hindsight-corrected from judge reason (was {tools_used})"

    has_db, has_docs, has_csv = _infer_source_flags(tools_used or labels)
    system_prompt = _build_router_system_prompt(has_db=has_db, has_docs=has_docs, has_csv=has_csv)
    user_content = f"Step: {question}"

    examples = []
    for label in labels:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": json.dumps({"tool": label})},
                ],
                # Metadata kept out of the "messages" shape TRL reads, but
                # useful for debugging/auditing the export — harmless extra
                # key, SFTTrainer only reads "messages".
                "_meta": {
                    "trace_id": trace_id,
                    "verdict": verdict,
                    "note": reason_note,
                },
            }
        )
    return examples, []


# ---------------------------------------------------------------------------
# DB access — reuses the async engine / session-factory pattern from
# api/trace_service.py + api/eval_service.py (create_async_engine +
# async_sessionmaker), rather than inventing a new DB access style.
# ---------------------------------------------------------------------------


async def _fetch_traces_with_judgments(database_url: str, pipeline: str) -> list[tuple[dict, Optional[dict]]]:
    from sqlalchemy import MetaData, select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    from api.eval_service import register_eval_judgments_table
    from api.trace_service import register_agent_traces_table

    normalized_url = (
        database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        .replace("postgres://", "postgresql+asyncpg://", 1)
    )

    metadata = MetaData()
    agent_traces = register_agent_traces_table(metadata)
    eval_judgments = register_eval_judgments_table(metadata)

    engine = create_async_engine(normalized_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as session:
            trace_rows = (
                await session.execute(
                    select(agent_traces).where(agent_traces.c.pipeline == pipeline)
                )
            ).mappings().all()
            judgment_rows = (
                await session.execute(select(eval_judgments))
            ).mappings().all()
    finally:
        await engine.dispose()

    # Most-recent-judgment-wins per trace_id (a trace could in principle be
    # judged more than once across eval batches).
    latest_judgment_by_trace: dict[str, dict] = {}
    for row in judgment_rows:
        row_d = dict(row)
        tid = row_d["trace_id"]
        existing = latest_judgment_by_trace.get(tid)
        if existing is None or (row_d.get("judged_at") or "") >= (existing.get("judged_at") or ""):
            latest_judgment_by_trace[tid] = row_d

    return [
        (dict(trace), latest_judgment_by_trace.get(dict(trace)["id"]))
        for trace in trace_rows
    ]


async def export_dataset(database_url: str, output_path: Path, pipeline: str = "autopilot") -> dict:
    """Fetch, build, and write the SFT dataset. Returns a summary dict."""
    pairs = await _fetch_traces_with_judgments(database_url, pipeline=pipeline)

    all_examples: list[dict] = []
    skip_reasons: list[str] = []
    for trace, judgment in pairs:
        examples, reasons = build_examples_for_trace(trace, judgment)
        all_examples.extend(examples)
        skip_reasons.extend(reasons)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for example in all_examples:
            # Strip the "_meta" debugging key before writing — SFTTrainer
            # only needs "messages", and we don't want to depend on
            # unrecognized keys being silently ignored by every consumer.
            f.write(json.dumps({"messages": example["messages"]}) + "\n")

    for reason in skip_reasons:
        logger.info(reason)

    return {
        "sampled_traces": len(pairs),
        "examples_written": len(all_examples),
        "skipped": len(skip_reasons),
        "output_path": str(output_path),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="finetune/data/routing_sft.jsonl",
        help="Output JSONL path (default: finetune/data/routing_sft.jsonl)",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1,
        help="Minimum examples required to write output without a hard warning banner "
        "(default: 1 — never hard-fails; just warns below SMALL_DATASET_WARNING_THRESHOLD).",
    )
    parser.add_argument(
        "--pipeline",
        default="autopilot",
        help="agent_traces.pipeline to export (default: autopilot — the only pipeline that "
        "calls _select_tool_llm).",
    )
    args = parser.parse_args(argv)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL is not set. Cannot export dataset.")
        return 1

    summary = asyncio.run(
        export_dataset(database_url, Path(args.output), pipeline=args.pipeline)
    )

    logger.info(
        "Exported %d examples from %d sampled traces (%d skipped) -> %s",
        summary["examples_written"],
        summary["sampled_traces"],
        summary["skipped"],
        summary["output_path"],
    )

    if summary["examples_written"] < args.min_examples:
        logger.warning(
            "Only %d examples written, below --min-examples=%d. Writing anyway — "
            "re-run as more traces accumulate.",
            summary["examples_written"], args.min_examples,
        )
    if summary["examples_written"] < SMALL_DATASET_WARNING_THRESHOLD:
        logger.warning(
            "Dataset has %d examples, below the %d-example threshold considered "
            "meaningful for any production fine-tune. This is expected for an early "
            "seed batch — re-run this export as more agent_traces/eval_judgments "
            "accumulate.",
            summary["examples_written"], SMALL_DATASET_WARNING_THRESHOLD,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
