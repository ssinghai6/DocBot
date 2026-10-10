"""LLM-judge offline eval harness — standalone runner for DOCBOT-1519.

Samples recent rows from the live `agent_traces` table (DOCBOT-1512) and
asks an LLM judge whether each trace's tool/intent selection was plausible
and whether its final answer looks supported by the retrieved evidence.
Prints a summary and persists every judgment to `eval_judgments` for later
reference.

This is a manual/operational script, not a pytest test — it needs a live
DATABASE_URL (to read real agent_traces rows) and a live Groq/Gemini key (to
judge them), same category as `tests/eval/eval_latency.py` ("needs a live
external dependency, not deterministic, operator-run on demand"). It has no
`test_*` functions for pytest to collect and is intentionally not wired into
either `ci.yml` or `nightly-eval.yml` — see `tests/eval/README.md`.

`scripts/` was considered instead (per CLAUDE.md's Jira-sync precedent of
"operational script, not a pytest test") but that directory is entirely
gitignored in this repo (`.gitignore`: "scripts (contain credentials)"), so
anything placed there would never be committed. `tests/eval/` already holds
exactly this category of tool (see `eval_latency.py`), so this follows that
existing precedent instead of introducing a new one.

Run:
    python -m tests.eval.run_eval_batch --pipeline autopilot --limit 50
    python -m tests.eval.run_eval_batch --limit 20 --since-hours 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _main(pipeline: str | None, limit: int, since_hours: int) -> None:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import os

    from api.eval_service import register_eval_judgments_table, run_eval_batch, wire_eval_store
    from api.trace_service import register_agent_traces_table, wire_trace_store

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("DATABASE_URL is not set — nothing to sample from agent_traces. Aborting.")
        sys.exit(1)

    # asyncpg needs the postgresql+asyncpg:// scheme; DATABASE_URL from
    # Railway/most providers is plain postgresql://.
    async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(async_url)
    metadata = MetaData()
    agent_traces_table = register_agent_traces_table(metadata)
    eval_judgments_table = register_eval_judgments_table(metadata)

    async with engine.begin() as conn:
        # eval_judgments may not exist yet on a DB that already has
        # agent_traces from DOCBOT-1512 — create only what's missing.
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    wire_trace_store(agent_traces_table, session_factory)
    wire_eval_store(eval_judgments_table, session_factory)

    print(f"Running eval batch: pipeline={pipeline!r} limit={limit} since_hours={since_hours}")
    summary = await run_eval_batch(pipeline=pipeline, limit=limit)

    print(f"\nSampled: {summary['sampled']}")
    print(f"Tool selection verdicts: {summary['tool_selection_counts']}")
    print(f"Evidence support verdicts: {summary['evidence_support_counts']}")
    print(f"Estimated judge cost: ${summary['estimated_cost_usd']:.6f}")
    print(f"Flagged traces (not clean correct/supported): {len(summary['flagged'])}")
    for flagged in summary["flagged"]:
        print(f"  - trace_id={flagged['trace_id']} "
              f"tool={flagged['tool_selection_verdict']} ({flagged['tool_selection_reason']}) "
              f"evidence={flagged['evidence_support_verdict']} ({flagged['evidence_support_reason']})")

    await engine.dispose()
    print("\nDone. Judgments persisted to eval_judgments.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DOCBOT-1519 LLM-judge offline eval harness")
    parser.add_argument("--pipeline", default=None, help="Filter to one pipeline (autopilot|hybrid|db_chat|chat)")
    parser.add_argument("--limit", type=int, default=50, help="Max traces to sample (default 50)")
    parser.add_argument("--since-hours", type=int, default=168, help="Lookback window in hours (default 168 = 7 days)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_main(args.pipeline, args.limit, args.since_hours))
