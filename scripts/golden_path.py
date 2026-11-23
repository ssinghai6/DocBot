#!/usr/bin/env python3
"""DOCBOT-2604 golden-path check: demo init -> tie-out chat -> XLSX export -> autopilot.

Usage: python3 scripts/golden_path.py --base-url https://docbot-production-c4c0.up.railway.app [--runs N]
Exits nonzero if any step of any run fails; prints per-step p50/p95 at the end.
Each run costs real LLM spend (anonymous demo cap is $10/day), so keep --runs small on prod.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
import time
from typing import Any, Callable, Iterable

# Planted QuickBite mismatches (api/demo_service.py): (doc value, DB value), in $M.
PLANTED = [(330.0, 325.0), (890.0, 885.0)]
TIEOUT_Q = "Are there any discrepancies between the filing and the database? Compare Q4 net income and segment revenue."
AUTOPILOT_Q = "Analyze quarterly revenue and net income trends and assess which segment drives growth."
STEPS = ["demo_init", "tieout_chat", "xlsx_export", "autopilot"]
LEAK = re.compile(r"<(reasoning|think)>", re.I)  # PR #135 strips these; one in an answer is a regression


def leaked(events: list[dict[str, Any]]) -> bool:
    """True when streamed answer text (token/answer events) contains a reasoning tag."""
    return any(LEAK.search(str(e.get("content") or "")) for e in events if e.get("type") in ("token", "answer"))


def percentiles(secs: list[float]) -> tuple[float, float]:
    """(p50, nearest-rank p95)."""
    s = sorted(secs)
    return statistics.median(s), s[-(-95 * len(s) // 100) - 1]


def parse_sse(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Decode `data: {json}` lines into dicts; non-JSON data lines are skipped."""
    events = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except ValueError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _matches(value: Any, target: float) -> bool:
    # Values may arrive in $M or raw dollars.
    return isinstance(value, (int, float)) and any(abs(value - target * s) < 0.51 * s for s in (1, 1e6))


def planted_discrepancy(events: list[dict[str, Any]]) -> tuple[bool, str]:
    """True when a lineage discrepancy carries a planted (doc, DB) pair and both citations."""
    for ev in events:
        if ev.get("type") != "lineage":
            continue
        for d in ev.get("discrepancies") or []:
            for doc_v, db_v in PLANTED:
                if _matches(d.get("report_value"), doc_v) and _matches(d.get("books_value"), db_v):
                    if any((d.get("doc_citation") or {}).values()) and any((d.get("db_citation") or {}).values()):
                        return True, f"{d.get('metric')}: doc {doc_v} vs db {db_v}"
                    return False, f"{d.get('metric')} flagged but a citation is empty"
    return False, "no planted discrepancy in lineage event"


def run_id_of(events: list[dict[str, Any]]) -> str | None:
    return next((e["run_id"] for e in events if e.get("type") == "done" and e.get("run_id")), None)


def check_xlsx(body: bytes) -> tuple[bool, str]:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(io.BytesIO(body), read_only=True)
    except Exception as exc:  # openpyxl raises many types for bad zips
        return False, f"not a valid xlsx: {exc}"
    rows = sum(ws.max_row or 0 for ws in wb.worksheets)
    return (rows > 1, f"{len(wb.worksheets)} sheets, {rows} rows")


def check_autopilot(events: list[dict[str, Any]]) -> tuple[bool, str]:
    errs = [e for e in events if e.get("type") == "error"]
    if leaked(events):
        return False, "reasoning tag leaked into answer"
    if errs:
        return False, f"error event: {str(errs[0].get('content'))[:120]}"
    answer = next((e for e in events if e.get("type") == "answer" and e.get("content")), None)
    done = any(e.get("type") == "done" for e in events)
    if not (answer and done):
        return False, f"answer={bool(answer)} done={done}"
    return True, f"{sum(e.get('type') == 'step' for e in events)} steps, answer {len(answer['content'])} chars"


def _stream(http: Any, url: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    with http.post(url, json=payload, stream=True, timeout=240) as r:
        r.raise_for_status()
        return parse_sse(ln for ln in r.iter_lines(decode_unicode=True) if ln)


def run_once(base: str, dataset: str) -> list[tuple[str, bool, float, str]]:
    import requests

    http = requests.Session()
    out: list[tuple[str, bool, float, str]] = []
    ctx: dict[str, Any] = {}

    def step(name: str, fn: Callable[[], tuple[bool, str]]) -> bool:
        t = time.perf_counter()
        try:
            ok, note = fn()
        except Exception as exc:  # report any failure as a failed step, not a crash
            ok, note = False, f"{type(exc).__name__}: {str(exc)[:160]}"
        out.append((name, ok, time.perf_counter() - t, note))
        return ok

    def init() -> tuple[bool, str]:
        r = http.post(f"{base}/api/demo/init", params={"dataset": dataset}, timeout=120)
        r.raise_for_status()
        ctx.update(r.json())
        return bool(ctx.get("session_id") and ctx.get("connection_id")), f"session={str(ctx.get('session_id'))[:8]}"

    def chat() -> tuple[bool, str]:
        ev = _stream(http, f"{base}/api/hybrid/chat", {
            "question": TIEOUT_Q, "session_id": ctx["session_id"], "connection_id": ctx["connection_id"],
            "has_docs": True, "persona": "Finance Expert"})
        ctx["run_id"] = run_id_of(ev)
        ok, note = planted_discrepancy(ev)
        if leaked(ev):
            return False, "reasoning tag leaked into answer"
        return ok and bool(ctx["run_id"]), note

    def xlsx() -> tuple[bool, str]:
        r = http.get(f"{base}/api/lineage/{ctx['run_id']}/tieout.xlsx", timeout=60)
        r.raise_for_status()
        return check_xlsx(r.content)

    def autopilot() -> tuple[bool, str]:
        return check_autopilot(_stream(http, f"{base}/api/autopilot/run", {
            "question": AUTOPILOT_Q, "session_id": ctx["session_id"], "connection_id": ctx["connection_id"],
            "has_docs": True, "has_db": True}))

    for name, fn in zip(STEPS, [init, chat, xlsx, autopilot]):
        if not step(name, fn):
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--dataset", default="quickbite")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    failed = 0
    times: dict[str, list[float]] = {n: [] for n in STEPS}
    for i in range(1, a.runs + 1):
        print(f"\nrun {i}/{a.runs} @ {base}")
        res = run_once(base, a.dataset)
        for name, ok, secs, note in res:
            if ok:
                times[name].append(secs)
            print(f"  {name:<12} {'PASS' if ok else 'FAIL'}  {secs:7.2f}s  {note}")
        if len(res) < len(STEPS) or not all(r[1] for r in res):
            failed += 1
    print(f"\n{a.runs - failed}/{a.runs} runs passed")
    for name, secs in times.items():
        if secs:
            p50, p95 = percentiles(secs)
            print(f"  {name:<12} p50 {p50:6.2f}s  p95 {p95:6.2f}s  (n={len(secs)} passing)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
