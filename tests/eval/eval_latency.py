"""Latency evaluation — measures perceived speed against a running DocBot server.

Reports time-to-first-token (TTFT) and end-to-end latency percentiles (p50/p95)
for the streaming chat pipelines. Requires a running backend; it is a manual
script, not a CI test.

Run against local:   python -m tests.eval.eval_latency
Run against prod:    DOCBOT_BASE_URL=https://<backend>.railway.app python -m tests.eval.eval_latency

TTFT is the most important UX metric for a streaming app — it's how fast the
user sees *something* happen, independent of total generation time.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import httpx

BASE_URL = os.getenv("DOCBOT_BASE_URL", "http://127.0.0.1:8000")
RUNS = int(os.getenv("EVAL_LATENCY_RUNS", "5"))

QUESTIONS = [
    "Summarize TechCorp's FY2024 financial highlights",
    "What was the total revenue and net income?",
    "Does the Q4 net income in the 10-K match the database?",
]


def _init_demo(client: httpx.Client) -> tuple[str, str]:
    r = client.post(f"{BASE_URL}/api/demo/init", timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("session_id", ""), data.get("connection_id", "")


def _time_stream(client: httpx.Client, path: str, payload: dict) -> tuple[float, float]:
    """Return (ttft_ms, total_ms) for one streaming request."""
    start = time.monotonic()
    ttft = None
    with client.stream("POST", f"{BASE_URL}{path}", json=payload, timeout=120) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_lines():
            if not chunk:
                continue
            if ttft is None and chunk.startswith("data:"):
                # First token event that carries content
                try:
                    evt = json.loads(chunk[5:].strip())
                except json.JSONDecodeError:
                    continue
                if evt.get("type") in ("token", "metadata", "plan"):
                    ttft = (time.monotonic() - start) * 1000
    total = (time.monotonic() - start) * 1000
    return (ttft if ttft is not None else total), total


def _percentiles(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    idx95 = max(0, int(round(0.95 * len(ordered))) - 1)
    return p50, ordered[idx95]


def main() -> None:
    print(f"=== Latency Evaluation ({BASE_URL}) ===")
    with httpx.Client() as client:
        session_id, connection_id = _init_demo(client)
        print(f"demo session={session_id[:8]}… connection={connection_id[:8]}…  runs={RUNS}\n")

        ttfts: list[float] = []
        totals: list[float] = []
        for i in range(RUNS):
            q = QUESTIONS[i % len(QUESTIONS)]
            payload = {"session_id": session_id, "message": q, "persona": "Finance Expert"}
            try:
                ttft, total = _time_stream(client, "/api/chat", payload)
                ttfts.append(ttft)
                totals.append(total)
                print(f"  run {i+1}: TTFT={ttft:7.0f}ms  total={total:7.0f}ms  | {q[:40]}")
            except Exception as exc:  # noqa: BLE001 — manual diagnostic script
                print(f"  run {i+1}: FAILED ({type(exc).__name__}: {exc})")

        ttft_p50, ttft_p95 = _percentiles(ttfts)
        tot_p50, tot_p95 = _percentiles(totals)
        print("\n  --- Chat pipeline (/api/chat) ---")
        print(f"  TTFT  p50={ttft_p50:.0f}ms  p95={ttft_p95:.0f}ms")
        print(f"  Total p50={tot_p50:.0f}ms  p95={tot_p95:.0f}ms")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
