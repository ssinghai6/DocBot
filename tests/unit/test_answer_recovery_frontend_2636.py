"""DOCBOT-2636: frontend recovery of a dropped hybrid answer (src/lib/chatStream.ts via node)."""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - runs the local node binary on a repo file, fixed args
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node binary not available")

PARTIAL = {"role": "assistant", "content": "", "sql": "SELECT segment, amount FROM revenue_by_segment"}
REPLY = {
    "answer": "## Executive Summary\nLab Services differs by $2.2M.",
    "citations": [{"source": "Tidewell-Draft-10K-FY2025.pdf", "page": 9, "text": "NOTE 12"}],
    "lineage": {"run_id": "run-1", "mode": "hybrid"},
    "traceId": "trace-1",
}


def _node(body: str) -> object:
    assert NODE is not None
    module = (REPO_ROOT / "src" / "lib" / "chatStream.ts").as_uri()
    script = (
        f"import * as cs from {json.dumps(module)};"
        "const sleep = async () => {};"
        f"const out = await (async () => {{ {body} }})();"
        "process.stdout.write(JSON.stringify(out));"
    )
    out = subprocess.run(  # nosec B603 - fixed argv, local node, no shell
        [NODE, "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return json.loads(out.stdout)


def test_polls_through_pending_then_returns_answer() -> None:
    got = _node(
        f"const seq = ['pending', 'pending', {json.dumps(REPLY)}]; let n = 0;"
        "const r = await cs.pollRecoveredAnswer(async () => seq[n++], { sleep });"
        "return { r, n };"
    )
    assert got == {"r": REPLY, "n": 3}


def test_gives_up_after_timeout() -> None:
    got = _node(
        "let n = 0;"
        "const r = await cs.pollRecoveredAnswer(async () => { n++; return 'pending' }, { intervalMs: 2000, timeoutMs: 60000, sleep });"
        "return { r, n };"
    )
    assert got == {"r": None, "n": 31}


def test_gone_run_stops_immediately() -> None:
    got = _node("let n = 0; const r = await cs.pollRecoveredAnswer(async () => { n++; return null }, { sleep }); return { r, n };")
    assert got == {"r": None, "n": 1}


def test_recovered_answer_replaces_partial_and_keeps_sql() -> None:
    msgs = [{"role": "user", "content": "q"}, PARTIAL]
    last = _node(f"return cs.applyRecoveredAnswer({json.dumps(msgs)}, {json.dumps(REPLY)}).at(-1);")
    assert isinstance(last, dict)
    assert last["content"] == REPLY["answer"]
    assert last["sql"] == PARTIAL["sql"]
    assert last["citations"] == REPLY["citations"]
    assert last["lineage"] == REPLY["lineage"]
    assert last["traceId"] == "trace-1"
