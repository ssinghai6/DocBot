"""DOCBOT-2635: a hybrid answer stream that breaks off keeps what arrived.

Prod 2026-11-24 (Tidewell demo, "Does segment revenue in Note 12 agree with
the books?"): the SQL and analysis-code events had rendered, then the body
read rejected and the client replaced the bubble with a generic "error with
the hybrid query" text, hiding the real cause. These tests run the frontend
helpers (src/lib/chatStream.ts) under the local node binary.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - runs the local node binary on a repo file, fixed args
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node binary not available")

# State of the assistant bubble after the captured prod events
# `metadata` (sql_query) and `analysis_code`, before any `token` arrived.
CAPTURED_SQL = (
    "SELECT segment AS metric, amount FROM revenue_by_segment WHERE period = 'FY2025'\n"
    "UNION ALL\n"
    "SELECT line_item AS metric, amount FROM income_statement WHERE period = 'FY2025'"
)
PARTIAL = {
    "role": "assistant",
    "content": "",
    "sql": CAPTURED_SQL,
    "analysisCode": "pivot = df.pivot(index='metric', columns='source', values='amount')",
    "charts": [],
}


def _run(messages: list[dict[str, object]], error_js: str) -> list[dict[str, object]]:
    assert NODE is not None
    module = (REPO_ROOT / "src" / "lib" / "chatStream.ts").as_uri()
    script = (
        f"import {{ keepPartialAnswer, streamDroppedNote }} from {json.dumps(module)};"
        f"const out = keepPartialAnswer({json.dumps(messages)}, streamDroppedNote({error_js}));"
        "process.stdout.write(JSON.stringify(out));"
    )
    out = subprocess.run(  # nosec B603 - fixed argv, local node, no shell
        [NODE, "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(out.stdout)


def test_drop_before_tokens_keeps_sql_and_names_cause() -> None:
    out = _run([{"role": "user", "content": "q"}, PARTIAL], 'new TypeError("network error")')
    last = out[-1]
    assert last["sql"] == CAPTURED_SQL
    assert last["analysisCode"] == PARTIAL["analysisCode"]
    assert "TypeError: network error" in last["content"]
    assert "error with the hybrid query" not in last["content"]


def test_drop_mid_answer_keeps_streamed_text() -> None:
    streamed = {**PARTIAL, "content": "## Executive Summary\nLab Services differs by $2.2M."}
    last = _run([streamed], 'new TypeError("terminated")')[-1]
    assert last["content"].startswith(streamed["content"])
    assert "TypeError: terminated" in last["content"]


def test_non_assistant_tail_untouched() -> None:
    msgs = [{"role": "user", "content": "q"}]
    assert _run(msgs, 'new Error("x")') == msgs
