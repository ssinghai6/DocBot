"""DOCBOT-2005: human-readable Autopilot step summaries (deterministic, no LLM).

``build_step_summary`` turns an executor step_result into one short line
("Queried trial_balance — 12 rows"); ``clean_step_details`` tidies the raw
tool output shown behind the UI's "details" toggle (wrapper tags stripped,
JSON pretty-printed).
"""

from __future__ import annotations

import json
import re
from typing import Any

from api.utils.exceptions import safe_error_message

# Internal step-result prefixes that signal a failed/empty step.
FAILURE_CONTENT_PREFIXES = (
    "Step failed",
    "CSV analysis failed",
    "CSV code generation failed",
    "No SQL result",
    "No prior data",
    "No data source",
    "Code generation failed",
)

_FAILURE_REASONS = {
    "sql_query": "the query returned no result",
    "doc_search": "the documents are no longer available",
    "python_analysis": "the analysis could not run",
}

_UNTRUSTED_TAG_RE = re.compile(r"</?untrusted[\w-]*>\s*", re.IGNORECASE)
_WRAPPER_LINE_RE = re.compile(r"^\s*</?[A-Za-z_][\w-]*>\s*$\n?", re.MULTILINE)
_UNSAFE_NAME_RE = re.compile(r"[<>{}\[\]`\n\r]")


def _name(value: Any) -> str:
    """Table/file name safe for a one-line summary (no markup, bounded)."""
    text = _UNSAFE_NAME_RE.sub("", str(value or "")).strip()
    return text[:60]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def is_failed_step(step: dict) -> bool:
    result = str(step.get("result") or "")
    return bool(
        step.get("error")
        or step.get("failure_reason")
        or step.get("tool") == "unsupported"
        or result.startswith(FAILURE_CONTENT_PREFIXES)
    )


def build_step_summary(step: dict) -> str:
    tool = str(step.get("tool") or "")

    if is_failed_step(step):
        if tool == "unsupported":
            reason = "no data source connected"
        elif step.get("failure_reason") == "step_timeout":  # DOCBOT-2310
            reason = "timed out"
        elif step.get("failure_reason") == "insufficient_data":  # DOCBOT-2725
            return "Insufficient data"
        elif step.get("failure_reason") and tool in _FAILURE_REASONS:
            reason = _FAILURE_REASONS[tool]
        else:
            reason = safe_error_message(str(step.get("error") or ""))
        return f"Step failed: {reason}"

    if tool == "sql_query":
        sources = (step.get("tieout_sql") or {}).get("sources") or []
        target = _name(sources[0]) if len(sources) == 1 else ""
        if not target:
            target = f"{len(sources)} tables" if len(sources) > 1 else "the database"
        rows = step.get("row_count")
        count = _plural(int(rows), "row") if isinstance(rows, (int, float)) else "results returned"
        return f"Queried {target} — {count}"

    if tool == "doc_search":
        passages = int(step.get("passages") or 0)
        if not passages:
            return "Searched documents — no matching passages"
        sources = [s for s in (_name(s) for s in step.get("doc_sources") or []) if s]
        target = sources[0] if len(sources) == 1 else (
            _plural(len(sources), "document") if sources else "documents"
        )
        return f"Searched {target} — {_plural(passages, 'passage')}"

    if tool == "python_analysis":
        return "Ran Python — chart created" if step.get("chart_b64") else "Ran Python — analysis complete"

    return "Step complete"


def clean_step_details(text: str) -> str:
    """Strip untrusted/XML-ish wrapper tags and pretty-print embedded JSON."""
    text = _UNTRUSTED_TAG_RE.sub("", text or "")
    text = _WRAPPER_LINE_RE.sub("", text).strip()
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start >= 0:
        try:
            parsed = json.loads(text[start:])
        except ValueError:
            return text
        pretty = json.dumps(parsed, indent=2, default=str)
        prefix = text[:start].strip()
        return f"{prefix}\n\n```json\n{pretty}\n```" if prefix else f"```json\n{pretty}\n```"
    return text
