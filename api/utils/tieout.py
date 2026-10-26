"""DOCBOT-2002: tie-out marker helpers.

The discrepancy detector produces structured ``DiscrepancyItem``s that reach
the client as ``lineage.discrepancies[]`` and render as TieOutCards. The answer
text only carries a short inline marker, ``[[discrepancy:<id>]]``, that the UI
turns into a badge linking to its card.

LLMs still sometimes write the legacy literal ``[DISCREPANCY] ...`` tag (or
echo the full pre-computed "Report says ... Delta: ..." line). These helpers
rewrite that output so a literal ``[DISCREPANCY]`` never reaches the user:

* ``replace_discrepancy_tags`` — tag / echoed block -> marker (or a neutral
  "**Discrepancy:**" label when no structured item matches).
* ``missing_marker_tail`` — a one-line "Tie-out:" footer for items the model
  never referenced, so every card has a badge in the text.
* ``detect_from_autopilot_steps`` — run detection over autopilot step data.

Pure functions; never raise on malformed input.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

from api.utils.discrepancy_detector import (
    DiscrepancyItem,
    DiscrepancyReport,
    _label_tokens,
    _parse_number,
    detect_discrepancies,
)

MARKER_RE = re.compile(r"\[\[discrepancy:(d-[0-9a-f]{8})\]\]")
_OPEN_TAG_RE = re.compile(r"\[\s*DISCREPANCY\s*\]", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"\[\s*/\s*DISCREPANCY\s*\]", re.IGNORECASE)
# The legacy prompt block echoed verbatim: "[DISCREPANCY] Net Income: Doc says
# 330 [Source: ...]. DB shows 325 [DB: ...]. Delta: -5 (-1.5%)" (also the
# DOCBOT-2002 "Report says / Books show" wording). Collapsed to a marker.
_ECHO_BLOCK_RE = re.compile(
    r"\[\s*DISCREPANCY\s*\][ \t]*(?:\*\*)?(?P<label>[^:\n\[]{1,100}?)(?:\*\*)?:[ \t]*"
    r"(?:Doc|Report)\s+says[^\n]*?Delta:[^\n]*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?P<sign>[-–]?)\s*[\$£€¥]?\s*(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)"
    r"\s*(?P<suffix>[KkMmBbTt](?:illion|n)?\b)?"
)
NEUTRAL_LABEL = "**Discrepancy:**"


def _line_around(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return text[line_start: line_end if line_end != -1 else len(text)]


def _numbers_in(text: str) -> list[float]:
    values: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        try:
            values.append(abs(_parse_number(m.group("number"), m.group("sign"), m.group("suffix"))))
        except ValueError:
            continue
    return values


def _close(a: float, b: float, tol: float = 0.005) -> bool:
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) / max(abs(a), abs(b)) <= tol


def match_item(context: str, items: Sequence[DiscrepancyItem]) -> Optional[DiscrepancyItem]:
    """Best structured item referred to by ``context`` (a line of answer text).

    Scores metric-token overlap plus report/books value hits. With exactly one
    item, an unscored tag still maps to it (the only discrepancy on record).
    """
    if not items:
        return None
    for m in MARKER_RE.finditer(context):
        for item in items:
            if item.id == m.group(1):
                return item

    ctx_tokens = _label_tokens(context)
    numbers = _numbers_in(context)
    best: Optional[DiscrepancyItem] = None
    best_score = 0.0
    for item in items:
        score = 0.0
        metric_tokens = _label_tokens(item.label)
        if metric_tokens and metric_tokens <= ctx_tokens:
            score += 2.0
        elif metric_tokens & ctx_tokens:
            score += len(metric_tokens & ctx_tokens) / len(metric_tokens)
        for target in (abs(item.doc_value), abs(item.db_value)):
            if any(_close(n, target) for n in numbers):
                score += 1.0
        if item.period and item.period.lower() in context.lower():
            score += 0.5
        if score > best_score:
            best, best_score = item, score
    if best is None and len(items) == 1:
        return items[0]
    return best


def replace_discrepancy_tags(text: str, items: Sequence[DiscrepancyItem]) -> str:
    """Rewrite literal ``[DISCREPANCY]`` tags / echoed blocks to inline markers."""
    if not text or "discrepancy" not in text.lower():
        return text

    def _echo(m: re.Match[str]) -> str:
        item = match_item(m.group(0), items)
        if item is None:
            return f"{NEUTRAL_LABEL} {m.group('label').strip()}"
        return f"**{item.display_label}** {item.marker}"

    out = _ECHO_BLOCK_RE.sub(_echo, text)

    pieces: list[str] = []
    last = 0
    for m in _OPEN_TAG_RE.finditer(out):
        pieces.append(out[last:m.start()])
        line = _line_around(out, m.start(), m.end())
        item = match_item(line, items)
        if item is None:
            replacement = NEUTRAL_LABEL
        elif item.marker in line:
            replacement = ""  # marker already on this line — just drop the tag
        else:
            replacement = item.marker
        pieces.append(replacement)
        last = m.end()
    pieces.append(out[last:])
    out = "".join(pieces)
    return _CLOSE_TAG_RE.sub("", out)


def referenced_ids(text: str) -> set[str]:
    return {m.group(1) for m in MARKER_RE.finditer(text or "")}


def missing_marker_tail(text: str, items: Iterable[DiscrepancyItem]) -> str:
    """Footer listing markers for items the answer never referenced ("" if none)."""
    seen = referenced_ids(text)
    missing = [i for i in items if i.id not in seen]
    if not missing:
        return ""
    parts = " · ".join(f"{i.display_label} {i.marker}" for i in missing)
    return f"\n\n**Tie-out:** {parts}"


def detect_from_autopilot_steps(step_results: Sequence[dict[str, Any]]) -> Optional[DiscrepancyReport]:
    """Run discrepancy detection over autopilot doc_search + sql_query steps.

    Steps stash their structured inputs under ``tieout_doc_context`` (full
    "Source: X, Page N" chunk text) and ``tieout_sql`` (result_preview /
    sources / sql_query). Returns None when either side is missing.
    """
    doc_parts: list[str] = []
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    sql_query: Optional[str] = None
    for step in step_results:
        if step.get("error"):
            continue
        ctx = step.get("tieout_doc_context")
        if isinstance(ctx, str) and ctx:
            doc_parts.append(ctx)
        sql = step.get("tieout_sql")
        if isinstance(sql, dict):
            preview = sql.get("result_preview") or []
            rows.extend(r for r in preview if isinstance(r, dict))
            for src in sql.get("sources") or []:
                if src not in sources:
                    sources.append(str(src))
            sql_query = sql_query or sql.get("sql_query")
    if not doc_parts or not rows:
        return None
    return detect_discrepancies(
        "\n\n".join(doc_parts),
        {"result_preview": rows, "sources": sources, "sql_query": sql_query},
    )
