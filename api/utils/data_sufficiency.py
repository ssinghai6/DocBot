"""DOCBOT-2724/2725: deterministic "is there enough data?" checks (no LLM).

A forecast/regression over document text with no numeric series used to run
anyway: the generated code regex-extracted nothing, fit on NaN and printed an
all-NaN forecast plus empty charts. These helpers let Autopilot skip E2B when
the docs hold too few (year, value) pairs, and recognise a run whose output
is the INSUFFICIENT_DATA marker or only NaN.

Contract with the UI: step ``failure_reason`` is exactly ``"insufficient_data"``
and the step result text starts with ``"INSUFFICIENT_DATA:"``.
"""

from __future__ import annotations

import re
from typing import Optional

INSUFFICIENT_DATA_REASON = "insufficient_data"
INSUFFICIENT_DATA_PREFIX = "INSUFFICIENT_DATA:"
MIN_HISTORY_POINTS = 3

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MARKER_RE = re.compile(r"INSUFFICIENT_DATA:?[ \t]*(.*)")
_NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)
_FINITE_DECIMAL_RE = re.compile(r"(?<![\w.])-?\$?\d[\d,]*\.\d+")
_CHART_META_LINE_RE = re.compile(r"^.*CHART_META:.*$", re.MULTILINE)

# Window around a year in which a non-year number counts as that year's value.
# Wide enough for "2019: $2.45", "| 2019 | 2.45 |" and one-cell-per-line PDF
# tables; lenient on purpose: a false "enough" only means the sandbox runs and
# the post-run NaN check still catches it.
_BEFORE, _AFTER = 30, 50

_METRIC_STRIP_RES = (
    re.compile(r"\b(?:over|for|in)?\s*(?:the\s+)?(?:next|coming|following)\s+\d*\s*\w+", re.I),
    re.compile(r"\b(?:in|by|until|through|to)\s+(?:19|20)\d{2}\b", re.I),
    re.compile(
        r"\b(?:please|can|could|you|what|will|would|be|the|a|an|me|show|give|"
        r"predict(?:ion)?s?|forecast(?:s|ing)?|project(?:ion|ed|ing)?s?|"
        r"extrapolat\w*|estimate|regress\w*|future|run|do|make|build|model|of|for)\b",
        re.I,
    ),
)


def count_year_value_pairs(text: str) -> int:
    """Number of distinct years that have a numeric value next to them."""
    text = text or ""
    years_with_value: set[str] = set()
    for match in _YEAR_RE.finditer(text):
        window = text[max(0, match.start() - _BEFORE): match.start()] + " " + text[match.end(): match.end() + _AFTER]
        if _NUMBER_RE.search(_YEAR_RE.sub(" ", window)):
            years_with_value.add(match.group(0))
    return len(years_with_value)


def guess_metric(question: str) -> str:
    """Best-effort metric name from the question ("predict fuel prices next
    10 years" -> "fuel prices"), for the user-facing message."""
    text = question or ""
    for pattern in _METRIC_STRIP_RES:
        text = pattern.sub(" ", text)
    text = re.sub(r"[^\w\s%$/-]", " ", text)
    text = " ".join(text.split())
    return text[:80] or "the requested metric"


def insufficient_history_message(found: int, metric: str) -> str:
    return (
        f"{INSUFFICIENT_DATA_PREFIX} found {found} historical data points; "
        f"need at least {MIN_HISTORY_POINTS} historical values of {metric}"
    )


def detect_insufficient_output(stdout: str, is_forecast: bool) -> Optional[str]:
    """Return an ``INSUFFICIENT_DATA: ...`` result string when the sandbox
    output says the data was insufficient, or (for a forecast) produced no
    values / only NaN. ``None`` means the output is usable."""
    text = stdout or ""
    marker = _MARKER_RE.search(text)
    if marker:
        detail = marker.group(1).strip() or "the data lacks the numeric values this analysis needs"
        return f"{INSUFFICIENT_DATA_PREFIX} {detail[:300]}"
    if not is_forecast:
        return None
    body = _CHART_META_LINE_RE.sub("", text).strip()
    if not body:
        return f"{INSUFFICIENT_DATA_PREFIX} the forecast produced no output values"
    # shortcut: "only NaN" = NaN present and no decimal number anywhere; a
    # forecast printed as bare integers next to NaN would slip through.
    if _NAN_RE.search(body) and not _FINITE_DECIMAL_RE.search(body):
        return (
            f"{INSUFFICIENT_DATA_PREFIX} the forecast produced only NaN values "
            "(no usable historical numeric series)"
        )
    return None


# DOCBOT-2726: shared answer rule for the synthesizer and verifier prompts.
INSUFFICIENT_DATA_ANSWER_RULE = (
    "INSUFFICIENT DATA RULE: one or more investigation steps returned "
    "INSUFFICIENT_DATA. Your answer MUST open with \"Not enough data to <task>\" "
    "(e.g. \"Not enough data to forecast fuel prices\"), then list what was found "
    "versus what is missing (e.g. how many historical values exist and how many "
    "are needed), then give concrete next actions: upload a CSV with the "
    "historical series, connect a database that holds it, ingest an EDGAR "
    "filing, or pull it from a named public source (e.g. the U.S. EIA for "
    "energy prices). NEVER invent coefficients, growth rates, assumed values "
    "or \"illustrative\" forecasts — unless the user explicitly asked for a "
    "hypothetical or scenario, in which case label every assumed number as an "
    "assumption."
)
