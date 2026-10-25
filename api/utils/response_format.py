"""Shared LLM response format contract + deterministic markdown normalizer.

DOCBOT-2001 (FMT-08 / FMT-10 / BUG-25).

Before this module, formatting rules were copy-pasted into five persona
prompts, two autopilot synthesizer prompts, the hybrid synthesis prompt, the
RAG prompt and the DB answer generator, and the copies had drifted apart
(e.g. "use headers only when..." vs. a mandatory heading contract; mandatory
LaTeX vs. none). Every final-answer prompt now gets its formatting rules from
:func:`build_format_block`, and every final answer can be passed through
:func:`normalize_markdown` for deterministic cleanup.

Public API:

- ``FORMAT_CONTRACT_VERSION`` -- version tag of the contract text ("v1").
- ``FORMAT_CONTRACT_V1`` -- the source-agnostic contract text.
- ``SOURCE_DOC`` / ``SOURCE_DB`` / ``SOURCE_CSV`` / ``SOURCE_EDGAR`` and
  ``KNOWN_SOURCES`` -- source keys accepted by ``build_format_block``.
- ``CITATION_FORMATS`` -- citation syntax per source key.
- ``build_format_block(sources)`` -- the contract plus citation rules for the
  sources actually present in the prompt.
- ``normalize_markdown(text)`` -- idempotent markdown cleanup.
- ``unclosed_fence_suffix(text)`` -- the text that would close a dangling code
  fence; append-only, so streaming paths can emit it as a final token without
  rewriting tokens the client already rendered.
"""

from __future__ import annotations

import re
from typing import Iterable

FORMAT_CONTRACT_VERSION = "v1"

SOURCE_DOC = "doc"
SOURCE_DB = "db"
SOURCE_CSV = "csv"
SOURCE_EDGAR = "edgar"

# Ordered so the rendered citation block is deterministic.
KNOWN_SOURCES: tuple[str, ...] = (SOURCE_DOC, SOURCE_DB, SOURCE_CSV, SOURCE_EDGAR)

CITATION_FORMATS: dict[str, str] = {
    SOURCE_DOC: "[Source: <filename> p.<page>]",
    SOURCE_DB: "[DB: <table>]",
    SOURCE_CSV: "[CSV: <file>]",
    SOURCE_EDGAR: "[EDGAR: <form>/<period>]",
}

_CITATION_HINTS: dict[str, str] = {
    SOURCE_DOC: "for facts taken from uploaded documents",
    SOURCE_DB: "for figures taken from database query results",
    SOURCE_CSV: "for figures taken from uploaded CSV analysis",
    SOURCE_EDGAR: "for facts taken from SEC EDGAR filings (e.g. [EDGAR: 10-K/FY2024])",
}

FORMAT_CONTRACT_V1 = (
    "OUTPUT FORMAT RULES (" + FORMAT_CONTRACT_VERSION + "):\n"
    "- Write GitHub-flavored Markdown only. Never output raw HTML tags and never embed images.\n"
    "- Answer directly. Use **bold** for key terms, final values and conclusions; use bullet "
    "points for lists of items, qualitative analysis and risk factors.\n"
    "- Use ## / ### headings only to separate distinct sections or question parts. If a "
    "required section list is given above, use exactly those headings instead.\n"
    "- Present numerical data, projections and comparisons in markdown tables with a header "
    "row and a separator row. Keep tables to at most 6 columns; split or transpose wider data.\n"
    "- Format currency concisely: $4.2M, $1.3B, $850K. Never put raw arithmetic expressions "
    "in table cells.\n"
    "- For calculations, state each formula once, then show only the computed results.\n"
    "- Do not use LaTeX unless the user explicitly asks for math or formulas; write formulas "
    "in plain text (e.g. EBITDA margin = EBITDA / Revenue). When math notation is requested, "
    "use only `$$...$$` delimiters, never `\\(...\\)`, `\\[...\\]` or a single `$`.\n"
    "- Close every code fence you open."
)


def build_format_block(sources: Iterable[str] | None = None) -> str:
    """Return the format contract tailored to the sources present.

    ``sources`` is any iterable of :data:`KNOWN_SOURCES` keys; unknown keys are
    ignored. With no sources the block contains no citation rules.
    """
    present = {str(s).strip().lower() for s in (sources or ())}
    lines = [FORMAT_CONTRACT_V1]
    cited = [s for s in KNOWN_SOURCES if s in present]
    if cited:
        lines.append("CITATIONS (cite every factual claim with the syntax for its source):")
        for key in cited:
            lines.append(f"- {CITATION_FORMATS[key]} {_CITATION_HINTS[key]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# normalize_markdown
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(\s|$)")
_LIST_RE = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])\s+\S")
_INLINE_CODE_RE = re.compile(r"(`+)(?:[^`]|[^`].*?[^`])\1")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
# Attributes, when present, must look like real HTML (contain "="), so prose
# placeholders such as "<table names>" are not mistaken for tags.
_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)(\s+[^<>]*=[^<>]*?)?\s*/?>")
_SEP_CELL_RE = re.compile(r"^\s*(:?)\s*[-=_‒–—―]+\s*(:?)\s*$")
_PLACEHOLDER = "\x00{}\x00"
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

# Only real HTML element names are stripped, so placeholders such as
# "<table names>" or comparisons like "a < b > c" survive untouched.
_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "br", "caption", "center", "dd",
    "details", "div", "dl", "dt", "figcaption", "figure", "footer", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "summary", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
})
_INLINE_TAGS = frozenset({
    "a", "abbr", "b", "big", "cite", "code", "del", "em", "font", "i", "img", "ins",
    "kbd", "mark", "q", "s", "samp", "small", "span", "strike", "strong", "sub",
    "sup", "tt", "u", "var", "wbr", "iframe", "object", "embed", "svg", "video",
    "audio", "source", "picture", "input", "button", "form", "label", "html",
    "body", "head", "meta", "link", "title",
})


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", stripped)]


def _is_separator_row(line: str) -> bool:
    cells = _split_cells(line)
    return bool(cells) and all(_SEP_CELL_RE.match(c) for c in cells)


def _canonical_separator(header_cols: int, sep_line: str | None) -> str:
    aligns: list[tuple[str, str]] = []
    if sep_line is not None:
        for cell in _split_cells(sep_line):
            m = _SEP_CELL_RE.match(cell)
            aligns.append((m.group(1), m.group(2)) if m else ("", ""))
    cells = []
    for i in range(header_cols):
        left, right = aligns[i] if i < len(aligns) else ("", "")
        cells.append(f"{left}---{right}")
    return "| " + " | ".join(cells) + " |"


def _segments(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """Split lines into (is_code, lines) runs; close a dangling fence."""
    out: list[tuple[bool, list[str]]] = []
    buf: list[str] = []
    fence: str | None = None
    for line in lines:
        m = _FENCE_RE.match(line)
        if fence is None:
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                if buf:
                    out.append((False, buf))
                buf = [line]
                fence = m.group(1)
            else:
                buf.append(line)
        else:
            buf.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
                out.append((True, buf))
                buf = []
                fence = None
    if fence is not None:
        while len(buf) > 1 and not buf[-1].strip():
            buf.pop()
        buf.append(fence)
        out.append((True, buf))
    elif buf:
        out.append((False, buf))
    return out


def unclosed_fence_suffix(text: str) -> str:
    """Return the text needed to close a dangling code fence ("" if none)."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fence: str | None = None
    for line in lines:
        m = _FENCE_RE.match(line)
        if not m:
            continue
        if fence is None:
            if not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = m.group(1)
        elif m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
            fence = None
    if fence is None:
        return ""
    return fence if (text or "").endswith("\n") else "\n" + fence


def _strip_html(text: str) -> str:
    """Remove HTML outside inline code spans. Block tags become line breaks."""
    spans: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        spans.append(m.group(0))
        return _PLACEHOLDER.format(len(spans) - 1)

    text = _INLINE_CODE_RE.sub(_stash, text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _SCRIPT_STYLE_RE.sub("", text)

    out_lines = []
    for line in text.split("\n"):
        in_table = _is_table_line(line)

        def _replace(m: re.Match[str], _in_table: bool = in_table) -> str:
            name = m.group(1).lower()
            if name in _BLOCK_TAGS:
                return " " if _in_table else "\n"
            if name in _INLINE_TAGS:
                return ""
            return m.group(0)

        new = _TAG_RE.sub(_replace, line)
        if new != line:
            # Only tidy whitespace on lines we actually changed.
            new = "\n".join(part.rstrip() for part in new.split("\n"))
            if in_table:
                new = re.sub(r"[ \t]{2,}", " ", new)
            elif not new.strip():
                new = ""
        out_lines.append(new)
    text = "\n".join(out_lines)
    return _PLACEHOLDER_RE.sub(lambda m: spans[int(m.group(1))], text)


def _repair_tables(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if not _is_table_line(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < n and _is_table_line(lines[j]):
            j += 1
        block = lines[i:j]
        if len(block) >= 2 and not _is_separator_row(block[0]):
            header_cols = len(_split_cells(block[0]))
            if _is_separator_row(block[1]):
                block = [block[0], _canonical_separator(header_cols, block[1])] + block[2:]
            else:
                block = [block[0], _canonical_separator(header_cols, None)] + block[1:]
        out.extend(block)
        i = j
    return out


def _block_spacing(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_list = False
    in_math = False
    for line in lines:
        stripped = line.strip()
        if stripped == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_math:
            out.append(line)
            continue
        prev = out[-1] if out else ""
        prev_blank = not prev.strip()
        is_table = _is_table_line(line)
        prev_table = _is_table_line(prev)
        is_list = bool(_LIST_RE.match(line))

        if not prev_blank:
            need_blank = False
            if _HEADING_RE.match(line):
                need_blank = True
            elif is_table and not prev_table:
                need_blank = True
            elif prev_table and not is_table:
                need_blank = True
            elif is_list and not in_list:
                need_blank = True
            elif _HEADING_RE.match(prev) and stripped:
                need_blank = True
            if need_blank and out:
                out.append("")
        out.append(line)

        if not stripped:
            in_list = False
        elif is_list:
            in_list = True
        elif in_list and not line.startswith((" ", "\t")):
            # Lazy continuation keeps the list open; a heading/table closes it.
            if _HEADING_RE.match(line) or is_table:
                in_list = False
    return out


def normalize_markdown(text: str) -> str:
    """Deterministically clean up LLM markdown. Idempotent.

    - strips stray HTML tags/comments (outside code fences and inline code)
    - closes unclosed code fences
    - ensures a blank line before headings, lists and tables (and after tables)
    - collapses runs of blank lines to a single blank line
    - repairs missing or malformed table separator rows
    Code fence contents are never modified.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Pass 1: HTML stripping on prose only.
    parts = []
    for is_code, seg in _segments(text.split("\n")):
        chunk = "\n".join(seg)
        if not is_code:
            # Loop until stable so nested junk like "<<b>div>" cannot leave a
            # new tag behind (keeps the function idempotent).
            for _ in range(5):
                stripped = _strip_html(chunk)
                if stripped == chunk:
                    break
                chunk = stripped
        parts.append(chunk)
    text = "\n".join(parts)

    # Pass 2: structural fixes on prose; fences closed and kept verbatim.
    result: list[str] = []
    for is_code, seg in _segments(text.split("\n")):
        if is_code:
            result.extend(seg)
        else:
            result.extend(_block_spacing(_repair_tables(seg)))
    result = _collapse_blank_runs_outside_code(result)
    return "\n".join(result).strip("\n")


def _collapse_blank_runs_outside_code(lines: list[str]) -> list[str]:
    out: list[str] = []
    for is_code, seg in _segments(lines):
        if is_code:
            out.extend(seg)
            continue
        for line in seg:
            if not line.strip() and out and not out[-1].strip():
                continue
            out.append("" if not line.strip() else line)
    return out


def doc_sources_from_metadata(metadatas: Iterable[dict | None]) -> set[str]:
    """Map retrieved-chunk metadata to source keys for :func:`build_format_block`.

    Chunks ingested by the EDGAR pipeline carry ``filing_type``; everything
    else is an uploaded document. Always returns at least ``{SOURCE_DOC}``.
    """
    sources: set[str] = set()
    for meta in metadatas:
        if isinstance(meta, dict) and meta.get("filing_type"):
            sources.add(SOURCE_EDGAR)
        else:
            sources.add(SOURCE_DOC)
    return sources or {SOURCE_DOC}
