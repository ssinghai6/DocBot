"""Rebuild financial table rows from PDF word positions (DOCBOT-2731).

PyMuPDF's plain text puts every table cell on its own line, so a row label and
its numbers end up far apart and no chunk holds "Energy & Utilities ...
64,016,368". This module groups a page's words into visual lines by y
position, splits each line into cells at wide x gaps, and renders a table
(a run of lines whose cells after the label are all numbers) as::

    <label header> | FY23 Adopted Budget | FY23 Current Budget | ...
    Revenue
    Property Taxes | 706,243,579 | 706,243,579 | ...

Header lines above the first row are merged per column by x position. Prose
keeps PyMuPDF's own block / line order (so two-column text is not
interleaved); only the words inside a detected table are taken out of it.
Pure functions over ``page.get_text("words")`` tuples, so tests need no PDF.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# x0, y0, x1, y1, text, block_no, line_no, word_no
Word = Sequence[Any]

_NUM_RE = re.compile(r"^\$?\(?[-–]?\$?\d[\d,]*(?:\.\d+)?%?\)?%?$")
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
_FOOTNOTE_RE = re.compile(r"(?<=[a-z])\d\b")  # "Actuals1" -> "Actuals"
_LABEL_MAX = 80
_DASHES = {"-", "–", "—"}  # a "nothing" value cell
_MAX_GAP_LINES = 3  # non-row lines allowed between two rows of one table


def is_number_cell(cell: str) -> bool:
    """A money / count / percent cell; a bare year is a header, not a value."""
    c = cell.replace(" ", "")
    return bool(_NUM_RE.match(c)) and not _YEAR_RE.match(c)


def _lines(words: list[Word]) -> list[list[Word]]:
    out: list[list[Word]] = []
    centers: list[float] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        yc, h = (w[1] + w[3]) / 2, w[3] - w[1]
        if out and abs(yc - centers[-1]) <= 0.4 * h:
            out[-1].append(w)
        else:
            out.append([w])
            centers.append(yc)
    return out


def _cells(line: list[Word]) -> list[tuple[float, float, str]]:
    cells: list[list[Any]] = []
    for w in sorted(line, key=lambda w: w[0]):
        if cells and w[0] - cells[-1][1] <= max(3.5, 0.5 * (w[3] - w[1])):
            cells[-1][1] = w[2]
            cells[-1][2] += " " + str(w[4])
        else:
            cells.append([w[0], w[2], str(w[4])])
    return [(x0, x1, _FOOTNOTE_RE.sub("", t)) for x0, x1, t in cells if t != "$"]


def _is_value(cell: str) -> bool:
    return cell in _DASHES or is_number_cell(cell)


def _is_row(cells: list[tuple[float, float, str]]) -> bool:
    """2+ numbers, and every cell from the first value on is a value."""
    vals = [i for i, c in enumerate(cells) if _is_value(c[2])]
    return (
        sum(is_number_cell(c[2]) for c in cells) >= 2
        and vals == list(range(vals[0], len(cells)))
    )


def _is_label(cells: list[tuple[float, float, str]]) -> bool:
    return len(cells) == 1 and not is_number_cell(cells[0][2]) and len(cells[0][2]) <= _LABEL_MAX


def _is_header(cells: list[tuple[float, float, str]]) -> bool:
    return len(cells) >= 2 and not any(is_number_cell(c[2]) for c in cells)


def _render(cells_by_line: list[list[tuple[float, float, str]]], start: int, end: int, first_row: int) -> str:
    """Table text for lines[start:end]; lines before first_row are headers / labels."""
    rows = [c for c in cells_by_line[first_row:end] if _is_row(c)]
    num_x0 = min(c[0] for r in rows for c in r if is_number_cell(c[2]))
    headers = [c for c in cells_by_line[start:first_row] if _is_header(c)]
    label_head: list[str] = []
    if headers:
        base = max(headers, key=len)
        centers = [(c[0] + c[1]) / 2 for c in base if c[1] > num_x0]
    else:
        widest = max(rows, key=len)
        centers = [(c[0] + c[1]) / 2 for c in widest if is_number_cell(c[2])]
    if not centers:
        return ""
    texts: list[list[str]] = [[] for _ in centers]

    def col(cell: tuple[float, float, str]) -> int:
        mid = (cell[0] + cell[1]) / 2
        return min(range(len(centers)), key=lambda k: abs(centers[k] - mid))

    for line in headers:
        for cell in line:
            if cell[1] <= num_x0:
                label_head.append(cell[2])  # e.g. "($ in thousands)" over the label column
            else:
                texts[col(cell)].append(cell[2])
    out: list[str] = []
    if headers:
        out.append(" | ".join([" ".join(label_head)] + [" ".join(t) for t in texts]))
    for i in range(start, end):
        cells = cells_by_line[i]
        if i < first_row and _is_header(cells):
            continue
        if not _is_row(cells):
            out.append(" ".join(c[2] for c in cells))  # section label ("Revenue")
            continue
        label = " ".join(c[2] for c in cells if not _is_value(c[2]))
        vals = [""] * len(centers)
        for cell in cells:
            if _is_value(cell[2]):
                k = col(cell)
                vals[k] = f"{vals[k]} {cell[2].lstrip('$ ')}".strip()
        out.append(" | ".join([label] + vals))
    return "\n".join(out)


def _tables(cells_by_line: list[list[tuple[float, float, str]]]) -> list[tuple[int, int, int]]:
    """(start, first_row, end) line spans of tables with 2+ rows."""
    row_idx = [i for i, c in enumerate(cells_by_line) if _is_row(c)]
    runs: list[list[int]] = []
    for i in row_idx:
        between = cells_by_line[runs[-1][-1] + 1:i] if runs else None
        if runs and len(between) <= _MAX_GAP_LINES and all(map(_is_label, between)):
            runs[-1].append(i)
        else:
            runs.append([i])
    spans: list[tuple[int, int, int]] = []
    for run in runs:
        if len(run) < 2:
            continue
        j, labels, headers = run[0] - 1, 0, 0
        while j >= 0 and labels < 2 and _is_label(cells_by_line[j]):
            j, labels = j - 1, labels + 1
        while j >= 0 and headers < 3 and _is_header(cells_by_line[j]):
            j, headers = j - 1, headers + 1
        start = j + 1 if headers else run[0]  # section labels count only under a header
        spans.append((start, run[0], run[-1] + 1))
    return spans


def layout_page_text(words: list[Word], native_text: str) -> str:
    """Page text with tables rendered as pipe rows; ``native_text`` when none."""
    lines = _lines(words)
    cells_by_line = [_cells(line) for line in lines]
    spans = _tables(cells_by_line)
    if not spans:
        return native_text
    table_words: set[tuple[Any, Any, Any]] = set()
    segments: list[tuple[float, str]] = []
    for start, first_row, end in spans:
        text = _render(cells_by_line, start, end, first_row)
        if not text:
            continue
        segments.append((min(w[1] for w in lines[start]), "\n\n" + text + "\n\n"))
        table_words.update((w[5], w[6], w[7]) for line in lines[start:end] for w in line)
    # prose: PyMuPDF's own block / line grouping, minus the table words
    blocks: dict[Any, dict[Any, list[str]]] = {}
    tops: dict[Any, float] = {}
    for w in words:
        if (w[5], w[6], w[7]) in table_words:
            continue
        blocks.setdefault(w[5], {}).setdefault(w[6], []).append(str(w[4]))
        tops[w[5]] = min(tops.get(w[5], w[1]), w[1])
    for b, blines in blocks.items():
        segments.append((tops[b], "\n".join(" ".join(ws) for ws in blines.values())))
    segments.sort(key=lambda s: s[0])
    return re.sub(r"\n{3,}", "\n\n", "\n".join(s for _, s in segments)).strip()
