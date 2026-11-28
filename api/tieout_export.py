"""Tie-out workpaper export (XLSX / PDF) — DOCBOT-2014.

One row per tie-out line in a persisted answer lineage: every exception
(``Lineage.discrepancies``, DOCBOT-2002) then every tied line
(``Lineage.ties``, DOCBOT-2732): item, report value + doc citation,
books value + DB/CSV citation, delta, delta %, severity, status.

XLSX via openpyxl; PDF via PyMuPDF (already a dependency for PDF parsing).
"""

from __future__ import annotations

import html
import io
from typing import Optional, Union

from api.utils.lineage import Lineage, LineageDiscrepancy

HEADERS = (
    "Item", "Period", "Report value", "Report source (file / page)",
    "Books value", "Books source (table / query)", "Delta (books - report)",
    "Delta %", "Severity", "Status",
)
TIE_TOLERANCE = 0.005  # rounding noise below half a cent

Cell = Union[str, float, None]


def _status(d: LineageDiscrepancy, tied: bool = False) -> str:
    if tied:
        return "Ties out"
    if d.delta is None:
        return "Needs review"
    return "Ties out" if abs(d.delta) < TIE_TOLERANCE else "Exception"


def _doc_source(d: LineageDiscrepancy) -> str:
    c = d.doc_citation
    return ", ".join(p for p in (c.file, f"p. {c.page}" if c.page is not None else None) if p)


def _db_source(d: LineageDiscrepancy) -> str:
    c = d.db_citation
    return " | ".join(p for p in (c.table, c.row, c.query) if p)


def tieout_rows(lineage: Lineage) -> list[tuple[Cell, ...]]:
    rows = [
        (
            d.metric, d.period or "", d.report_value, _doc_source(d),
            d.books_value, _db_source(d), d.delta, d.delta_pct,
            d.severity, _status(d, tied),
        )
        for lines, tied in ((lineage.discrepancies, False), (lineage.ties, True))
        for d in lines
    ]
    return sorted(rows, key=lambda r: r[-1] == "Ties out")  # exceptions first, stable


def _title(lineage: Lineage) -> str:
    return f"DocBot tie-out workpaper - run {lineage.run_id}"


def build_xlsx(lineage: Lineage) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Tie-out"
    ws.append([_title(lineage)])
    ws.append(["Question", lineage.question or ""])
    ws.append([])
    ws.append(list(HEADERS))
    for cell in ws[4]:
        cell.font = Font(bold=True)
    for row in tieout_rows(lineage):
        ws.append(list(row))
    for row in ws.iter_rows():
        for cell in row:
            # Document/LLM text must never become a live formula in the workpaper.
            if isinstance(cell.value, str) and cell.value.startswith(("=", "+", "-", "@")):
                cell.data_type = "s"
    for col in ("C", "E", "G"):
        for cell in ws[col][4:]:
            cell.number_format = "#,##0.00"
    for cell in ws["H"][4:]:
        cell.number_format = '0.00"%"'
    for col, width in zip("ABCDEFGHIJ", (32, 12, 16, 36, 16, 48, 18, 10, 10, 14)):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A5"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _fmt(v: Cell, suffix: str = "") -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}{suffix}"
    return html.escape(str(v))


def build_pdf(lineage: Lineage) -> bytes:
    import pymupdf

    head = "".join(f"<th>{html.escape(h)}</th>" for h in HEADERS)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{_fmt(v, '%' if i == 7 else '')}</td>" for i, v in enumerate(row)
        ) + "</tr>"
        for row in tieout_rows(lineage)
    )
    doc_html = (
        f"<h3>{html.escape(_title(lineage))}</h3>"
        f"<p>Question: {html.escape(lineage.question or '')}</p>"
        f"<table border='1' cellpadding='3' style='font-size:8px;border-collapse:collapse'>"
        f"<tr>{head}</tr>{body}</table>"
    )
    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)  # A4 landscape
    # shortcut: one page, text shrinks to fit; paginate if workpapers exceed ~40 lines.
    page.insert_htmlbox(pymupdf.Rect(28, 28, 814, 567), doc_html)
    out = doc.tobytes()
    doc.close()
    return out


def filename(lineage: Lineage, ext: str) -> str:
    safe = "".join(ch for ch in lineage.run_id if ch.isalnum() or ch in "-_")[:36] or "run"
    return f"tieout-{safe}.{ext}"


def render(lineage: Lineage, fmt: str) -> Optional[tuple[bytes, str, str]]:
    """(body, media type, filename) for ``fmt`` in {xlsx, pdf}; None for unknown formats."""
    if fmt == "xlsx":
        return (
            build_xlsx(lineage),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename(lineage, "xlsx"),
        )
    if fmt == "pdf":
        return build_pdf(lineage), "application/pdf", filename(lineage, "pdf")
    return None
