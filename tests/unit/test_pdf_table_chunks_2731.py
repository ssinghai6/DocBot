"""DOCBOT-2731: PDF summary tables become row-level chunks with the true page.

Fixture: ``pdftotext -f 3 -l 3 -layout`` of the City of San Diego FY2023
Financial Performance Report (public), re-typeset into a tiny PDF in Courier
so word positions keep the report's column layout.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from langchain_core.documents import Document

from api.utils import chunker
from api.utils.chunker import chunk_pdf_pages
from api.utils.pdf_io import extract_pdf_pages
from api.utils.pdf_layout import is_number_cell, layout_page_text
from api.utils.query_expansion import deduplicate_docs

P3 = (Path(__file__).parent.parent / "fixtures" / "sandiego_fy23_p3_layout.txt").read_text()


def _pdf(tmp_path: Path, pages: list[str]) -> str:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=900, height=700)
        for i, line in enumerate(text.splitlines()):
            if line.strip():
                page.insert_text((20, 30 + i * 9), line, fontname="cour", fontsize=6)
    path = tmp_path / "report.pdf"
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def report(tmp_path: Path) -> str:
    def prose(n: int) -> str:
        return "\n".join(f"Page {n} line {i}: this report summarises General Fund results." for i in range(25))

    return _pdf(tmp_path, [prose(1), prose(2), P3])


def _rows(docs: list[Document]) -> list[Document]:
    return [d for d in docs if d.metadata["chunk_type"] == "table_row"]


def test_number_cells() -> None:
    for c in ("706,243,579", "(5,443,628)", "-31.3%", "$ 1,808,942,388", "102.2%"):
        assert is_number_cell(c), c
    for c in ("2023", "FY23", "Revenue", "-"):
        assert not is_number_cell(c), c


def test_p3_rows_carry_title_headers_values_and_true_page(report: str) -> None:
    pages, n = extract_pdf_pages(report, 10)
    assert n == 3
    docs = chunk_pdf_pages(pages, "report.pdf", "financial")
    rows = _rows(docs)
    energy = [d for d in rows if "| Energy & Utilities |" in d.page_content]
    assert len(energy) == 1
    text = energy[0].page_content
    assert text.startswith("General Fund Summary")
    assert "(p.3) | Expenditures | Energy & Utilities |" in text
    assert "FY23 Adopted Budget 55,510,820" in text
    assert "FY23 Current Budget 59,370,507" in text
    assert "FY23 Actuals 64,016,368" in text
    assert energy[0].metadata == {"source": "report.pdf", "page": 3, "chunk_type": "table_row", "chunk_index": energy[0].metadata["chunk_index"]}
    # both "Transfers" rows survive, told apart by their section
    transfers = [d.page_content for d in rows if "| Transfers |" in d.page_content]
    assert any("| Revenue | Transfers |" in t and "240,414,164" in t for t in transfers)
    assert any("| Expenditures | Transfers |" in t and "149,860,395" in t for t in transfers)
    assert any("Total General Fund Expenditures" in d.page_content and "1,959,931,710" in d.page_content for d in rows)
    assert all(d.metadata["page"] == 3 for d in rows)


def test_prose_keeps_its_page(report: str) -> None:
    pages, _ = extract_pdf_pages(report, 10)
    docs = chunk_pdf_pages(pages, "report.pdf", "general")
    text = [d for d in docs if d.metadata["chunk_type"] == "text"]
    assert {d.metadata["page"] for d in text} >= {1, 2}
    for d in text:
        if d.metadata["page"] == 3:
            continue  # the summary page's own title / footnote prose
        first = next(ln for ln in d.page_content.splitlines() if ln.startswith("Page "))
        assert first.startswith(f"Page {d.metadata['page']} ")
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))


def test_page_without_table_keeps_native_text() -> None:
    words = [(10, 10, 40, 20, "Hello", 0, 0, 0), (45, 10, 80, 20, "world", 0, 0, 1)]
    assert layout_page_text(words, "native text") == "native text"


def test_row_chunk_cap_falls_back_to_whole_table(report: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunker, "PDF_TABLE_ROW_CHUNKS_MAX", 0)
    pages, _ = extract_pdf_pages(report, 10)
    docs = chunk_pdf_pages(pages, "report.pdf", "general")
    assert not _rows(docs)
    tables = [d for d in docs if d.metadata["chunk_type"] == "table"]
    assert tables and all(len(d.page_content) <= chunker.FILING_TABLE_MAX_CHARS + 200 for d in tables)


def test_dedupe_keeps_rows_sharing_a_title_prefix() -> None:
    prefix = "General Fund Summary / As of Period 12, Ended June, 2023 (100% Completed) / (Unaudited) (p.3)"
    a = Document(page_content=f"{prefix} | Supplies | 1", metadata={"source": "r", "page": 3})
    b = Document(page_content=f"{prefix} | Debt | 2", metadata={"source": "r", "page": 3})
    assert len(deduplicate_docs([[a, b], [a]])) == 2
