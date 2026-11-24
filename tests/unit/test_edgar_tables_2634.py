"""DOCBOT-2634: EDGAR tables render as rows, stay whole in chunks, and every
chunk carries a locator so citations and lineage keep distinct chunks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from api.connectors.edgar_connector import html_to_text
from api.edgar_service import FILING_FORMAT_VERSION
from api.utils.chunker import chunk_filing
from api.utils.lineage import LineageCollector

_INCOME_STATEMENT = """
<html><body>
<p>Item 8. Financial Statements and Supplementary Data.</p>
<p>Some prose about the year.</p>
<p>Consolidated Statements of Operations</p>
<p>(In thousands)</p>
<table>
 <tr><td colspan="3"></td><td></td><td></td></tr>
 <tr><td></td><td colspan="2">Year Ended<br/>December 31,</td></tr>
 <tr><td></td><td>2025</td><td></td><td>2024</td></tr>
 <tr><td>Revenue</td><td>$</td><td>2,883,501</td><td></td><td>$</td><td>2,808,332</td></tr>
 <tr><td>Operating expenses:</td></tr>
 <tr><td>Marketing</td><td>914,830</td><td></td><td>856,565</td></tr>
 <tr><td>Interest expense</td><td>(</td><td>18,509</td><td>)</td><td>(<span>13,806</span>)</td></tr>
 <tr><td>Growth</td><td>6.8</td><td>%</td></tr>
</table>
<p>Item 9. Changes in and Disagreements with Accountants.</p>
<p>None.</p>
</body></html>
"""


def test_table_rows_render_one_line_with_merged_symbols():
    text = html_to_text(_INCOME_STATEMENT)
    lines = text.splitlines()
    assert "Year Ended December 31," in lines
    assert "2025 | 2024" in lines
    assert "Revenue | $2,883,501 | $2,808,332" in lines
    assert "Marketing | 914,830 | 856,565" in lines
    assert "Interest expense | (18,509) | (13,806)" in lines
    assert "Growth | 6.8%" in lines
    # prose stays one element per line, the table is its own blank-line block
    assert "Some prose about the year." in lines
    assert "\n\nYear Ended December 31,\n2025 | 2024\nRevenue" in text


def test_chunk_filing_keeps_table_whole_with_caption_and_locators():
    docs = chunk_filing(html_to_text(_INCOME_STATEMENT))
    table = [d for d in docs if "Revenue | $2,883,501" in d.page_content]
    assert len(table) == 1
    content = table[0].page_content
    assert "Consolidated Statements of Operations\n(In thousands)\nYear Ended December 31," in content
    assert "Marketing | 914,830 | 856,565" in content
    assert table[0].metadata["section"] == "Item 8. Financial Statements and Supplementary Data."
    # DOCBOT-2637: no page footers in this snippet, so no page; chunks stay distinct by index
    assert [d.metadata["page"] for d in docs] == [None] * len(docs)
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    assert docs[-1].metadata["section"].startswith("Item 9.")


def test_long_table_splits_on_row_boundaries_repeating_header():
    rows = "\n".join(f"Line item {i} | {i * 1000:,} | {i * 900:,}" for i in range(80))
    text = f"Balance Sheet\n(In thousands)\n\n2025 | 2024\n{rows}\n\nAfter."
    docs = chunk_filing(text, table_max_chars=400)
    pieces = [d.page_content for d in docs if "Line item" in d.page_content]
    assert len(pieces) > 1
    seen: list[str] = []
    for piece in pieces:
        assert len(piece) <= 400
        assert piece.startswith("Balance Sheet\n(In thousands)\n2025 | 2024\n")
        body = piece.splitlines()[3:]
        assert all(line.startswith("Line item ") and line.count(" | ") == 2 for line in body)
        seen += body
    assert seen == rows.splitlines()  # every row once, in order, none cut


def test_toc_rows_do_not_set_section():
    text = "Item 1. | Business | 4\nItem 8. | Financial Statements | 60\n\nPART I\nItem 1. Business\nWe sell things."
    docs = chunk_filing(text)
    assert docs[-1].metadata["section"] in ("PART I", "Item 1. Business")
    assert not docs[0].metadata["section"].startswith("Item 8.")


def test_lineage_keeps_distinct_edgar_chunks():
    docs = chunk_filing("\n\n".join(f"Para {i} " + "word " * 300 for i in range(5)))
    assert len(docs) > 2
    lin = LineageCollector("r", "docs", "q")
    for d in docs:
        lin.add_source(kind="pdf", label="ETSY-10-K", page=d.metadata["page"],
                       chunk=d.metadata["chunk_index"], snippet=d.page_content)
    assert len(lin.build().sources) == len(docs)


def _session_factory():
    session = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock(return_value=None)
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    session.execute = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


def test_old_format_cache_entry_is_reingested_not_cloned():
    cached = {"accession_number": "acc-1", "session_id": "old-session", "text_hash": "abc123"}
    text = "Item 8. Financial Statements\n\nRevenue | $1 | $2\n\n" + "prose " * 100
    with (
        patch("api.connector_store.get_cached_filing", new_callable=AsyncMock, return_value=cached),
        patch("api.connector_store.save_filing_cache", new_callable=AsyncMock) as save,
        patch("api.connector_store.update_filing_cache_session", new_callable=AsyncMock) as repoint,
        patch("api.utils.vector_store.clone_store") as clone,
        patch("api.connectors.edgar_connector.EdgarConnector.fetch_filing_text",
              new_callable=AsyncMock, return_value=text),
        patch("api.utils.vector_store.create_store", return_value=MagicMock()) as create,
        patch("api.document_extractor.is_extractable_document", return_value=False),
        patch("api.edgar_service.insert", MagicMock()),
    ):
        from api.edgar_service import ingest_edgar_filing

        result = asyncio.run(ingest_edgar_filing(
            cik="1", accession_number="acc-1", primary_document="d.htm",
            filing_type="10-K", filing_date="2024-01-01", company_name="Co",
            ticker="CO", owner={"user_id": "u", "org_id": "o", "anon_owner": None},
            sessions_table=MagicMock(), async_session_factory=_session_factory(),
            vector_stores={}, get_embeddings_fn=MagicMock, extracted_fields_store={},
        ))

    clone.assert_not_called()
    save.assert_not_called()
    assert result["cached"] is False
    assert repoint.await_args.kwargs["text_hash"].startswith(f"{FILING_FORMAT_VERSION}:")
    chunks = create.call_args.args[1]
    assert all(c.metadata["source"] == "CO-10-K-2024-01-01" for c in chunks)
    assert any(c.page_content.endswith("Revenue | $1 | $2") for c in chunks)
