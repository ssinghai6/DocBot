# DOCBOT-2634 BUG: EDGAR filing tables are not retrievable and citations say Page 0
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 3 | senior-dev | 2026-11-25 | Testing (2026-11-24) |

## User story
As a fractional CFO tying a 10-K out to my books, I want the financial statement tables of an EDGAR filing to reach the answer with their numbers, so that the comparison checks the actual statement lines and each citation points at a distinct place in the filing.

## Why now
Measured 2026-11-24 on Etsy's FY2025 10-K (622 chunks, hybrid tie-out on Bedrock): the EDGAR HTML stripping wrote one table cell per line, so statement rows were split across lines and chunks. All 6 Q2 runs ("Do the FY2025 operating expense lines in the 10-K agree with the books?") retrieved the same 5 prose chunks with no opex figures, the "Income from operations 266,210" chunk was never retrieved, and the "Revenue" caption was cut from its value. EDGAR chunks had no page, so every citation said "Page 0" and the Inspector showed 1 of 5 chunks. Investor demo is 2026-12-01.

## Scope
In:
- `html_to_text`: each table row becomes one line, cells joined by " | ", spacer cells dropped, "$", ")" and "%" fragments merged into the number cell; each table is its own block. Prose unchanged.
- `chunk_filing` (EDGAR path only): prose uses the same recursive splitter; a table is never split mid-row, carries its caption (up to 3 short lines before it), and a long table splits on row boundaries repeating caption and header rows.
- Chunk locator: `page` = 1-based chunk ordinal (unique, so citation and lineage dedupe keep distinct chunks), `section` = enclosing Part/Item heading, `chunk_index`.
- EDGAR cache format version (`v2:` prefix on `text_hash`); older cache rows are re-ingested instead of cloned.

Out:
- PDF upload chunking, lineage dedupe rules, citation schema (unchanged).
- Retrieval candidate pool size and lexical/hybrid search (follow-up: Q1 operating income still not in top 5).

## Acceptance criteria
- [ ] Given a filing table row with "$", parenthesised negatives, "%" and colspan spacers, when the HTML is converted, then the row is one line like "Interest expense | (18,509) | (13,806)".
- [ ] Given a table longer than the chunk budget, when the filing is chunked, then every piece holds whole rows only and repeats the caption and header rows.
- [ ] Given an EDGAR chunk, when it is stored, then it has a unique `page` >= 1 and a `section`.
- [ ] Given 5 distinct EDGAR chunks retrieved, when lineage records them, then the Inspector shows 5 sources.
- [ ] Given a cache row written before this change, when the filing is ingested again, then it is re-downloaded and re-chunked.
- [ ] Given the Etsy FY2025 10-K and Q2, when retrieval runs with the local reranker, then the income statement chunk with the opex rows is in the top 5.

## Deliverable and approach
`api/connectors/edgar_connector.py` (`html_to_text`, `render_table`), `api/utils/chunker.py` (`chunk_filing`), `api/edgar_service.py` (uses `chunk_filing`, `FILING_FORMAT_VERSION`), `api/connector_store.py` (`update_filing_cache_session` takes `text_hash`). Tests in `tests/unit/test_edgar_tables_2634.py`.

## Test / verification
Unit tests for row rendering, whole-row splits, locators, lineage and cache upgrade. Retrieval before/after on the Etsy 10-K for Q1 and Q2 through `rag_retrieve` with the local reranker. Offline tie-out eval.

## Dependencies
Blocked by: none.
Blocks: investor demo 2026-12-01.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- `page` is a chunk ordinal, not a printed page number of the filing. Real page numbers would need page-break parsing and would put several chunks on one page.
