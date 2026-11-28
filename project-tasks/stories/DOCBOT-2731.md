# DOCBOT-2731 BUG: PDF summary tables unreadable; SQL sign flip, broad category column, revenue mixed into expense totals
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 3 | senior-dev | 2026-11-30 | Testing (2026-11-28) |

## User story
As a fractional CFO tying a city budget report PDF to its ledger, I want the docs lane to read the report's summary table rows with their real page, and the SQL lane to total the right lines with the right sign, so that a tie-out question gives the correct report figure, books figure and difference.

## Why now
Prod run 2026-11-28 after DOCBOT-2723 (`sql-baseline-2026-11-27/RESULTS_after_2723_2026-11-28.md`): books figure right 6/8, end-to-end 2/8.
1. PDF uploads went through one SemanticChunker pass over the concatenated text. PyMuPDF puts every table cell on its own line, so no chunk held "Energy & Utilities ... 64,016,368"; page numbers were restored by an 80-char prefix match and fell back to page 1 (one answer cited p.7 rows as "page 3").
2. Prompt rule 5 told the model revenue = -SUM(amount); it applied it on a ledger that stores all amounts positive.
3. "Personnel Services" was filtered on `account_type = 'Personnel'` (incl. fringe, $1.28B) instead of `account_class = 'Personnel Cost'` ($765.3M).
4. An expense total once included revenue account_types ($2.38B).

## Scope
In:
1. `api/utils/pdf_layout.py`: rebuild visual lines from PyMuPDF word positions; a run of 2+ lines whose cells after the label are numbers is a table; multi-line headers are merged per column by x position; tables render as `label | v1 | v2` rows, prose keeps PyMuPDF block order. `extract_pdf_pages` returns this text.
2. `chunk_pdf_pages` (chunker.py): per PDF, one chunk per table row = table title (caption, reused `_caption`) + `(p.N)` + section (Revenue / Expenditures) + label + `header value` pairs; true page in metadata and the `Source: X, Page N` header. Prose goes through `chunk_document` in one pass and is mapped back to the page it starts on. Row chunks capped at 3000 per PDF (`PDF_TABLE_ROW_CHUNKS_MAX`), then whole tables via `_split_table`.
3. `deduplicate_docs` keyed on full text (row chunks share a 100-char title prefix and were collapsed to one).
4. `_attach_amount_signs`: amount-like numeric columns get `amount_sign` = `credits_negative` (negative rows >= 10% of absolute value) or `positive`. The prompt marks the column and emits the -SUM rule only for `credits_negative`; `positive` says never negate. `_SCHEMA_FMT` 5.
5. `_category_hints`: a hint in a broad column is replaced by values of a finer column (more listed values) that start with the same phrase (`account_type 'Personnel'` -> `account_class 'Personnel Cost'`); rule 9 says filter the finer, report-line-level column.
6. `_side_exclusions`: for an expense-only (or revenue-only) question, lists the other side's values of mixed category columns as "never include". Name-based, deterministic. SQL gen already runs at temperature 0 (test added). `PROMPT_VERSION_SQL_GEN` v9.

Out:
- Rotated / scanned tables, row labels wrapped over two lines.
- Revenue/expense classification of category names with no revenue/expense words (e.g. 'Mission Bay') stays with the model.

## Acceptance criteria
- [x] San Diego p.3 layout text in a generated PDF yields a row chunk `General Fund Summary ... (p.3) | Expenditures | Energy & Utilities | FY23 Adopted Budget 55,510,820 | ... | FY23 Actuals 64,016,368 ...` with page 3 metadata.
- [x] Both "Transfers" rows (revenue and expenditure) survive with their section.
- [x] Retrieval for all 8 San Diego questions returns the p.3 row with the report figure (local, local ONNX embeddings + reranker).
- [x] San Diego `amount` -> `positive` (no -SUM rule in the prompt); Chipotle `amount` -> `credits_negative` (rule kept).
- [x] "Personnel Services" hints `account_class = 'Personnel Cost'`.
- [x] An expense question lists the revenue account_types to exclude.
- [ ] Prod check of the 8 San Diego questions on the prod model.

## Test / verification
`tests/unit/test_pdf_table_chunks_2731.py`, `tests/unit/test_sql_signs_granularity_2731.py`. Local end-to-end results in the PR description.

Blocked by: none. Related: DOCBOT-2723, DOCBOT-2634 (EDGAR table chunker), DOCBOT-2637 (page locators).
