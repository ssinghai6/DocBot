# DOCBOT-2637 BUG: Real-filing tie-out says a books line is missing, cites a chunk number as the page
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 8 | senior-dev | 2026-11-25 | In progress (2026-11-24) |

## User story
As a fractional CFO tying out a real 10-K against my books, I want every line of a filing table compared, real page citations, and a working app after I sign in, so that I can trust and check what DocBot tells me.

## Why now
Real-data test on prod 2026-11-24 (Etsy FY2025 10-K vs XBRL-built books, signed in, owner mode): the planted $5M Marketing break was caught 3/3 with 0 false flags, but the run found four defects. Investor call 2026-12-01.

## Scope
In:
1. Skipped books line: the SQL step's `IN (...)` filter left out "Asset impairment charges" (between G&A and Total operating expenses) and synthesis told the user the line is "not in the books" (5 of 6 answers, one marked "Match: No"). Deterministic fix: `api/utils/tieout_lines.py` adds filing table rows that sit in a short gap between lines the query already selected, only when the books' complete value list holds them; `db_service.complete_tieout_lines` re-runs the query (sqlglot validation, read-only executor, PII masking); `hybrid_service` applies it after both lanes. Synthesis rule: a line without a database row was not part of the query result, never "missing from the books".
2. Citation "Page 495": filing chunks stored the chunk ordinal as `page`. `chunk_filing` now maps each chunk to its printed page from the filing's page footers (Etsy income statement = page 67); filings without footers get `page=None` and are cited by section. Shared `api/utils/doc_refs.py`; lineage keeps chunks distinct by `chunk_index`; filing cache format bumped to v3 so cached filings re-chunk.
3. `GET /api/connectors` 500 for signed-in users: `column marketplace_connections.sync_status does not exist`. The prod table predates DOCBOT-704 and `create_all` never adds columns. `CONNECTOR_COLUMN_MIGRATIONS` (`ADD COLUMN IF NOT EXISTS`) run in `init_db`.
4. "Clear session" / "Clear chat" left the previous answer's lineage in the Inspector.
5. Wrong year-over-year claims: 5 of 24 recorded answers stated a wrong trend (operating income "+8.8%" when -30.0%, tax "rose" from $107.5M to $83.7M, opex "+5%" when +8.8%, impairment "steady" when FY2024 was nil). `api/utils/yoy_facts.py` computes each filing table row's latest-vs-prior-year change in code (dash = nil, parenthesised lines described by size) and the hybrid, docs-chat and Autopilot prompts must use those facts instead of their own arithmetic.
6. Year-over-year question defeated the tie-out: books queries pivoted by year (`amount_2025`) carried no period, filing table cells had none either, and retrieval returned a stock-comp table instead of the income statement. Detector reads the year from pivoted column names and from the filing table's year header; when books lines have no comparable filing figure, hybrid re-retrieves by those line names (no LLM call, filings only); SQL generation gets 3000 output tokens (1200 cut a reasoning-model query mid-SQL).

Out:
- `=`-filters and multi-column filters (only `IN` lists are completed).
- Charts and the SQL-lane answer are built from the first query's rows.

## Acceptance criteria
- [x] Given the Etsy opex query that skips "Asset impairment charges", when hybrid runs, then the books row is queried and no answer calls the line missing from the books.
- [x] Given a revenue / operating income / net income question, then no lines are added (far-apart picks are left alone).
- [x] Given the Etsy 10-K, then the income statement chunk is cited as page 67; a filing without page footers shows no page number.
- [x] Given a prod DB whose marketplace_connections lacks the sync columns, after deploy `GET /api/connectors` returns 200.
- [x] Given Clear session, then the Inspector shows no lineage from the cleared answers.
- [x] Real-data rerun (12 runs, Bedrock): planted gap caught 3/3, 0 false flags, no "not in the books" claim.
- [x] Given a filing table with years, then every year-over-year statement in the answer matches the computed change (scored on the Etsy runs).
- [x] Given "How did each FY2025 operating expense line change compared with FY2024, and do the FY2025 figures agree with the books?", then the planted Marketing break is flagged.

## Deliverable and approach
See Scope. Tests: `tests/unit/test_real_filing_fixes_2637.py`, `tests/integration/test_tieout_line_completion.py` (real SQLite books), updated `test_edgar_tables_2634.py`, `test_edgar_service.py`.

## Test / verification
CI suite; `npx tsc --noEmit`; lint; real-data rerun locally with prod models, then on prod in Chrome after merge.

## Dependencies
Blocked by: none.
Blocks: investor demo 2026-12-01.

## Definition of done
- [x] AC met
- [x] tests/CI or evidence
- [x] OpenProject ticket updated
- [x] tracker row updated
- [x] no secrets / no banned claims

## Open questions
- None.
