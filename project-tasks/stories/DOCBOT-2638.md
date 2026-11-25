# DOCBOT-2638 BUG: Tie-out against a journal-entry ledger in Postgres misses the planted break on YoY questions
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 3 | senior-dev | 2026-11-25 | In progress (2026-11-24) |

## User story
As a fractional CFO connecting my client's real accounting database (journal entries, not a statement table), I want the tie-out against a 10-K to compare every line and flag a break whatever way I phrase the question, so that I can trust it on my own books.

## Why now
Real-data test 2026-11-24 with the books as a live Postgres general ledger (accounts / journal_entries / journal_lines, 506 lines, Etsy FY2025 10-K; planted +$5M Marketing accrual): the opex question caught the break 3/3 with 0 false flags, but the year-over-year question caught it 0/3. The DOCBOT-2637 tests used a single statement-level table, so these gaps did not show. Investor call 2026-12-01.

## Scope
In:
1. Book-guided re-retrieval searched by the bare line names ("General and administrative Marketing Product development"), which matches MD&A prose, not the income statement. The query now starts with "Consolidated statements of operations" (page 67 ranks first).
2. Line completion only added lines between two selected rows. A ledger has no "Total operating expenses" line, so "Asset impairment charges" (last line before the total) was never added. Once the query picked 2+ rows of a block, rows up to the block's "Total ..." row are added too.
3. Rows re-run by `complete_tieout_lines` kept Postgres NUMERIC as `Decimal`, so the detector compared 0 pairs after completion. They now go through the same JSON round trip as the main pipeline (numbers as floats).
4. SQL generation once grouped by sub-account and invented an `agree_with_books = 'Yes'` column, which synthesis repeated as a reconciliation. Prompt rule 5: the database cannot see the document, never add agree / match / variance flag columns; group by the statement-line rollup column when accounts roll up to one. Prompt version v5.

Out:
- Balance-sheet-only line lists (the re-retrieval query names the statement of operations).
- Ledger traps (draft / void entries, sign conventions beyond account_type).

## Acceptance criteria
- [x] Given a ledger opex query selecting Marketing, Product development and G&A, when the filing table has Asset impairment charges before Total operating expenses, then that line is added; a single selected line is never extended.
- [x] Given Postgres NUMERIC amounts, then completed rows reach the detector as numbers.
- [x] Given books lines with no comparable filing figure, then the re-retrieval query names the consolidated statements of operations.
- [x] Live Postgres rerun (12 runs, Bedrock gpt-oss-120b): opex and YoY questions flag the planted Marketing break 6/6, 0 false flags on clean books 6/6, income statement (page 67) cited in 12/12, no agreement-flag columns.

## Deliverable and approach
See Scope. Tests: `tests/unit/test_real_filing_fixes_2637.py` (DOCBOT-2638 section), `tests/integration/test_tieout_line_completion.py` (Decimal rows).

## Test / verification
CI suite; live rerun against local Postgres 18 via psycopg2 (only harness change: the SSRF resolver allows localhost).

## Dependencies
Blocked by: DOCBOT-2637 (merged).
Blocks: investor demo 2026-12-01.

## Definition of done
- [x] AC met
- [x] tests/CI or evidence
- [ ] OpenProject ticket updated
- [x] tracker row updated
- [x] no secrets / no banned claims

## Open questions
- None.
