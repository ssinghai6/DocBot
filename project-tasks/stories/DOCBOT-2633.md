# DOCBOT-2633 BUG: SQL generator sees only 5 values per category column, misses 'Marketing'
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 2 | senior-dev | 2026-11-25 | Testing (2026-11-24) |

## User story
As a fractional CFO tying a 10-K out to my books, I want the SQL step to see every account name in my books table, so that it returns the rows the question asks about instead of guessing names.

## Why now
Measured 2026-11-24 on Etsy's FY2025 10-K against a 14-account books table: the generator saw 5 alphabetical values per text column, never saw 'Marketing', 'Income from operations' or 'Net income', guessed names, and returned the Marketing row in 0 of 12 runs. The planted Marketing discrepancy was never found. Investor demo is 2026-12-01.

## Scope
In:
- Text columns with at most 50 distinct values: the full sorted list is cached with the schema and shown to the SQL generator as `values`.
- Above 50 distinct values: no values are sent (unchanged from before), so real values such as customer names never reach the prompt.
- Per-column (1500 chars) and prompt-wide (6000 chars) budgets so wide schemas cannot blow up the prompt.
- Prompt rule: a `values` list is complete; filter with those exact strings and never invent names.
- Schema cache format marker so cache rows written before this change are re-introspected.

Out:
- CSV path (E2B pandas, does not use this code).
- Table selector and its 20 columns / 8 tables caps (unchanged).

## Acceptance criteria
- [ ] Given a text column with <= 50 distinct values in a table of <= 5000 rows, when the schema is introspected, then every value is cached and the column is marked complete.
- [ ] Given a text column with > 50 distinct values, when the schema is introspected, then no values are cached for it.
- [ ] Given the full list exceeds the character budget, when the schema is introspected or the prompt is built, then the column falls back to 5 examples.
- [ ] Given a schema cached before this change, when it is read, then it is re-introspected.
- [ ] Given the Etsy books table and "Do the FY2025 operating expense lines in the 10-K agree with the books?", when the SQL step runs 3 times on Bedrock, then the Marketing row is returned each time.

## Deliverable and approach
`api/db_service.py`: `_attach_sample_values` (bounded scan, full list for <= 50 distinct values), `_SCHEMA_FMT` cache marker checked in `get_schema`, `_generate_sql` renders `values` vs `e.g.` and adds rule 7. Tests in `tests/integration/test_schema_sample_values.py`.

## Test / verification
Unit/integration tests for full list, nothing above 50, char budget, scan bound, prompt rendering and cache upgrade. Offline tie-out eval, live tie-out eval (3 runs, Bedrock) before/after, Etsy SQL step 3x.

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
- none
