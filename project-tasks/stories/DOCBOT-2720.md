# DOCBOT-2720 BUG: SQL step pairs account numbers to names by list position
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 2 | senior-dev | 2026-11-30 | Testing (2026-11-27) |

## User story
As a fractional CFO tying a 10-K to a ledger that has an account number and an account name column, I want the SQL step to pick the right account numbers, so that clean books are not reported as billions out.

## Why now
Verified live on prod 2026-11-27 with the Chipotle ledger `gl_journal(account_no, account_name, ...)`, 11 accounts. `_attach_sample_values` gives each low-cardinality text column its own sorted list, so the prompt showed account_no '4000'..'9100' and account_name alphabetically ('Depreciation and amortization', 'Food, beverage and packaging', ...). The model zipped them by position: Food/Labor/Occupancy/Other operating became 5100/6200/6300/6400 (true 5100-5400) and Revenue became '9100' or '8000' (true 4000). Tie-out then claimed an $11.9B mismatch on clean books. The Etsy ledger only passed because the model filtered by name. Investor call 2026-12-01.

## Scope
In:
1. Introspection: when two sampled text columns of a table have the same number of values (>= 3) and are 1:1 in the scanned rows (one bounded `SELECT DISTINCT a, b`), the shorter-valued column gets `value_pairs` ([[code, name], ...], sorted by code) and both get `paired_with`. Detection by data only, no column names. At most 10 pair checks per table.
2. SQL prompt renders the pairs (`account_no (TEXT, values paired account_no=account_name: '4000'='Revenue', ...)`) and the name column as `see account_no pairs`, instead of two unrelated lists. Over the prompt value budget it shows the first 5 pairs. Plain lists (`sample_values`) are kept in the schema for tie-out line recovery.
3. Prompt rule 8: value lists are sorted independently; never match code to name by list position. `PROMPT_VERSION_SQL_GEN` v7.
4. `_SCHEMA_FMT` 3 so cached schemas are re-introspected once and get pairs; a schema without pairs still renders the old way.

Out:
- Code/name columns that are not 1:1 (one code, two names) or whose list was truncated: unchanged behaviour (separate lists).
- Tables over 5000 rows: pairs come from the first 5000 rows, the same scan the value lists already use.

## Acceptance criteria
- [x] Given the GL table above, then account_no carries the 11 true pairs and the prompt shows '5200'='Labor', never '5100'='Depreciation and amortization'.
- [x] Given a code with two names, then no pairs and the separate lists render as before.
- [x] Given a cached schema without pairs, then the prompt renders as before.
- [x] Given the prompt value budget is spent, then the first 5 pairs render (not two misaligned 5-value lists).
- [ ] Live SQL step on the prod model (Bedrock gpt-oss-120b) selects 4000 for revenue and 5100-5400 for the operating lines. Local run used the Gemini fallback (Groq blocked from this network, no Bedrock key): new code 3/3 correct codes; old code filtered by name 3/3, so it did not reproduce the bug.

## Deliverable and approach
See Scope. Tests: `tests/unit/test_sql_code_name_pairs_2720.py` (3 fail on the old code; the other 3 guard the fallback paths).

## Test / verification
CI suite (`-m "not external and not postgres"`): 3034 passed, 1 pre-existing failure (`test_v1_openapi_snapshot`, also fails on main).

## Dependencies
Blocked by: none.
Blocks: investor demo 2026-12-01.

## Definition of done
- [x] AC met (except live prod-model check)
- [x] tests/CI or evidence
- [x] OpenProject ticket updated
- [x] tracker row updated
- [x] no secrets / no banned claims

## Open questions
- Merge after the 2026-11-25 cut-off needs founder approval.
