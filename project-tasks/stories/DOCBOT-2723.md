# DOCBOT-2723 BUG: SQL grounding on a real ledger: category values, empty-result retry, verified few-shot
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 3 | senior-dev | 2026-11-30 | Testing (2026-11-27) |

## User story
As a fractional CFO tying a city budget report to its account-level ledger, I want the SQL step to filter the category column that holds the report's line item, so that a line like "Energy & Utilities" is found in the books and not reported as missing.

## Why now
Prod run 2026-11-27 on the City of San Diego FY2023 General Fund ledger (`gl_actuals`, 29,206 rows; `real-data-test-2026-11-27.md` section 2): 0 of 6 line-item questions correct. SQL used `account LIKE '%Energy & Utilities%'` (NULL) and the answer said "not in the books"; the correct query is `SUM(amount) WHERE account_class = 'Energy and Utilities'` = 64,016,064. Totals used `-SUM(amount)` over 4 of 12 revenue types (-$294.6M, false $2.3B gap).

Root causes found in code:
1. The value-list scan read only the first 5000 rows. The file is sorted by department, so it saw 20 of 36 `account_class` values and 9 of 14 `account_type` values, marked the lists incomplete, and the prompt showed 5 of them. "Energy and Utilities", "Transient Occupancy Taxes", "Property Taxes" never reached the model.
2. A NULL total or 0 rows went straight to the answer as a real result.
3. Every query was stored as a few-shot example (`_store_query_history` ran for all runs, `_retrieve_few_shot` read them all), so a wrong filter could be copied into later questions.

## Scope
In:
1. Value-list scan 5000 -> 100,000 rows (`_SAMPLE_SCAN_ROWS`); `_SCHEMA_FMT` 4 so cached schemas re-introspect once.
2. Deterministic `_category_hints`: listed column values the question names (case, punctuation and '&'/'and' ignored, longest match wins) go into the prompt as `- gl_actuals.account_class = 'Energy and Utilities'`. No LLM call, no column names in code.
3. Prompt rules 9 (match line items to listed values first; LIKE on unlisted name columns only when nothing matches) and 10 (no unrequested sign/type filters, nothing copied from past queries, totals by category/type values, never revenue + expense in one sum). `PROMPT_VERSION_SQL_GEN` v8.
4. `_is_empty_result`: 0 rows, or every aggregate column NULL (or 0 under COALESCE). Triggers ONE corrective SQL-gen call with the SQL, "found nothing", and the matched listed values. It shares the DOCBOT-2320 one-extra-call budget (validation retry, drift retry, empty retry: at most one), so the worst case stays 6 LLM calls.
5. Still empty: metadata `empty_result: true`; the answer prompt (`PROMPT_VERSION_SQL_ANSWER_GEN` v2) and hybrid synthesis (`PROMPT_VERSION_HYBRID_SYNTHESIS` v3) say the books figure could not be determined, never "not in the books".
6. Few-shot hygiene: only first-try, non-empty runs are stored with the ` (verified)` marker in `result_summary`; `_retrieve_few_shot` reads only those. Older stored examples (unverified) are no longer used. History UI unchanged (`_parse_row_count` reads the first token).

Out:
- User thumbs-up as the verification signal: trace feedback (DOCBOT-1512) is not linked to query history rows.
- Sign convention: the model still sometimes negates a revenue line (`-SUM`) on a ledger that stores revenue positive (see verification). Needs a data signal for signed amounts.
- Tables over 100k rows: value lists still come from the first 100k rows.
- Docs-lane table reading and detector-as-gate (fixes 2 and 3 in the test report).

## Acceptance criteria
- [x] Given a ledger shaped like `gl_actuals`, when the question names "Energy & Utilities", then the prompt hints `account_class = 'Energy and Utilities'` and the account-name column gets no list.
- [x] Given categories past row 5000, then the value list is complete.
- [x] Given a NULL total, then exactly one retry with the SQL and the listed values as feedback, and the fixed SQL is used.
- [x] Given still empty after the retry, then `empty_result` is true and the answer prompt says the figure could not be determined.
- [x] Given a retried or empty run, then it is not stored as verified; few-shot reads verified rows only.
- [ ] Live check on the prod model (Bedrock gpt-oss-120b) for the 6 San Diego questions.

## Test / verification
`tests/unit/test_sql_grounding_2723.py`: 18 tests, all 18 fail on the old code. Existing 2633/2637/2638/2639/2720 tests pass (two mocks updated for the new `empty` kwarg / feedback retry; `v7`->`v8`, schema_fmt 3->4 literals).
Suite `-m "not external and not postgres"`: 3083 passed, 1 pre-existing failure (`test_v1_openapi_snapshot`, also fails on main).

Local SQL step on `sd_gl_fy23_clean.sqlite` (Gemini 2.5 Flash fallback, not the prod model), before -> after:

| Question | Books truth | Before | After |
|---|---|---|---|
| Total GF revenue | 2,011,950,249 | -294,618,640 (`-SUM`, 4 of 12 types) | 2,011,950,249 (all 12 revenue types) |
| Total GF expenditures | 1,959,919,069 | 362,138,303 (3 classes) | 1,955,827,934 (9 classes, missed 'Other' 4.09M) |
| Transient Occupancy Taxes | 161,811,336 | NULL (`account = ...`) | -161,811,336 (right filter, sign flipped) |
| Energy & Utilities | 64,016,064 | NULL (`account = 'Energy & Utilities'`) | 64,016,064 |
| Transfers (expense) | 150,426,510 | NULL (`account = 'Transfers (expense)'`) | 150,426,510 (`account_class = 'Transfers Out'`) |
| Personnel Services | 765,310,025 | NULL (`account_type = 'Personnel Services'`) | 1,283,319,361 (`account_type = 'Personnel'`, includes fringe) |

## Dependencies
Blocked by: none. Related: DOCBOT-2720 (value pairs), DOCBOT-2320 (corrective retry).
Blocks: investor demo 2026-12-01.

## Definition of done
- [x] AC met (except live prod-model check)
- [x] tests/CI or evidence
- [x] OpenProject ticket updated
- [x] tracker row updated
- [x] no secrets / no banned claims

## Open questions
- Merge after the 2026-11-25 cut-off needs founder approval.
