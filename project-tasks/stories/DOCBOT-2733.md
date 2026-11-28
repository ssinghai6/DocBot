# DOCBOT-2733 BUG: uploaded books fail as "internal error" after a restart; chart sandbox runs on every tie-out
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 2 | senior-dev | 2026-11-30 | Testing (2026-11-28) |

## User story
As a fractional CFO running a tie-out, I want a clear "please re-upload" message when my uploaded books are gone, and I want tie-out answers to skip chart code I did not ask for, so that the answer is fast and my books never leave the SQL pipeline.

## Why now
- Prod 2026-11-27: after a container restart every uploaded SQLite connection answered "An internal error occurred." while /api/health said ok. The "re-upload" check in `get_schema` ran only on a schema-cache miss; with a cached schema the query opened the missing path, SQLite created an empty file there, so even the cache-miss check passed from then on. Hybrid swallowed the error (`_collect_sql_result` returned None) and answered without books.
- Prod 2026-11-28 San Diego run: step 6.5 (LLM pandas code + E2B) ran on every non-empty result, including 1-row tie-out totals in hybrid. Extra LLM call + sandbox run per answer; playbook rule 2.1 says books never go to E2B.

## Scope
In:
1. `_require_sqlite_file(creds)` (missing or 0-byte file -> ConnectionNotFoundError "please re-upload"), called in `get_schema` and before every query in `run_sql_pipeline`.
2. Hybrid re-raises ConnectionNotFoundError; `/api/hybrid/chat` streams it as an error event, `/api/v1` hybrid returns 404.
3. `_should_run_analysis`: step 6.5 runs only when a chart type is picked, the question asks for a chart/plot/graph, or a non-tie-out query returns 2+ rows.
Out: persisting uploaded SQLite bytes across restarts (follow-up).

## Acceptance
- Unit tests `tests/unit/test_restart_and_chart_gate_2733.py`.
- Prod: hybrid San Diego tie-out shows no `chart`/`analysis_code` events; latency compared with the 2026-11-28 run (38-53 s).
