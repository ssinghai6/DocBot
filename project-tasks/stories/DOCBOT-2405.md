# DOCBOT-2405 Engine pool keyed (org, conn); cost totals in Postgres
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-24 Stage A | 1 | S7 | P1 | 3 | senior-dev | - | Not on main |

## User story
As a firm admin connecting client databases, I want DB engines never shared across orgs and session LLM cost totals kept across restarts, so that one org can never ride another's connection pool and cost ceilings hold after a deploy.

## Why now
`_engine_cache` in `api/db_service.py:147` is an LRU keyed by the sync URL only, so two orgs with the same URL share an engine. `_session_cost_totals` in `api/utils/llm_provider.py` is an in-memory OrderedDict, so `SESSION_COST_CEILING_USD` resets on every deploy.

## Scope
In:
- Engine cache key `(org_id, connection_id)`; dispose on eviction kept.
- Session cost totals persisted in Postgres, reusing the DOCBOT-1801 credit ledger tables where possible.

Out:
- RLS, Alembic, pgvector, BYOK and KMS (Stage A exclusions: rebuilt in the platform).
- Changing the 20-engine LRU size.

## Acceptance criteria
- [ ] Given two orgs with connections to the same database URL, when both query, then two distinct engines exist in the cache.
- [ ] Given a run_id with $0.40 accumulated, when the API restarts and the run continues, then `get_session_cost_usd` returns at least $0.40 and the $0.50 ceiling still trips.
- [ ] Given 21 cached engines, when a new one is created, then the least recent is disposed as today.
- [ ] Given the change, when the unit suite runs, then no test hits a real database outside SQLite.

## Deliverable and approach
- Change `_get_or_create_engine(sync_url, ...)` signature to take the key; update callers in `db_service.py`.
- Cost: write-through upsert on `_accumulate_session_cost`, read on cache miss; keep the in-memory dict as a cache.

## Test / verification
- Unit tests for key isolation and eviction.
- Integration test on SQLite for cost persistence across a simulated restart (clear the dict).

## Dependencies
Blocked by: 2401
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Is a write per LLM call acceptable, or batch per run step?
