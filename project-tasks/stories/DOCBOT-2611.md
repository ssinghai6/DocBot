# DOCBOT-2611 NEW: /v1/chat and /v1/db/query
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P2 | 3 | senior-dev | - | New (2026-11-23) |

## User story
As an API client, I want plain document chat and database query over `/v1`, so that I do not have to route every question through hybrid.

## Why now
Only hybrid and autopilot exist on `/v1`. Doc-only and DB-only questions are the most common calls.

## Scope
In:
- Buffered wrappers over the `/api/chat` and `/api/db/chat` pipelines.
- Credit reservation (`chat` 20, `db_chat` 30); audit with `api_key_id`.
- PII masking contract test.

Out:
- Streaming (DOCBOT-2615).

## Acceptance criteria
- [ ] Given a session the key can see, when it calls `POST /v1/chat`, then it gets a buffered answer with citations.
- [ ] Given a connection the key can see, when it calls `POST /v1/db/query`, then it gets the answer, the SQL and the rows, read-only.
- [ ] Given seeded PII (an email and an SSN) in the data, when either route answers, then the PII is masked.
- [ ] Given a call, when it completes, then the audit row carries `api_key_id` and the reserved credits are settled.

## Deliverable and approach
Reuse the existing pipelines and the hybrid buffering helper; no new LLM calls. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
`tests/unit/test_v1_pii_masking.py` covering `/v1/chat`, `/v1/db/query`, `/v1/hybrid/chat`.

## Dependencies
Blocked by: 3901, 2599.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- none
