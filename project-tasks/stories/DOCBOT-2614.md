# DOCBOT-2614 NEW: Idempotency-Key on cost-bearing /v1 POSTs
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P2 | 2 | senior-dev | - | New (2026-11-23) |

## User story
As an API client, I want to retry a POST safely, so that a network retry never double-charges me.

## Why now
Retries of cost-bearing POSTs double-charge today.

## Scope
In:
- `Idempotency-Key` on cost-bearing POSTs.
- `(api_key_id, key)` store with request hash and 24h TTL.

Out:
- Idempotency for GET or DELETE.

## Acceptance criteria
- [ ] Given a completed POST with an `Idempotency-Key`, when it is replayed with the same body, then the stored response is returned with `Idempotent-Replayed: true` and no charge.
- [ ] Given the same key with a different body, when it is sent, then the API returns 400.
- [ ] Given the first request still running, when the same key is sent, then the API returns 409.

## Deliverable and approach
One small store table plus a dependency used by the POST handlers. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
`tests/unit/test_v1_idempotency.py`: replay, mismatch 400, in-flight 409, no double charge.

## Dependencies
Blocked by: 3901.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- none
