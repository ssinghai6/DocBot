# DOCBOT-2613 NEW: Per-key API usage metering + GET /v1/usage
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P2 | 3 | senior-dev | - | New (2026-11-23) |

## User story
As an org admin, I want to see credits spent per API key, so that I can control spend and spot a leaked key.

## Why now
`api_key_id` lives only in audit metadata today; there is no balance or usage route.

## Scope
In:
- `api_key_id` on usage events / ledger metadata for every `/v1` call.
- `GET /v1/usage` (by route + paged events).
- `X-DocBot-Credits-Charged` header.
- Per-key spend in `/apis` and the admin key list.

Out:
- Invoices or alerts.

## Acceptance criteria
- [ ] Given any cost-bearing `/v1` call, when it completes, then the response has `X-DocBot-Credits-Charged` and the ledger row carries `api_key_id`.
- [ ] Given a key with `usage:read`, when it calls `GET /v1/usage`, then it sees totals by route and paged events for that key only.
- [ ] Given the admin key list, when an admin opens it, then each key shows its spend.

## Deliverable and approach
Add the field where `RequestMeter` settles; reuse `api/utils/billing_context.py`. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests for attribution, header and isolation.

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
