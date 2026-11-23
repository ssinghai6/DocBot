# DOCBOT-2617 NEW: Re-expose connectors in /v1 with /api parity
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | - | P3 | 3 | senior-dev | - | New (2026-11-23; later, not scheduled) |

## User story
As an API client, I want commerce connectors back in the public spec with the same routes as `/api/connectors*`, so that I can sync orders and financials from my own system.

## Why now
`/v1/connectors*` stays hidden from the spec until DOCBOT-2591 fixes org scoping; afterwards it should match `/api` (types, orders, financials, delete).

## Scope
In:
- Connectors back in `/v1/openapi.json` behind `connectors:manage`.
- Parity routes: types, orders, financials, delete.

Out:
- New connector types.

## Acceptance criteria
- [ ] Given DOCBOT-2591 is merged, when the spec is built, then the connector routes appear under `connectors:manage`.
- [ ] Given a key of org A, when it lists, syncs or deletes a connector of org B, then the API returns 404.
- [ ] Given each `/api/connectors*` capability, when the `/v1` routes are compared, then each one has a `/v1` route.

## Deliverable and approach
Reuse `connector_in_org` and the service calls behind the `/api` handlers. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Cross-org unit tests per route.

## Dependencies
Blocked by: 2591, 3901.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Is a buyer asking for this before it is scheduled?
