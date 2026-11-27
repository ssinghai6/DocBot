# DOCBOT-2618 NEW: /v1/audit-log export (audit:read scope)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | - | P3 | 2 | senior-dev | - | New (2026-11-23; later, not scheduled) |

## User story
As an org admin, I want to export the audit log over the API, so that our compliance tooling can pull it on a schedule.

## Why now
Buyers ask for audit export in security reviews; the CSV export exists in the web app only.

## Scope
In:
- `GET /v1/audit-log` with `audit:read`, live admin-minted keys only, cursor pagination, CSV option.

Out:
- Streaming to a SIEM.

## Acceptance criteria
- [ ] Given a live admin-minted key with `audit:read`, when it calls `GET /v1/audit-log`, then it gets its org's events paged by cursor.
- [ ] Given a test key or a key without `audit:read`, when it calls the route, then it gets 403.
- [ ] Given `format=csv`, when the route is called, then it returns the same columns as the web CSV export.

## Deliverable and approach
Thin handler over the `audit_service` export. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests for scope, key mode and pagination.

## Dependencies
Blocked by: 3901, 2907.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- none
