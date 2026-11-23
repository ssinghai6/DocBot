# DOCBOT-2616 NEW: /v1 tie-out workpaper download
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P3 | 1 | senior-dev | - | New (2026-11-23) |

## User story
As an API client, I want to download the tie-out workpaper for a run, so that I can file it without opening the web app.

## Why now
The export exists in `tieout_routes`; only the key-scoped route is missing.

## Scope
In:
- `GET /v1/lineage/{run_id}/tieout.{xlsx,pdf}` reusing the `tieout_routes` export with the key-org `run_row_accessible` check.

Out:
- New export formats.

## Acceptance criteria
- [ ] Given a run the key can see, when it requests `tieout.xlsx` or `tieout.pdf`, then it gets the same file as the web export.
- [ ] Given a run of another org, when it is requested, then the API returns 404.
- [ ] Given a key without `lineage:read`, when it calls the route, then it gets 403.

## Deliverable and approach
Thin handler over the existing export function. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests for scope and cross-org access.

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
