# DOCBOT-2564 Workspace switcher + CRUD
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S5 | - | 3 | senior-dev | - | To do |

## User story
As a firm admin, I want to create, rename, archive and switch client workspaces in the app, so that I can set up a new client in under a minute without asking support.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Onboarding can only rename the personal org today (`api/routes/onboarding_routes.py:49-70`).

## Scope
In:
- `api/routes/workspace_routes.py` (APIRouter): create, list, rename, archive; admin-only for writes; Pydantic models.
- Sidebar switcher in `src/components/Sidebar.tsx` with Zod schemas; selection stored and sent as the 2563 header.

Out:
- Delete with data purge (2575).
- Billing changes on add/archive (2569).

## Acceptance criteria
- [ ] Given a firm admin, when they create a workspace named 'Acme', then it appears in the switcher and becomes active.
- [ ] Given an analyst or viewer, when they call create, rename or archive, then the API returns 403.
- [ ] Given an archived workspace, when the user opens the switcher, then it is hidden by default and its data is read-only.
- [ ] Given every new route, when `tests/unit/test_route_guards.py` runs, then each has a guard and scope.
- [ ] Given the frontend, when `tsc` and lint run, then there are no `any` types in the new code.

## Deliverable and approach
- Routes call `api/workspace_service.py`; `require_role('admin')` for writes.
- Audit events for create/rename/archive (feeds 2570).

## Test / verification
- Route unit tests; Zod schema test; manual check on the Vercel preview.

## Dependencies
Blocked by: 2563
Blocks: 2569 (archive affects quantity)

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should creating a workspace be blocked when the firm is over its paid quantity, or bill on the next cycle?
