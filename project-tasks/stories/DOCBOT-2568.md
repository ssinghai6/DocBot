# DOCBOT-2568 Per-workspace access grants
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S6 | - | 3 | senior-dev | - | To do |

## User story
As a firm admin, I want to grant each staff member access to only the clients they work on, so that juniors never see clients they are not assigned to.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Roles are org-wide viewer/analyst/admin (`api/rbac_service.py:44-48`); there are no per-workspace grants.

## Scope
In:
- `workspace_grants` (workspace_id, user_id, granted_by, created_at).
- Non-admins see and can activate only granted workspaces; admins see all.
- Admin API to grant and revoke; audit events.

Out:
- Per-workspace roles (grant is access only; role stays org-wide).

## Acceptance criteria
- [ ] Given an analyst granted workspace A only, when they list workspaces, then only A appears.
- [ ] Given the same analyst, when they send workspace B as active, then every scoped route returns 404.
- [ ] Given an admin, when they revoke a grant, then the next request for that workspace returns 404.
- [ ] Given grant changes, when they happen, then audit_log rows record them.

## Deliverable and approach
- Check the grant where 2563 verifies the active workspace; one place.

## Test / verification
- Integration tests; sweep extended with a non-granted member.

## Dependencies
Blocked by: 2563, 2565
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should new workspaces auto-grant their creator?
