# DOCBOT-2570 Audit log per workspace
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S6 | - | 3 | senior-dev | - | To do |

## User story
As a firm admin, I want the audit log filtered and exported per client workspace, so that I can hand a client the access history for their data alone.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Audit log is append-only with org_id at insert (`api/audit_service.py:60-92`), and `GET /admin/audit-log` is org-scoped with CSV export, but there is no workspace column or filter and invite, role and workspace events are not logged.

## Scope
In:
- `workspace_id` stamped on events by `log_event` from the active scope.
- `workspace_id` filter and per-workspace CSV export on the existing admin route.
- Events for invite, role change, workspace create/rename/archive/delete, grant changes.

Out:
- Changing the immutability trigger.

## Acceptance criteria
- [ ] Given events in workspaces A and B, when the admin filters by A, then only A's events and org-level events marked as such are returned.
- [ ] Given a CSV export for A, when opened in Excel, then it has only A's rows and the same columns as today plus workspace.
- [ ] Given an invite is created or a role changes, when the action completes, then an audit row exists with actor, target and timestamp.
- [ ] Given Postgres, when an UPDATE is attempted on audit_log, then the immutability trigger still blocks it.

## Deliverable and approach
- Read workspace from scope inside `log_event` so callers do not change.

## Test / verification
- Unit tests for filter and export; `@pytest.mark.postgres` trigger test unchanged.

## Dependencies
Blocked by: 2562, 2563
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should a client-facing export redact staff email addresses?
