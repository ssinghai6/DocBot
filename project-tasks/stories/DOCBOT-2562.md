# DOCBOT-2562 Client workspace data model
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S4 | - | 5 | senior-dev | - | To do |

## User story
As a fractional CFO serving several clients, I want each client to be its own workspace inside my firm, so that one client's documents and books never mix with another's.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. The org is the only tenant unit today (`api/org_service.py:100-127`); nothing exists below it.

## Scope
In:
- `client_workspaces` table (id, org_id, name, created_at, archived_at).
- Nullable `workspace_id` on sessions, db_connections, marketplace connections, answer_lineage and audit_log via `ADD COLUMN IF NOT EXISTS` in `init_db()`.
- Service module `api/workspace_service.py` with create/get/list helpers (no routes yet).

Out:
- Filtering by workspace (2563), UI (2564), billing (2569).
- RLS or Alembic.

## Acceptance criteria
- [ ] Given an existing database, when the app boots twice, then the table and columns exist and the second boot makes no changes.
- [ ] Given an org, when `create_workspace(org_id, name)` runs, then a row is returned and a duplicate active name in the same org is rejected.
- [ ] Given rows created before this change, when they are read, then `workspace_id` is NULL and every existing test passes unchanged.
- [ ] Given org B, when it lists workspaces, then org A's workspaces are not returned (added to the 2401 sweep).

## Deliverable and approach
- Table registration follows `register_organizations_table` in `api/org_service.py`.
- Index on (org_id, archived_at).

## Test / verification
- `tests/unit/test_workspace_service.py` on SQLite; 2401 sweep extended.

## Dependencies
Blocked by: 2401, 2402
Blocks: 2563, 2569, 2570

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Unique name per org including archived workspaces, or only active ones?
