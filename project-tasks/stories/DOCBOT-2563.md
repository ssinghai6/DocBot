# DOCBOT-2563 Workspace scoping in CallerScope
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S5 | - | 5 | senior-dev | - | To do |

## User story
As a fractional CFO, I want everything I do to happen inside the client workspace I picked, so that a question about client A can never retrieve client B's documents, connections or tie-outs.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. `CallerScope` (`api/utils/caller_scope.py:81`) has user, org, anon and admin only; `scope_filter`, `connection_access_clause` (`api/db_service.py:269`) and `run_row_accessible` (`api/resource_scope_service.py:70`) filter by user and org, and Chroma is keyed by session.

## Scope
In:
- `workspace_id` on `CallerScope`, read from a header or cookie and verified against the org (and 2568 grants once they exist).
- `scope_filter`, `connection_access_clause` and `run_row_accessible` add the workspace predicate when set.
- `owner_values` stamps `workspace_id` on insert.
- Chroma collection per workspace, building on the 2403 org prefix.

Out:
- Switcher UI (2564).
- Per-member grants (2568).

## Acceptance criteria
- [ ] Given a user with workspace A active, when they list sessions, then only sessions with workspace A are returned.
- [ ] Given a request with a workspace id from another org, when any scoped route runs, then it returns 404 and logs nothing of the foreign workspace.
- [ ] Given workspace A active, when a session is created, then the row and its Chroma collection carry workspace A.
- [ ] Given no workspace header (personal org users), when they use the app, then behavior is unchanged and all existing tests pass.
- [ ] Given two workspaces in one org, when the 2401 sweep is extended to workspaces, then every cross-workspace call returns 403 or 404.

## Deliverable and approach
- One change in `scope_filter`; avoid per-route edits.
- Header name `X-DocBot-Workspace`, cookie fallback; validation in the dependency that builds `CallerScope`.

## Test / verification
- Unit tests on `scope_filter` predicates; integration test with two workspaces; sweep extension.

## Dependencies
Blocked by: 2562, 2403
Blocks: 2564, 2568, 2575

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Admins: see all workspaces at once with no active workspace, or always one at a time?
