# DOCBOT-2402 org_id NOT NULL for user-owned rows
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-24 Stage A | 1 | S4 | P0 | 3 | senior-dev | - | Not on main |

## User story
As the founder, I want every row owned by a logged-in user to carry an org_id, so that org-scoped filters can never miss a row because its org is NULL.

## Why now
`org_id` is nullable on sessions, audit_log, users and db_connections (`api/index.py` table definitions) and on `connector_store.py`. The 2561 gap analysis found pre-DOCBOT-1524 audit rows with NULL org. `scope_filter` matches on `org_id`, so a NULL row is invisible to admins and a risk for future workspace filters (2563).

## Scope
In:
- Idempotent `ADD CONSTRAINT`/check run from `init_db()` (the codebase pattern, no Alembic): user-owned rows (`user_id IS NOT NULL`) must have `org_id`.
- Anonymous rows tagged with a `source` value instead of a fake org.
- A legacy-row check script that reports and backfills NULL org rows from `users.org_id`.

Out:
- RLS, Alembic, pgvector, BYOK and KMS (Stage A exclusions: rebuilt in the platform).
- Changing the anonymous ownership model (`anon_owner`).

## Acceptance criteria
- [ ] Given a database with user-owned rows that have NULL org_id, when the check script runs in report mode, then it prints counts per table and exits non-zero.
- [ ] Given the same database, when the backfill runs twice, then the second run changes 0 rows and every user-owned row has the owner's org_id.
- [ ] Given the migration has run, when code tries to insert a row with `user_id` set and `org_id` NULL, then the insert fails with a constraint error caught in a unit test.
- [ ] Given anonymous rows, when the migration runs, then they keep `org_id` NULL, carry a `source` tag, and stay invisible to org admins (`scope_filter` rule from DOCBOT-2317).
- [ ] Given prod data on Railway, when the report runs before deploy, then the founder has the NULL counts recorded in the PR.

## Deliverable and approach
- DDL strings next to `ANON_OWNER_COLUMN_MIGRATIONS` in `api/utils/caller_scope.py` or a new list in `api/org_service.py`; Postgres uses a `CHECK (user_id IS NULL OR org_id IS NOT NULL)` added `NOT VALID` then validated after backfill.
- Script in `scripts/` reuses `get_user_org` from `api/org_service.py`.
- SQLite path: skip the constraint, keep the check script.

## Test / verification
- `tests/integration/` test on SQLite for the backfill idempotency.
- `@pytest.mark.postgres` test for the CHECK constraint.
- Report output from prod attached to the PR (counts only).

## Dependencies
Blocked by: 2401
Blocks: 2403, 2562

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Which tables count as user-owned beyond sessions, db_connections, audit_log, marketplace connections, lineage and traces?
- Delete or keep orphaned rows whose user no longer exists?
