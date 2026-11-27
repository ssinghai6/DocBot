# DOCBOT-2575 Client workspace offboarding
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S6 | - | 3 | senior-dev | - | To do |

## User story
As a firm admin ending a client engagement, I want to delete a client workspace and all its data in one action, so that I can confirm to the client that their data is gone.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Deletion today is per session or connection only (`delete_owned_session` in `api/session_service.py:153`); there is no grouping to delete as a unit.

## Scope
In:
- Delete workspace: purge sessions, messages, Chroma collections, db_connections, marketplace connections, lineage and artifacts for that workspace.
- Audit rows kept (append-only) with a deletion event.
- Confirmation step and an admin-only route.

Out:
- Backups: RDS PITR and EFS backups expire on their own schedule; state this in the confirmation text.

## Acceptance criteria
- [ ] Given a workspace with sessions, vectors and connections, when the admin deletes it, then a count query returns 0 rows for each table and no Chroma collection with its prefix remains.
- [ ] Given the deletion, when the audit log is read, then the workspace's earlier rows still exist plus one deletion event.
- [ ] Given a non-admin, when they call delete, then the API returns 403.
- [ ] Given a failure halfway, when the delete is retried, then it completes and is idempotent.
- [ ] Given the delete completes, when Stripe quantity is checked, then it has dropped as in 2569.

## Deliverable and approach
- Reuse the 2403 purge helper with the workspace prefix; DB deletes in one transaction, vectors after commit with retry.

## Test / verification
- Integration test with seeded workspace; count checks per table.

## Dependencies
Blocked by: 2563, 2403
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Soft-delete grace period (for example 7 days) before purge, or immediate?
