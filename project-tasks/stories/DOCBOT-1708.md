# DOCBOT-1708 Account deletion + export
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S5 | P1 | 2 | senior-dev | - | Open |

## User story
As a design partner ending a pilot, I want to export my data and then delete my account and everything it owns, so that I can show my clients their books and documents no longer sit with a vendor.

## Why now
The DP agreement (2549) and trust package (2531) promise deletion on exit. No account delete or export route exists on main; deletion is per session (`session_service.delete_owned_session`) or per connection only. First DP terms end around the 6-month mark, but the promise is made at signing.

## Scope
In:
- `GET /api/account/export`: zip of the caller's sessions (reuse `session_service.render_export`), connection names (never credentials) and audit rows for their org.
- `DELETE /api/account` with re-auth: hard-delete owned sessions, messages, db_connections, marketplace connections, Chroma collections (`vector_store.delete_store`), lineage; delete user sessions; audit row kept (append-only).
- Coordinate the vector purge with 2403 (org-namespaced collections) so one purge helper serves both.

Out:
- Firm workspace deletion (2575).
- Deleting a multi-member org (admin must remove members first).

## Acceptance criteria
- [ ] Given a logged-in user, when they call export, then they receive a zip with their sessions and chats, connection metadata without secrets, and audit rows.
- [ ] Given a user who re-authenticated within 15 minutes, when they call delete, then their sessions, messages, connections and Chroma collections are removed and a subsequent export or login returns nothing.
- [ ] Given the delete, when the audit log is queried by an operator, then an "account_deleted" row exists and earlier audit rows are unchanged.
- [ ] Given a user who did not re-authenticate, when they call delete, then it returns 401 and nothing is removed.
- [ ] Given two orgs, when a user in org A deletes their account, then no row or collection belonging to org B is touched.

## Deliverable and approach
- `api/account_service.py` with purge and export helpers; thin router in `api/routes/`; Zod schema on the frontend settings page.

## Test / verification
- `tests/integration/test_account_deletion.py` on SQLite plus a temp Chroma dir: export contents, purge completeness, two-org isolation, audit kept.
- CI green.

## Dependencies
Blocked by: none (2403 recommended first so the purge uses org-namespaced names).
Blocks: 2575 (reuses the purge helper).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Grace period (soft delete for 7 days) or immediate hard delete?
- Credits and Stripe customer on delete: cancel the subscription automatically?
- Points: export, re-auth before delete and the vector purge shared with 2403 make this closer to 3 unless 2403 lands first and provides the purge helper. Proposed to the founder, not changed here.
