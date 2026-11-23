# DOCBOT-2598 NEW: Self-serve personal test API key for signed-in users
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S1 | P1 | 2 | senior-dev | - | New (2026-11-23) |

## User story
As a signed-in DocBot user, I want to mint my own test key in one click, so that I can try the API without asking an admin.

## Why now
Keys are admin-only today. The `/apis` portal (DOCBOT-3905) needs a session-cookie route that returns a personal test key.

## Scope
In:
- `POST /api/developer/test-key` and `DELETE /api/developer/test-key/{id}` in `api/routes/developer_routes.py`.
- Models in `api/developer_models.py` + Zod mirror `src/lib/schemas/developer.ts`.
- Signed-in, email-verified user (viewer+); preset scopes; 30-day expiry; max 2 active per user.

Out:
- Live key minting (stays admin-only).
- Portal UI (DOCBOT-3905).

## Acceptance criteria
- [ ] Given a signed-in, email-verified user, when they call `POST /api/developer/test-key`, then they get a `dk_test_` key once, with preset scopes and a 30-day expiry.
- [ ] Given a user with an unverified email, when they call the route, then it returns 403.
- [ ] Given a user with 2 active test keys, when they mint a third, then it is refused until one is revoked.
- [ ] Given the user's own test key id, when they call `DELETE /api/developer/test-key/{id}`, then the key returns 401 on its next use.

## Deliverable and approach
Thin router over `create_api_key(mode='test')` from DOCBOT-2907; the raw key is never stored or logged. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
`tests/unit/test_developer_routes.py`: 403 unverified, max 2, raw key returned once.

## Dependencies
Blocked by: 2907.
Blocks: 3905.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should viewers be allowed, or analyst+ only?
