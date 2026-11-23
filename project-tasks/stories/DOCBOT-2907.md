# DOCBOT-2907 API key modes (test/live), expiry, owner, scopes + service accounts
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-29 Enterprise Identity and Access | 1 | S1 | P1 | 3 | senior-dev | - | New (scheduled from parking lot) |

## User story
As an org admin, I want API keys that carry a mode, an expiry, an owner and narrow scopes, so that a test key can never touch live data and a leaked key stops working on its own.

## Why now
Test keys and the sandbox (2597, 2598) need `mode` and `expires_at`. The original 2907 scope (scopes + service accounts) is kept.

## Scope
In:
- `api_keys` gains `mode` (`test`/`live`), `expires_at`, `created_by`; `create_api_key(mode=...)` emits `dk_test_` or `dk_live_`.
- Existing rows backfill `mode=live`.
- Scopes `sessions:read`, `sessions:write`, `query:run`, `usage:read`; `hybrid:read` kept as an alias.
- Audit events for mint, revoke, expire; a service account survives member removal.

Out:
- Self-serve minting (2598).

## Acceptance criteria
- [ ] Given `create_api_key(mode='test')`, when it runs, then the key starts with `dk_test_`.
- [ ] Given an expired key, when it is used, then the API returns 401 `expired_key`.
- [ ] Given keys issued before this change, when the migration runs, then they are `live` and keep working.
- [ ] Given a key with `hybrid:read`, when it calls `/v1/hybrid/chat`, then it is allowed as before.
- [ ] Given a service-account key, when the member who made it is removed, then the key keeps working.

## Deliverable and approach
Additive migration in `init_db()`; extend `api/api_key_service.py`. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Extend `test_api_key_service.py`: prefix, expiry 401, backfill, alias scope.

## Dependencies
Blocked by: 1530, 3901.
Blocks: 2597, 2598, 2599, 2618.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- none
