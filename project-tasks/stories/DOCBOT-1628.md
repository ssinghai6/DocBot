# DOCBOT-1628 Stop echoing exceptions to clients
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S3 | P1 | 1 | senior-dev | - | Open |

## User story
As a design partner connecting a client database, I want error messages that say what went wrong without exposing internals, so that driver errors, hostnames or SQL fragments from my systems never appear in the browser or in a screenshot I share.

## Why now
Finding SEC-23. `safe_error_message()` exists in `api/utils/exceptions.py:17`, but about 19 `detail=str(exc)` or f-string echo sites remain in `api/index.py` (for example 1845, 2125, 2162, 2605, 3343, 3925, 3934 "Connection test failed: {exc}", 3968 "Sync failed: {exc}") and 2 in `api/db_service.py` (2028, 2045 `sql_gen_error: {exc}`). Client data starts flowing in December.

## Scope
In:
- Replace every exception echo in `api/` with `safe_error_message()` or a fixed message; keep the full error in server logs only.
- A unit test that greps `api/` for `detail=str(` and `detail=f"...{exc` patterns and fails if any remain (allowlist for deliberate user-input validation messages).

Out:
- Changing status codes or response schemas.

## Acceptance criteria
- [ ] Given the code on main after the change, when `git grep -n "detail=str(e" -- api` and the f-string pattern are run, then only allowlisted validation sites remain.
- [ ] Given a DB connection test that fails with a driver error containing a hostname, when the route returns, then the response body contains no hostname, driver text or stack trace.
- [ ] Given any replaced site, when the error happens, then the full exception is still logged server-side with its type.
- [ ] Given the guard test, when someone adds a new echo site, then CI fails.

## Deliverable and approach
- Mechanical edit across `api/index.py`, `api/db_service.py`, `api/routes/*`; guard test in `tests/unit/`.

## Test / verification
- New guard test plus one route test with a forced driver error.
- CI green.

## Dependencies
Blocked by: none.
Blocks: 1614 (finding closed).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Which validation errors (for example invalid date input) are safe to echo and belong on the allowlist?
- Points: about 21 echo sites remain on `main` (19 in `api/index.py`, 2 in `api/db_service.py`) plus a CI guard against new ones; 2 is more realistic than 1. Proposed to the founder, not changed here.
