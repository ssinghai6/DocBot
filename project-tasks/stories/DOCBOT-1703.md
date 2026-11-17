# DOCBOT-1703 Password reset
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S6 | P1 | 2 | senior-dev | - | Open |

## User story
As a fractional CFO who signed up with email and password, I want to reset a forgotten password by email, so that I can get back into my clients' work without contacting the founder.

## Why now
No reset flow exists on main (`/api/auth/register` and `/api/auth/login` only). By S6 there are paying firms with several staff each, and every lockout is a manual support task. The email verification flow already has the token pattern to reuse.

## Scope
In:
- `POST /api/auth/password-reset/request` (always generic 200) and `POST /api/auth/password-reset/confirm` with Pydantic models, in `api/routes/`.
- Single-use, 30-minute, hashed token reusing the `email_verification_service.py` pattern (`generate_token`, `_hash_token`, consume once).
- On success: bcrypt new password, revoke all sessions for the user (same delete as `api/identity_service.py:382`), audit event via `audit_service.log_event`.
- Minimal frontend pages with Zod schemas.

Out:
- Password change while logged in.
- MFA.

## Acceptance criteria
- [ ] Given any email, registered or not, when a reset is requested, then the response is the same 200 body and timing does not reveal account existence.
- [ ] Given a valid token, when it is confirmed within 30 minutes with a new password, then the password changes, every existing session for the user is deleted and an audit row is written.
- [ ] Given a token that was already used or is older than 30 minutes, when confirmed, then it returns 400 and nothing changes.
- [ ] Given the token table, when inspected, then only the SHA-256 hash of the token is stored.
- [ ] Given repeated requests for one email, when more than the throttle allows, then extra emails are not sent.

## Deliverable and approach
- `api/password_reset_service.py` plus a thin router; reuse the verification email sender.

## Test / verification
- `tests/unit/test_password_reset.py`: generic response, success revokes sessions, reuse and expiry rejected, hash-only storage.
- CI green.

## Dependencies
Blocked by: none.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- OAuth-only users: send a "you sign in with Google/GitHub" email or stay silent?
