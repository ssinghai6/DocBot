# DOCBOT-2565 Org invites (was 1709)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S4 | - | 5 | senior-dev | - | To do |

## User story
As a firm admin, I want to invite staff by email with a role, so that my team works in the firm's org instead of each in their own personal org.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Every first login creates a personal org (`api/auth_service.py:350-400`, `create_personal_org` in `org_service.py:154`), and there is no invite flow; it is first on the 2561 critical path.

## Scope
In:
- `org_invites` table: hashed token, email, role, expiry, created_by, accepted_at, revoked_at.
- Admin create/list/revoke routes; accept route that joins the firm org.
- Invite email through the existing verification email sender (`api/email_verification_service.py`).
- Audit events for create, revoke, accept.

Out:
- Membership in several orgs at once (2566): until then accepting moves the user.
- Seat limits (2567).

## Acceptance criteria
- [ ] Given an admin, when they invite an email with role analyst, then one row exists with only a SHA-256 of the token and an email is sent with a link.
- [ ] Given a valid unexpired token, when the invited user (email matching, verified) accepts, then their org becomes the firm org with the invited role and the token cannot be reused.
- [ ] Given an expired, revoked or reused token, when it is accepted, then the API returns a generic 400 and nothing changes.
- [ ] Given a logged-in user whose email differs from the invite, when they accept, then it is refused.
- [ ] Given any invite action, when it completes, then an audit_log row with the firm's org_id exists.

## Deliverable and approach
- Token pattern copied from `mint_verification_token`/`consume_verification_token`.
- Routes in `api/routes/invite_routes.py`, admin via `require_role`.

## Test / verification
- Unit tests for token lifecycle; integration test for accept moving `users.org_id`; route-guard test.

## Dependencies
Blocked by: none
Blocks: 2566, 2567, 2568

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- When a user with data in a personal org accepts, do their sessions move with them or stay behind? Default: stay; 2566 resolves it.
