# DOCBOT-1626 SAML replay protection
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S3 | P1 | 1 | senior-dev | - | Open |

## User story
As a firm admin whose staff sign in through our IdP, I want DocBot to reject unsolicited and replayed SAML responses, so that a captured assertion cannot be reused to log in as one of my staff.

## Why now
Finding SEC-19 from the 10-24 audit. `api/auth_service.py:101` sets `"rejectUnsolicitedResponsesWithInResponseTo": False` and the ACS handler (`POST /api/auth/saml/acs`, `api/index.py:2585`) keeps no assertion-ID cache. Firm-tier buyers ask for SSO; this should be closed before any customer IdP is configured.

## Scope
In:
- Set `rejectUnsolicitedResponsesWithInResponseTo` to True and store the AuthnRequest ID at `/api/auth/saml/login` so `InResponseTo` can be checked.
- Assertion-ID replay cache in Postgres (ID, expiry from `NotOnOrAfter`), checked before session creation.
- Unit tests with a signed fixture.

Out:
- Per-org SAML config (2572).
- IdP-initiated SSO support.

## Acceptance criteria
- [ ] Given a valid SAML response, when it is posted to the ACS the first time, then login succeeds as today.
- [ ] Given the same response, when it is posted a second time, then the ACS returns 401 and no session row is created.
- [ ] Given a response whose `InResponseTo` does not match a request issued by `/api/auth/saml/login`, when posted, then it is rejected with 401.
- [ ] Given cache rows past their `NotOnOrAfter`, when the cleanup job runs, then they are deleted.

## Deliverable and approach
- Small table (for example `saml_assertion_ids`) registered like other service tables, plus helper functions in `api/auth_service.py`; route stays thin.
- Request ID stored in a short-lived signed cookie or table keyed by relay state.

## Test / verification
- `tests/unit/test_saml_replay.py`: first post ok, replay 401, unsolicited 401, expiry cleanup.
- CI green.

## Dependencies
Blocked by: none.
Blocks: 2572 (per-org SAML).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Any existing customer using IdP-initiated SAML that this would break? (None known.)
