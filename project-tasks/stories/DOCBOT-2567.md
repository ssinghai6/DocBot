# DOCBOT-2567 Seats + offboarding
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S5 | - | 3 | senior-dev | - | To do |

## User story
As a firm admin, I want a seat limit and a remove-member action that cuts access at once, so that I pay for the seats I use and a departing employee loses access to client data the same day.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. There are no plan or seat columns (`api/org_service.py:103-104`), no member removal, and no session revocation on removal.

## Scope
In:
- Seat limit per org, enforced at invite accept.
- Remove member: membership removed, their browser sessions revoked (user_sessions), their API keys revoked (`revoke_api_key` in `api/api_key_service.py:274`).
- Audit events.

Out:
- Stripe seat billing (Firm bills per workspace, 2569).

## Acceptance criteria
- [ ] Given an org at its seat limit, when a new invite is accepted, then it fails with a clear message and no membership is added.
- [ ] Given a member is removed, when they make their next request, then it returns 401 and their API keys return 401.
- [ ] Given the last admin, when someone tries to remove them, then it is refused.
- [ ] Given removal, when it completes, then the audit_log shows who removed whom and when.

## Deliverable and approach
- Seat limit column on organizations via `ADD COLUMN IF NOT EXISTS`; set by operator for now.

## Test / verification
- Unit and integration tests for each AC.

## Dependencies
Blocked by: 2565, 2566
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Do the removed member's sessions transfer to an admin or stay owned and hidden?
