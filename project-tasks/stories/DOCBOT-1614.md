# DOCBOT-1614 Security review of new surfaces (G1)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S2 | P0 | 2 | PM | - | Open |

## User story
As the founder about to put design-partner ledgers on prod and cut over to AWS, I want a written security review of every surface added since the 10-24 audit (signup, email verification, Stripe webhook, credit ledger, Pro subscription), so that G1 closes on evidence instead of assumption.

## Why now
G1 is a precondition of 2519 (cutover). The yc-10-day-plan Gate B required "1614 sign-off" and it never happened. Since then billing (1810, PR #103), onboarding and tie-out routes (`api/routes/*_routes.py`) were added. Design partners send client books in December.

## Scope
In:
- Review signup, verification (`api/email_verification_service.py`), Stripe webhook and subscription (`api/routes/billing_routes.py`, PR #103 `api/subscription_service.py`), credit ledger (`api/credit_service.py`), onboarding and tie-out routes against the 7 security gates in yc-10-day-plan section 4.
- A findings table: gate, route, result, evidence (test name or file:line), owner ticket.
- Founder sign-off line.

Out:
- Fixing findings (each becomes its own ticket; P0 findings block G1).
- External pen test (EPIC-28, parked).

## Acceptance criteria
- [ ] Given the 7 gates (route manifest, two-org isolation, canary secret, webhook sig/replay/order, quota race, verified email, re-auth), when each new route is checked, then the findings table has a pass/fail and evidence for every route and gate pair.
- [ ] Given `tests/unit/test_route_guards.py`, when it is run on main, then every new route is either guarded or listed in `PUBLIC_ROUTES`, and the result is in the review.
- [ ] Given a failing gate, when the review is finished, then a ticket exists for it with priority, and P0 findings are listed as G1 blockers.
- [ ] Given the review document, when the founder reads it, then it has a dated sign-off line and G1 status is recorded in `founder-actions-aws.md` item 11.
- [ ] Given the document, when scanned, then it contains no secret values and makes no SOC 2 or "enterprise-grade" claim.

## Deliverable and approach
- `project-tasks/security-review-g1.md` (docs branch) with the findings table and sign-off.
- PM agent drives; senior-dev agent supplies test evidence; founder signs.

## Test / verification
- Linked test names for each pass; route-guard test output pasted.
- Founder sign-off date in the doc and in Plane.

## Dependencies
Blocked by: 1810 merged or explicitly excluded from the review scope.
Blocks: G1, 2519.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Review PR #103 before merge, or wait for merge and review main?
- Which findings severity blocks G1: only P0, or P0 and P1?
