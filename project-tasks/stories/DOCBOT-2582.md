# DOCBOT-2582 NEW: Remove "enterprise" claims from landing page
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P1 | 1 | frontend | 2026-11-19 | In progress (2026-11-17) |

## User story
As the founder, I want the landing page to make only claims the product backs, so that an investor or buyer checking the site before Dec 1 finds nothing we can't defend.

## Why now
The EPIC-26 claim audit bans "enterprise-grade" wording (no SOC 2). `/` still says "Enterprise-ready from day one", "Enterprise-grade intelligence", "Enterprise Security" and "Enterprise Document Intelligence".

## Scope
In:
- In `src/app/page.tsx`: CTA line -> "SSO, RBAC and audit log built in"; features heading -> "One analyst across every data source"; feature card -> "Security Built In"; hero badge -> "Financial Document Intelligence".

Out:
- Wider landing copy realignment (DOCBOT-2537, waits on the wedge line).

## Acceptance criteria
- [ ] Given the `/` page on main, when searched for "enterprise", then it does not appear.
- [ ] Given the final CTA, when rendered, then it shows "Free to start" and "SSO, RBAC and audit log built in".

## Deliverable and approach
Four-line copy change.

## Test / verification
`grep -ni enterprise src/app/page.tsx` returns nothing; `npm run lint` and `npx tsc --noEmit` pass.

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
- None.
