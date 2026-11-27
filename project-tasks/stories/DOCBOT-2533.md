# DOCBOT-2533 Design-partner offer + order form
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 GTM | 0 | - | P0 | 2 | founder | 11-24 | Open (pulled into EPIC-26, due 11-24) |

## User story
As a small advisory firm owner who just saw DocBot tie out my last closed month, I want a one-page order form with the price, term, what is included and a payment link, so that I can say yes on the call and pay without a procurement cycle.

## Why now
EPIC-26 gate needs 1 LOI or verbal DP yes by 12-01, and W4 (about 12-20) needs 3 paid DPs. A yes with no paper decays in days.

## Scope
In:
- Offer: $500/mo x 6 months, paid monthly in advance, founder-led onboarding, proof-run on one closed month before the first invoice, signed DPA (2531), 8-slot cap, offer ends when filled or 2027-01-15
- Number of client workspaces included stated as a number (per-client anchoring risk vs Fathom $450/mo for 25 companies)
- Stripe Payment Link (recurring $500/mo) or invoice template, tested end to end
- Short LOI variant for a verbal yes before the agreement (2549) is ready
Out:
- Full DP agreement (2549)
- Pro $79 or Firm self-serve billing (1810, 2569)

## Acceptance criteria
- [ ] Given the order form, when read, then it states $500/mo, 6-month term, client workspaces included (a number), proof-run before first invoice, 8-slot cap, end date 2027-01-15, and references the DPA
- [ ] Given the Stripe Payment Link, when the founder completes a test payment in Stripe test mode, then a $500 recurring subscription is created and visible in the dashboard, then refunded/cancelled
- [ ] Given the form, when compared with the locked pricing table (2026-10-30), then no other price, discount or tier appears
- [ ] Given the form, when read, then it says how client workspaces are delivered today (one org per client, set up by the founder) until DOCBOT-2562/2563 ship, with no promise of features not on main
- [ ] Given the LOI variant, when read, then it is one page, non-binding, and names the start condition (proof-run accepted)

## Deliverable and approach
`project-tasks/gtm/2533-dp-order-form.md` (export to PDF), `project-tasks/gtm/2533-dp-loi.md`. Stripe link created by the founder (agents do not touch Stripe config, per revenue plan section 8). Inputs: `market-gtm-plan.html` section 7 launch offer and section 5 pricing risks; revenue plan section 7 Firm gaps.

## Test / verification
Stripe test-mode transaction screenshot saved next to the form; PM checks form against locked pricing; one dry-read with a friendly CFO.

## Dependencies
Blocked by: 2601, 2531 (DPA referenced)  Blocks: 2549, 2551 (DP count)

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- How many client workspaces does $500/mo cover? Revenue plan implies about $50 to $75 per workspace, so 7 to 10; the plan also prices a 3+ workspace DP at $750. Founder to set the number.
- Offer ends 2027-01-15 but the Phase 1 gate is 02-28: logos 7+ after 01-15 must come from Firm or Pro. Confirm.
