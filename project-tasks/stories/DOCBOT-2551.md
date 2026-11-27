# DOCBOT-2551 Weekly gate tracker
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 0 | S0 | P0 | 1 | PM | starts 11-17, every Fri | Starts now (2026-11-17) |

## User story
As the founder deciding every Friday whether to keep selling, change the hook or stop feature work, I want one sheet that compares this week's calls, demos, design partners, Pro subs, packs and MRR with the gate for that week, so that a miss is visible in days, not at the end of February.

## Why now
The plan has hard gates (12-01, W4, W8, W12, 02-28) and a kill rule (> 50% miss means reprioritize). It is also the log that decides parking-lot pulls: a written or paid buyer ask must be recorded here before any EPIC-27..40 story is pulled.

## Scope
In:
- Weekly rows (Fridays 11-20, 11-27, 12-04 ... 02-26, plus 02-28): contacted, replies, calls booked, calls held, quotable pains, free tie-outs run, LOIs/verbal yes, paid DPs, Pro subs, pack revenue (trailing 30 days), MRR (from Stripe), paying logos
- Gate columns: 12-01 (150 contacted, >= 12 booked, >= 6 held, >= 3 pains, 1 LOI); W4 about 12-20 (3 paid DPs); W8 about 01-06 (>= $2.5K MRR or 4 logos); W12 02-11 (>= $4K MRR); 02-28 (>= $5K MRR + 6 logos; stretch $8.3K)
- Status per gate: on track / miss < 50% / miss > 50% (triggers reprioritization review)
- Funnel rates: DM to call, call to free tie-out, tie-out to paid
- Buyer-ask log: date, buyer, document (email, questionnaire, RFP, order form), ask, stories pulled
Out:
- CRM tooling
- Automated Stripe sync (manual copy is fine at this volume)

## Acceptance criteria
- [ ] Given the tracker on 2026-11-20, when opened, then the first Friday row is filled and every gate above has a column with its target
- [ ] Given any Friday row, when MRR is compared with the Stripe dashboard that day, then the numbers match
- [ ] Given a gate missed by more than 50%, when the row is saved, then the status cell reads "REPRIORITIZE" and a one-line decision is recorded within 7 days
- [ ] Given a buyer's written security or SOC 2 ask, when it arrives, then it is logged with date, buyer and document before any EPIC-27..40 story moves out of the parking lot
- [ ] Given the 2602 traction slide and 2605 metrics file, when compared with the latest row, then they match

## Deliverable and approach
`project-tasks/gtm/2551-gate-tracker.md` (markdown tables: weekly rows, gates, buyer-ask log), or a Google Sheet with the same columns if the founder prefers; one owner updates it each Friday. Inputs: 2548 call notes, Stripe dashboard, 2546 list status, master backlog section 2 gates and section 9 risks.

## Test / verification
Friday review: founder reads the row; PM checks totals against call notes count and Stripe; gate status formulas or manual marks reviewed once per month.

## Dependencies
Blocked by: none  Blocks: 2602 (traction), 2605 (metrics), 2535 (funnel), every parking-lot pull

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Interim gates W8 and W12 are derived from the $5K target (decision D11); use them unless the founder changes them.
- Lane: tracker says owner "Orchestrator", backlog says PM. Kept PM.
