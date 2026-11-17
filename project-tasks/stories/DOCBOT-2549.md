# DOCBOT-2549 Design-partner agreement template
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 1 | S1 | P0 | 1 | founder | - | Open |

## User story
As a small advisory firm owner agreeing to pay $500/mo for six months, I want a short agreement that states scope, how my clients' data is handled and how either of us can exit, so that I can sign it without a lawyer of my own and show it to my clients if asked.

## Why now
W4 gate (about 12-20) is 3 paid DPs. The 2533 order form/LOI gets the yes; this agreement must be signed before the first invoice, and real client data starts flowing after it.

## Scope
In:
- Parties, term (6 months from proof-run acceptance), fee ($500/mo paid monthly in advance), client workspaces included (number from 2533)
- Scope: founder-led onboarding, proof-run on one closed month before first invoice, weekly feedback call, access to new features, case-study consent clause (opt-in, approval before publishing, for 2553)
- Data handling: incorporates the 2531 DPA; read-only access; QBO/Xero CSV exports (2577) rather than live DB credentials; deletion within N days of exit; sub-processor list
- Exit: either party with 30 days notice; immediate exit on data breach; export of their outputs on exit
- Limits: no SOC 2 report, stated plainly; liability cap; DocBot flags discrepancies, the CFO signs off
- Lawyer review note
Out:
- Enterprise MSA (parked EPIC-40)
- Stripe setup (2533)

## Acceptance criteria
- [ ] Given the template, when read, then it contains every Scope In heading and is no longer than 4 pages
- [ ] Given the fee and term clauses, when compared with the 2533 order form, then price, term, workspace count and end date match exactly
- [ ] Given the data clause, when read, then it references the 2531 DPA, states read-only access, a deletion window in days and the sub-processor list identical to /security
- [ ] Given the limits clause, when read, then it says DocBot has no SOC 2 report and that a human reviewer remains responsible for sign-off
- [ ] Given the template, when the founder's lawyer reviews it, then a dated "reviewed by counsel" note is in the file before the first signature

## Deliverable and approach
`project-tasks/gtm/2549-dp-agreement-template.md`, exported to PDF for e-signature. Agent drafts; founder edits; lawyer reviews ($200 legal budget line in the GTM plan). Inputs: 2533, 2531 DPA, `revenue-100k-plan.html` DP terms.

## Test / verification
Clause checklist against AC; side-by-side check with order form numbers; counsel note present.

## Dependencies
Blocked by: 2533, 2531  Blocks: first paid DP (W4), 2553

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Sprint S1 (after 12-01) but EPIC-26 aims for an LOI by 12-01. The 2533 LOI covers the gap; if a DP wants to pay before 12-02, this needs to move into Phase 0. Kept S1.
- Deletion window in days depends on the retention decision raised in 2531.
