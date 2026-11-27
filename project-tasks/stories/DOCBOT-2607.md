# DOCBOT-2607 Raise ask + use of funds
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 1 | founder | 11-25 | In progress (EPIC-26, due by 12-01) |

## User story
As a pre-seed investor deciding whether to take a second meeting, I want the amount, instrument and an 18-month plan with milestones tied to revenue, so that I can see what my money buys and when the next round is justified.

## Why now
The ask slide (2602) and Q&A (2606) both depend on it, and it is due before deck v2.

## Scope
In:
- Amount and instrument (SAFE, post-money cap)
- Use of funds split: build (Firm workspace 2562-2575, single-tenant deploy), first hire, GTM, infra (AWS after $5K credits), compliance (SOC 2 only when a buyer asks)
- 18-month milestones: 2027-02-28 >= $5K MRR + 6 paying logos; $8.3K MRR stretch ($100K ARR run-rate); later milestones per founder
- Runway months at the chosen burn
Out:
- Cap table details (2605)
- Any milestone that depends on an enterprise contract in the Feb math

## Acceptance criteria
- [ ] Given the doc, when read, then amount, instrument and cap (or "cap TBD with lead") are stated in one line
- [ ] Given the use-of-funds table, when summed, then rows equal the raise amount exactly
- [ ] Given the milestones, when compared with the master backlog Phase 1 gate, then the first milestone is ">= $5K MRR + >= 6 paying logos by 2027-02-28" and no milestone counts the $30K pooled enterprise line before Phase 2
- [ ] Given the runway line, when recomputed from the burn table, then months of runway match within 1 month
- [ ] Given the doc, when checked, then SOC 2 spend is shown as triggered by a written buyer ask, matching the parking-lot rule

## Deliverable and approach
`project-tasks/preseed/2607-raise-and-use-of-funds.md`: 1) Ask line 2) Use-of-funds table (line, $, % , ticket range) 3) Monthly burn table 4) 18-month milestone table (date, metric, target, evidence source). Inputs: `revenue-100k-plan.html` MRR math, master backlog phases and parking-lot triggers, `MEMORY` note that AWS runs on $5K credits.

## Test / verification
Spreadsheet check of sums and runway; founder sign-off line.

## Dependencies
Blocked by: none  Blocks: 2602, 2605, 2606

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Raise amount and cap are founder decisions not recorded anywhere in the sources. Agent cannot draft numbers; founder supplies them.
