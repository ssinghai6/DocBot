# DOCBOT-2552 Enterprise target list + pooled one-pager
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 1 | S6 | P1 | 2 | founder | - | Open |

## User story
As a controller or VP Finance at a multi-entity mid-market company who saw DocBot through a fractional CFO, I want a one-page description of a pooled annual plan with an honest security posture, so that I can decide whether to start a security review without being misled about SOC 2.

## Why now
Phase 2 opens 03-01 if the Feb gate is met, and enterprise sales cycles are long. Building the list in S6 lets the first conversations start in March. It is out of the Feb math by decision 4.

## Scope
In:
- 20 named targets: multi-entity companies with recurring lender/board reporting, reached through a current DP or Firm customer where possible; name, company, role, entities, warm path, source link
- One-pager: what pooled means (shared credit pool across entities and users), price band $25K to $60K/yr, what is shipped (SSO, RBAC, encryption, append-only audit log, read-only DB access), what is not (no SOC 2 report; SOC 2 starts on a written ask; single-tenant deploy is roadmap), and the DP proof-run as the first step
Out:
- MSA, procurement pack, security questionnaire beyond 2531 (parked EPIC-40, EPIC-27)
- Any outreach before 03-01 unless a buyer asks first

## Acceptance criteria
- [ ] Given the target CSV, when counted, then it has 20 rows with a source link and a stated warm path or "cold"
- [ ] Given the one-pager, when read, then the price band is $25K to $60K/yr and no other price appears
- [ ] Given the security section, when read, then it says "no SOC 2 report yet" and never "enterprise-grade" or "SOC 2 compliant"
- [ ] Given each shipped capability listed, when traced, then it maps to the same evidence as the 2531 page
- [ ] Given the 2551 tracker, when a target replies with a written security ask, then the ask is logged there before any parking-lot story is pulled

## Deliverable and approach
`project-tasks/gtm/2552-enterprise-targets.csv` (gitignored) and `project-tasks/gtm/2552-pooled-one-pager.md` (export PDF). Inputs: locked pricing (Enterprise pooled $25K to $60K/yr), 2531 claims, master backlog section 7 trigger table.

## Test / verification
Banned-word grep; PM traces each claim to 2531 evidence; founder sign-off.

## Dependencies
Blocked by: 2531  Blocks: parked EPIC-40 stories (only on pull)

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Priority P1 looks high: the enterprise line is out of the Feb math, and the risk table says to defer 2552 first when the founder lane overloads. Suggest P2. Backlog value kept.
