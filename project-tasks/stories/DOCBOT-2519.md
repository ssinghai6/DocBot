# DOCBOT-2519 Founder-only: prod cutover
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S4 | P1 | 3 | founder | - | Open (Wave B; moved to mid-January, after 3 DPs stable) |

## User story
As the founder, I want to move `api.docbot.space` from Railway to AWS prod in one rehearsed window, so that DocBot runs on credit-funded infrastructure and design partners see no data loss.

## Why now
Scheduled mid-January (S4) once 3 design partners have been stable for 2 weeks (founder decision 5, 2026-11-17). Railway cost keeps running until this and 2520 are done; AWS credits have an expiry.

## Scope
In:
- Go/no-go signed (2514), write freeze announced to design partners.
- Final `pg_migrate.sh` dump and restore with counts matching; final Chroma copy (2516).
- DNS flip of `api.docbot.space` CNAME from Railway to the ALB; 1-hour watch.
- Prod smoke: health, PDF chat, CSV, hybrid, login, billing debit.

Out:
- Railway deletion (2520), any schema change during the window.

## Acceptance criteria
- [ ] Given G1 (1613 smoke log, 1614 sign-off) and G4 (2509, 2510, 2513, 2518) are closed, when the founder signs the go/no-go, then the runbook header holds the date and every checklist line is ticked.
- [ ] Given the write freeze, when the final dump and restore run, then `pg_migrate.sh` exits 0 with matching per-table counts and checksums.
- [ ] Given the CNAME flip, when 3 public resolvers are queried, then all return the ALB within 10 minutes.
- [ ] Given 1 hour after the flip, when the 2518 triggers are checked, then none fired and `smoke.sh https://api.docbot.space` plus the manual PDF, CSV, hybrid, login and billing-debit checks pass.
- [ ] Given cutover is complete, when design partners are notified, then the message goes out within 2 hours of the freeze end.

## Deliverable and approach
Founder executes (checklist item 27) with devops and senior-dev on standby in chat. Agents never run DNS, Railway or AWS console actions and never see connection strings.

## Test / verification
Cutover log appended to `infra/aws/runbooks/cutover.md` with timestamps, counts summary and smoke results.

## Dependencies
Blocked by: 2509, 2510, 2513, 2518, G1 (1613, 1614), 3 design partners stable for 2 weeks.
Blocks: 2520, 2523, 2524.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Window: a weekday evening Pacific or a Saturday morning? Design partners' month-end close dates should decide it.
