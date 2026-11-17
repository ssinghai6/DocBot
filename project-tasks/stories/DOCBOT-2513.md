# DOCBOT-2513 7-day prod-env soak + cost check
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S1 | P1 | 2 | devops | - | Open |

## User story
As the founder deciding on cutover, I want the empty prod environment to run for 7 days under a synthetic health check with a measured idle cost, so that I know it stays up and what it costs before design partners depend on it.

## Why now
Gate G4 includes a 7-day soak at 99.5% or better; the clock starts as soon as 2576 applies prod in S1, and the cutover (2519) is mid-January. This replaced the older 14-day staging soak when dev became staging.

## Scope
In:
- Synthetic `/api/health` check every 5 minutes for 7 days against the prod ALB (for example a scheduled GitHub Action using `CONNECT_TO=<alb> smoke.sh https://api.docbot.space`, or a Route 53 health check).
- One promote-by-digest run (2512) during the soak.
- Prod idle run rate from Cost Explorer (`Environment=prod`) and months of credit runway vs expiry.

Out:
- Load testing; customer traffic (none until 2519).

## Acceptance criteria
- [ ] Given prod is applied with no customer traffic, when the synthetic check has run every 5 minutes for 7 consecutive days, then at least 99.5% of checks pass (no more than 10 of 2016 fail) and every failure has a cause noted.
- [ ] Given the soak window, when one approved promote-by-digest runs, then the check stays green apart from at most one failed probe during the deployment.
- [ ] Given 7 days of Cost Explorer data for `Environment=prod`, when the run rate is projected, then it is recorded against the $120/mo target, and the doc states months of credit runway at dev plus prod rates vs the credit expiry.
- [ ] Given the soak result, when written up in the tracker row and `infra/aws/docs/cost.md`, then it lists start date, end date, pass rate, promotion digest and run rate.

## Deliverable and approach
Devops sets up the check (code in `.github/workflows/` or Terraform), the founder approves any apply. Results are numbers only.

## Test / verification
Check history export (pass/fail counts); Cost Explorer totals; promotion run link.

## Dependencies
Blocked by: 2576, 2511, 2512.
Blocks: 2519 (G4 part).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Run the check from GitHub Actions (free, 5-minute cron is best-effort) or Route 53 health checks (about $0.50/mo each, exact interval)?
