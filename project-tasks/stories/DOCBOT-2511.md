# DOCBOT-2511 Alarms and notifications
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 3 | devops | merge by 11-30 | PR #109 open |

## User story
As the solo founder on call for prod, I want alarms on the failure modes that hurt users and on spend, so that I hear about an outage or a runaway bill by email before a design partner tells me.

## Why now
Prod has no human watching dashboards. The alarms are merged to dev (PR #109, `infra/aws/modules/app/alarms.tf`); 2576 and 2513 need them in place before the soak starts.

## Scope
In:
- Prod alarms to SNS email: ALB 5xx over 5% of requests, unhealthy targets, healthy targets below `desired_count`, RDS CPU over 80%, RDS free storage under 2 GiB, RDS connections over 60, EFS burst credits under 500 GiB.
- AWS Budgets on prod (150) and account (250) vs credits (in `infra/aws/account/budgets.tf`).
- Forced test of each alarm path.

Out:
- Paging (PagerDuty), uptime from outside AWS (2513 synthetic check), log-based alarms.

## Acceptance criteria
- [ ] Given prod is applied and the SNS subscription is confirmed, when `aws cloudwatch set-alarm-state --state-value ALARM` is run on each alarm, then an email arrives for each within 5 minutes.
- [ ] Given prod, when the API service is scaled to 0 for a drill, then the running-below-desired and unhealthy-target alarms fire within about 5 minutes and clear after scaling back to 1.
- [ ] Given the Budgets console, when prod and account budgets are listed, then prod is $150, account is $250, and alerts are set at 50, 80 and 100 percent.
- [ ] Given the alarm list in `infra/aws/README.md`, when compared with `aws cloudwatch describe-alarms --alarm-name-prefix docbot-prod`, then every listed alarm exists.

## Deliverable and approach
Code shipped on `origin/dev`. Remaining is the forced test after 2576, recorded in the README "Alarms" section with dates. The founder confirms the SNS email (checklist item 23).

## Test / verification
`terraform validate` in CI; forced-test log with alarm names and email receipt times.

## Dependencies
Blocked by: 2508.
Blocks: 2576, 2513.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should the 5xx alarm use 2% for 10 minutes to match the rollback trigger in 2518?
