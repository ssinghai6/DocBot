# DOCBOT-2507 Dev scale-to-zero + cost evidence
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 2 | devops | 11-30 | PR #115 open |

## User story
As the founder paying from a fixed $5K credit pool, I want the dev environment to cost almost nothing when idle and a monthly cost record, so that the credits last through the prod cutover and beyond.

## Why now
Dev 24/7 costs about $60/mo against $43 with `desired_count = 0` (`infra/aws/README.md` "Cost"). The code is merged to dev (PR #115, budgets fixed to dev 60 / prod 150 / account 250 in 4885956), but `infra/aws/docs/cost.md` and `infra/aws/docs/dev-cost.md` were lost to the `.gitignore` bug and must be re-authored, and the 7-day evidence needs a live env.

## Scope
In:
- `desired_count` toggle in `infra/aws/envs/dev/variables.tf` (0 stops Fargate; ALB, RDS, EFS still bill) and the documented one-command toggle.
- Dev RDS stop/start documented (auto-restarts after 7 days).
- Re-author `infra/aws/docs/cost.md` and `dev-cost.md` with the Cost Explorer commands.
- 7 days of Cost Explorer data filtered on the `Environment` tag.

Out:
- Scheduled scaling automation beyond the toggle; prod cost (2513).

## Acceptance criteria
- [ ] Given dev is running, when the founder applies `desired_count = 0`, then the ECS service has 0 running tasks and `terraform plan` afterwards shows no drift on `desired_count`.
- [ ] Given the `Environment` cost-allocation tag is active, when Cost Explorer is filtered on `Environment=dev` for 7 consecutive days, then the projected monthly run rate is at most $50.
- [ ] Given `infra/aws/docs/cost.md` and `dev-cost.md`, when `git check-ignore` is run on both, then nothing is printed, and each lists the toggle, the RDS stop/start note and the Cost Explorer commands.
- [ ] Given the cost doc, when the founder reads it, then it states months of credit runway at the measured dev plus planned prod rate against the credit expiry date.

## Deliverable and approach
Devops re-authors the docs from the README "Cost" section and PR #115; the founder applies the toggle (agents never apply) and pastes the Cost Explorer totals (numbers only).

## Test / verification
`terraform validate` in CI; `aws ce get-cost-and-usage` output totals pasted in the doc with the date range.

## Dependencies
Blocked by: 2505.
Blocks: 2513 (runway statement uses the dev number).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Is the manual toggle enough, or should off-hours scheduled scaling be added (Terraform must then ignore `desired_count`)?
