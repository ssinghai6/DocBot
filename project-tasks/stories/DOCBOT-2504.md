# DOCBOT-2504 Dev apply, ACM, api-dev, dev secrets
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 3 | founder | 11-21 | Open (Phase 0, Wave A) |

## User story
As the founder, I want the AWS dev environment applied behind `api-dev.docbot.space` with its own secrets and spend guardrails, so that we have a real AWS test env that mirrors prod without touching Railway or prod keys.

## Why now
The founder's ask for Phase 0 is a working AWS test env before Dec 1. Budget guardrails (`infra/aws/account/budgets.tf`) are merged to dev (PR #107), so the account apply and the dev apply can both run now. Plane shows In Progress via `STATE_FLOOR`.

## Scope
In:
- `infra/aws/account` apply (budgets: dev 60, prod 150, account 250, alerts 50/80/100%) and activating the `Environment` cost-allocation tag in Billing.
- `infra/aws/envs/dev` apply (README steps 3 to 8), ACM validation CNAME and `api-dev.docbot.space` CNAME at the DNS host.
- Fill 5 dev secrets in Secrets Manager: `groq_api_key`, `GEMINI_API_KEY`, `E2B_API_KEY`, a new dev `DB_ENCRYPTION_KEY`, a new `OWNER_KEY`.
- GitHub repo variables `AWS_DEV_DEPLOY_ROLE_ARN`, `AWS_DEV_REGION`, `AWS_DEV_API_URL`.

Out:
- Prod apply (2576), Vercel Preview (2506).

## Acceptance criteria
- [ ] Given the dev plan, when the founder shares the summary line with devops, then it shows 0 to destroy and no resources outside `infra/aws/envs/dev` and `infra/aws/account`.
- [ ] Given the apply, when `aws acm describe-certificate` is run for `api-dev.docbot.space`, then the status is ISSUED, and `dig +short api-dev.docbot.space CNAME` returns the ALB DNS name.
- [ ] Given the 5 dev secrets are filled, when ECS starts a task, then it reaches RUNNING (it fails closed while any secret is empty).
- [ ] Given the dev and prod `DB_ENCRYPTION_KEY`, when the founder compares their SHA-256 hashes locally, then they differ (dev never reuses the prod key).
- [ ] Given the `Environment` tag is activated, when the Budgets console is opened 24 hours later, then the per-env budgets show non-zero actual spend.

## Deliverable and approach
Founder-only steps from `project-tasks/exec-plan/founder-actions-aws.md` items 5, 14, 15, 16. Devops reviews the plan summary and the outputs list (names, ARNs, never values). Key generation commands are in `infra/aws/README.md` on `origin/dev`. Agents never apply Terraform and never see secret values.

## Test / verification
Answers table updated; `terraform output` names shared; first workflow run of "Deploy AWS dev" triggered (verified in 2505).

## Dependencies
Blocked by: 2503.
Blocks: 2505, 2506, 2507.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Is `docbot.space` DNS on Cloudflare? If so `api-dev` must be DNS-only (grey cloud) or SSE and ACM validation break.
- Fargate Spot for dev (checklist 9a, recommended yes)?
