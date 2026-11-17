# DOCBOT-2576 Founder-only: prod apply, prod secrets, aws-prod env
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S1 | P1 | 3 | founder | - | Open (Wave A, apply after 12-01) |

## User story
As the founder, I want the AWS prod environment applied with real prod secrets and a protected GitHub deploy environment, so that the soak clock (2513) can start while `api.docbot.space` still serves from Railway.

## Why now
First founder ticket after Dec 1 (S1, next-10 #7). The 7-day soak, restore drill, rehearsals and rotation test all need prod to exist, and the cutover is mid-January.

## Scope
In:
- `infra/aws/envs/prod` apply (plan reviewed: no destroys, no dev resources touched).
- ACM for `api.docbot.space` validated with only the validation CNAME; `api` itself still points at Railway.
- Every prod secret filled from Railway via the console; `DB_ENCRYPTION_KEY` copied unchanged.
- GitHub Environment `aws-prod` (founder as required reviewer, deployment branch `main`), repo variables `AWS_PROD_DEPLOY_ROLE_ARN`, `AWS_PROD_REGION`, `AWS_PROD_API_URL`, `AWS_PROD_ALB_DNS`, `AWS_PROD_FRONTEND_ORIGIN`.
- Confirm the prod SNS alarm subscription.

Out:
- DNS flip (2519); the Vercel `production` environment (never reused).

## Acceptance criteria
- [ ] Given the prod plan, when the founder shares its summary line, then it shows 0 to destroy and no change to `envs/dev` resources.
- [ ] Given the apply, when `aws acm describe-certificate` is run for `api.docbot.space`, then it is ISSUED while `dig +short api.docbot.space` still returns the Railway target.
- [ ] Given the prod secrets, when the founder compares the SHA-256 of the prod `DB_ENCRYPTION_KEY` with the Railway value locally, then they match, and an ECS task reaches RUNNING.
- [ ] Given GitHub settings, when `aws-prod` is opened, then it requires the founder's review and allows only branch `main`.
- [ ] Given the SNS email, when a 2511 forced alarm fires, then the founder receives it.

## Deliverable and approach
Founder-only (checklist items 20 to 23). Devops reviews the plan summary and `terraform output` names. Trim `app_secret_names` in `envs/prod/variables.tf` to what Railway prod actually sets before the apply (README "Secrets"). Agents never apply and never see values.

## Test / verification
Answers table updated with dates; `CONNECT_TO=<alb> smoke.sh https://api.docbot.space` passes against the ALB.

## Dependencies
Blocked by: 2508, 2511, 2579 (dev merged into main).
Blocks: 2513, 2510, 2515, 2516, 2509 live test.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- `main` must contain `infra/aws` and `deploy-aws-prod.yml` before this runs; owned by 2579.
