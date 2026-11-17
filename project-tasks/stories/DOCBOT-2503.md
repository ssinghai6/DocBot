# DOCBOT-2503 Merge #86, account, credits, bootstrap
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 2 | founder | 11-20 | Open (Phase 0, Wave A) |

## User story
As the founder, I want the AWS account, region and credits confirmed and the Terraform state bootstrap applied, so that every later apply has a safe, versioned, locked state backend on an account the $5K credits actually cover.

## Why now
Wave A runs before Dec 1 and every apply (2504, 2576) needs the state bucket and lock table. #86 is already merged to `dev` (PR #86 merged, see 2501), so the remaining work is the founder's account decisions (checklist items 1 to 5) and the bootstrap apply (item 13). Plane shows In Progress via `STATE_FLOOR`.

## Scope
In:
- Founder checklist items 1 to 5 recorded in the "Answers" table of `project-tasks/exec-plan/founder-actions-aws.md` (account ID, region, credit expiry and eligible services, alarm email, budgets 60/150/250).
- `infra/aws/bootstrap` apply from the founder's laptop, state file kept outside the repo.
- Root MFA on, no IAM user access keys.

Out:
- The `account/` budgets apply and dev env apply (2504).
- Recording the promo code or any secret.

## Acceptance criteria
- [ ] Given the AWS console Billing > Credits page, when the founder records the credit account, expiry date, remaining amount and eligible-services gaps in founder-actions-aws.md, then ECS/Fargate, RDS, EFS, ELB, Secrets Manager, KMS, CloudWatch, ECR, S3 and public IPv4 are each marked covered or not.
- [ ] Given the bootstrap apply has run, when `aws s3api get-bucket-versioning` and `get-public-access-block` are run on the state bucket, then versioning is Enabled, default SSE is on and all 4 public-access blocks are true.
- [ ] Given the bootstrap apply has run, when `aws dynamodb describe-table` is run on the lock table, then the table exists.
- [ ] Given the IAM console, when the credential report is generated, then root MFA is on and no IAM user has an active access key.

## Deliverable and approach
Founder-only. The agent's part is limited to the checklist and README step 2 on `origin/dev` (`infra/aws/README.md`). The founder shares only non-secret facts (account ID, region, dates, yes/no) and the plan summary line. Agents never apply Terraform.

## Test / verification
Evidence pasted into the Answers table (no secrets): credits screenshot summary, bucket versioning output, lock table name, credential report summary.

## Dependencies
Blocked by: 2501, 2502.
Blocks: 2504, 2508 apply, 2576.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- One account for dev and prod (recommended while credits sit on one account) or two?
- If any service in the stack is not credit-eligible, what is its expected monthly cost?
