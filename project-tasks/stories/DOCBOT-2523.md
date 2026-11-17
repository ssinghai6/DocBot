# DOCBOT-2523 KMS CMKs for remaining services
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S5 | P2 | 3 | devops | - | Open (Wave C) |

## User story
As a firm admin answering a client's security questionnaire, I want DocBot's secrets, logs, images and backups encrypted with keys DocBot controls and rotates, so that I can answer the encryption-at-rest questions with evidence instead of "AWS default".

## Why now
RDS and EFS already use one CMK from 2508. The rest (Secrets Manager, CloudWatch Logs, ECR, S3 buckets) still use AWS-managed keys. It is SOC 2 groundwork evidence for 2526 and is kept in Phase 1 (founder decision 1) even though EPIC-27 is parked.

## Scope
In:
- Customer-managed KMS keys with annual rotation and least-privilege key policies for Secrets Manager, CloudWatch log groups, ECR repositories and S3 (state, migration bucket) in `infra/aws/modules/app` and `infra/aws/account`.
- Evidence saved: `aws kms describe-key` and `get-key-rotation-status`.

Out:
- RDS and EFS (done in 2508), customer-managed keys per tenant (CMEK, 2807, parked).

## Acceptance criteria
- [ ] Given prod after the founder applies, when `aws secretsmanager describe-secret` is run on each `docbot-prod/*` secret, then `KmsKeyId` is the new CMK.
- [ ] Given the prod log groups and ECR repository, when described, then each shows the CMK ARN.
- [ ] Given each new CMK, when `get-key-rotation-status` is run, then `KeyRotationEnabled` is true.
- [ ] Given each key policy, when read, then only the ECS task role, the execution role, the redeploy role and the founder's admin role have use permissions, and no `kms:*` grant to `*` exists.
- [ ] Given the change, when the API is smoke-tested after a forced redeploy, then `smoke.sh` passes (tasks can still read secrets).

## Deliverable and approach
Devops PR on `infra/aws/**`; the founder reviews the plan (secrets re-encrypt in place, ECR repository encryption may require recreate: plan must show no replace on prod ECR, or a documented re-push). Agents never apply.

## Test / verification
`terraform validate` in CI; describe outputs saved in `infra/aws/docs/` as evidence (ARNs only).

## Dependencies
Blocked by: 2519.
Blocks: 2526.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- ECR encryption config is immutable on an existing repository; create a new CMK-encrypted repo and re-push the current digest, or accept AES256 for ECR?
