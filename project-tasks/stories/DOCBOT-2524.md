# DOCBOT-2524 CloudTrail, GuardDuty, Config
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S5 | P3 | 3 | devops | - | Open (Wave C) |

## User story
As the founder who will one day face a SOC 2 auditor or a client's security reviewer, I want a tamper-evident record of every AWS API call plus threat detection and config history, so that I can show who changed what and that someone would notice an intrusion.

## Why now
Kept in Phase 1 as SOC 2 evidence (2526 depends on it). It also unblocks automatic redeploys on app-secret changes in 2509, because `PutSecretValue` EventBridge events only arrive once a trail exists (`infra/aws/modules/app/rotation.tf`).

## Scope
In:
- Multi-region CloudTrail with log file validation to an S3 bucket with Object Lock.
- GuardDuty enabled with findings to the 2511 SNS topic.
- AWS Config with a starter conformance pack.

Out:
- Security Hub, a SIEM, organization-level trails (2530).

## Acceptance criteria
- [ ] Given the trail is applied, when the founder runs a harmless API call (for example `aws s3 ls`), then the event appears in CloudTrail within 15 minutes and `validate-logs` reports the digest valid.
- [ ] Given the trail bucket, when `get-object-lock-configuration` is run, then Object Lock is enabled with a retention period.
- [ ] Given GuardDuty, when a sample finding is generated (`create-sample-findings`), then an SNS email arrives.
- [ ] Given AWS Config, when the conformance pack is deployed, then the compliance dashboard lists its rules with results.
- [ ] Given an app secret is updated, when EventBridge is observed, then the 2509 redeploy state machine starts without the manual step.

## Deliverable and approach
Devops Terraform in `infra/aws/account` (account-wide services). Monthly cost estimate added to `infra/aws/docs/cost.md` before the founder applies (Config rules are billed per evaluation).

## Test / verification
`terraform validate` in CI; evidence commands and outputs (IDs only) in the docs.

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
- Config starter pack scope: all resources (higher cost) or only the resource types in the DocBot stack?
