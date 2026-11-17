# DOCBOT-2526 SOC 2 groundwork: control map + human access
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S6 | P3 | 3 | founder | - | Open (Wave C; pulled to Phase 1) |

## User story
As the founder fielding security questionnaires from firms, I want the AWS controls mapped to SOC 2 criteria and human access locked to SSO with MFA, so that when a buyer asks in writing for SOC 2 the evidence clock is already running and the honest answer is "in preparation" with proof.

## Why now
Pulled into Phase 1 as Wave C SOC 2 evidence (decision 1). The full SOC 2 program (EPIC-27) stays parked until a written ask; this story keeps that start cheap.

## Scope
In:
- Map CC6 (logical access), CC7 (operations and monitoring) and CC8 (change management) to existing AWS evidence: 2523 keys, 2524 trail and GuardDuty, 2511 alarms, 2512 approvals, 2510 restore drill.
- Humans access AWS only via IAM Identity Center with MFA; no IAM user access keys.
- Quarterly access review template.

Out:
- An auditor, a readiness tool purchase, any SOC 2 claim in copy.

## Acceptance criteria
- [ ] Given the control map document, when read, then each CC6, CC7 and CC8 criterion in scope lists the control, the evidence source and its owner.
- [ ] Given the IAM credential report, when generated, then no IAM user has an active access key and every human signs in through Identity Center with MFA.
- [ ] Given the access review template, when filled once for this quarter, then it lists every human principal and their permission set with a reviewer and date.
- [ ] Given public copy (landing, trust page 2531, deck), when searched for "SOC 2", then it only says not certified or in preparation, never compliant or certified.

## Deliverable and approach
Founder owns the access changes and the review. A PM or devops agent drafts the control map in `infra/aws/docs/soc2-control-map.md` from the merged Terraform and runbooks. Claims rule from the backlog: no "enterprise-grade" before a SOC 2 report.

## Test / verification
Credential report summary; filled review template; grep result on `src/` and trust docs.

## Dependencies
Blocked by: 2524 (and 2523 for CC6 key evidence).
Blocks: EPIC-27 pull (2701-2705, 2716) when a written ask arrives.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Identity Center needs AWS Organizations; enable it now on the single account, or wait for the PLAT-0400 layout (2530)?
