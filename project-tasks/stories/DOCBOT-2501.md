# DOCBOT-2501 PR #86 pre-merge review
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 2 | devops | 11-19 | Open (Phase 0, Wave A) |

## User story
As the founder about to spend AWS credits, I want PR #86 (the AWS dev lift) reviewed against its own checklist before it merges, so that the first `terraform apply` creates only what was reviewed and nothing insecure slips into the account.

## Why now
Shipped record. PR #86 was the gate for every Wave A step (2502 to 2508). It was reviewed and merged to `dev` (merge commit 118d4c7 on `origin/dev`), so the backlog row is stale and Plane holds it at Done via `STATE_FLOOR`.

## Scope
In:
- Review of PR #86 (`infra/aws/bootstrap`, the original flat `infra/aws/dev`, `.github/workflows/deploy-aws-dev.yml`).
- `terraform fmt -check` and `validate`, checkov/tfsec triage, action SHA pinning check.

Out:
- Any apply (2503, 2504), the shared module refactor (2508).

## Acceptance criteria
- [ ] Given PR #86, when its review comment is read, then every checklist item is ticked or mapped to a follow-up story.
- [ ] Given `origin/dev`, when `terraform fmt -check -recursive infra/aws` and `terraform validate` run in each root, then both pass.
- [ ] Given the checkov/tfsec output on #86, when each finding is listed, then it is either accepted for dev with a reason or mapped to a Wave C story (2521 to 2525), and every GitHub action is pinned to a SHA that matches its tag.

## Deliverable and approach
Shipped: PR #86 review comment plus merge to `dev` (118d4c7). Findings that were deferred live in the Wave C rows (private subnets 2521, CMKs 2523, WAF 2522, CloudTrail 2524, IaC scanning 2525).

## Test / verification
`git log origin/dev --oneline | grep "#86"` shows the merge; CI on `dev` green for `infra/aws/**`.

## Dependencies
Blocked by: none.
Blocks: 2502, 2503, 2508.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Backlog and tracker rows still say Open; update both to Done (PR #86 merged to dev) in the next tracker pass.
