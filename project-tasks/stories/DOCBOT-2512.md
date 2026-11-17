# DOCBOT-2512 Promote-by-digest dev to prod
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 3 | devops | merge by 11-30 | PR #113 open |

## User story
As the founder approving prod deploys, I want prod to run the exact image digest that already passed the dev smoke, behind a manual approval, so that nothing reaches design partners that was not tested and no one can deploy prod by pushing a branch.

## Why now
Prod deploys must be controlled before the soak (2513) runs one promotion. The workflow is merged to dev (PR #113, `.github/workflows/deploy-aws-prod.yml`, `workflow_dispatch` with a 40-character SHA input).

## Scope
In:
- Prod deploy job uses GitHub Environment `aws-prod` (founder as required reviewer, deployment branch `main`).
- No rebuild: the prod job deploys the digest the dev run recorded for that SHA.
- OIDC trust on `environment:aws-prod`; prod ECR tags immutable.

Out:
- Automatic prod deploys, canary or blue/green.

## Acceptance criteria
- [ ] Given a SHA whose dev run passed smoke, when "Deploy AWS prod" is dispatched with it, then the job waits for founder approval and, after approval, the ECS task definition image equals the dev digest (same `sha256:`).
- [ ] Given a SHA with no passing dev run, when the workflow is dispatched, then it fails before requesting approval.
- [ ] Given a push to any branch, when workflows run, then no prod deploy starts.
- [ ] Given the prod ECR repository, when `describe-repositories` is run, then `imageTagMutability` is IMMUTABLE.

## Deliverable and approach
Code on `origin/dev`. The founder creates the `aws-prod` environment in 2576 (deployment branch `main`, corrected 2026-11-17). First real promotion happens during 2513.

## Test / verification
actionlint in CI; one approved promotion during the soak with digest match recorded; one negative dispatch with an untested SHA.

## Dependencies
Blocked by: 2508.
Blocks: 2513, 2519.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- The prod workflow runs from `main`; `dev` merges into `main` in 2579 before 2576.
