# DOCBOT-2505 First deploy + smoke
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 3 | devops | 11-23 | App readiness merged (PR #105); smoke PR #112 open |

## User story
As the founder, I want the first AWS dev deploy checked by a repeatable smoke test, so that I know the AWS stack serves the real product (PDF chat, CSV, streaming, Chroma persistence) before any prod work builds on it.

## Why now
Gate G2 (dev smoke passed) is the exit of Wave A. The smoke script is merged to dev (PR #112, `infra/aws/scripts/smoke.sh`) and app readiness is merged (PR #105). What remains is running it after 2504 and re-authoring `infra/aws/docs/first-deploy.md`, which was lost to the `.gitignore` bug.

## Scope
In:
- Run "Deploy AWS dev" and `infra/aws/scripts/smoke.sh https://api-dev.docbot.space`.
- Manual checks the script cannot do: PDF Q&A with a citation, CSV query in E2B, SSE stream over 120 s, Chroma survives `--force-new-deployment`.
- Re-author `infra/aws/docs/first-deploy.md` with the manual checklist and results.
- Senior-dev fixes app blockers found by the smoke.

Out:
- Prod smoke (2513, 2519).

## Acceptance criteria
- [ ] Given the dev env is applied with secrets, when "Deploy AWS dev" runs on `dev`, then the workflow is green and prints the promotable digest notice.
- [ ] Given the deploy, when `smoke.sh https://api-dev.docbot.space` runs, then it exits 0 (health with db ok, CSP header, `/v1` 401, unsigned webhook rejected, demo init returns a session).
- [ ] Given a PDF uploaded on api-dev, when a question is asked, then the answer cites a page; and given a CSV, when a question is asked, then the E2B result returns.
- [ ] Given a session with a PDF, when `aws ecs update-service --force-new-deployment` completes, then the same session still answers from its vectors.
- [ ] Given CloudWatch Logs for the dev task, when searched for the canary patterns `gsk_`, `sk-`, `AIza` and the word `password=`, then there are no matches.

## Deliverable and approach
Devops runs the checks after the founder confirms 2504; results table goes into `infra/aws/docs/first-deploy.md` (tracked; `git check-ignore` prints nothing). An SSE check uses a long autopilot question through the ALB (idle timeout must exceed 120 s).

## Test / verification
`smoke.sh` exit code and output pasted (no secrets); manual check table with date and image digest.

## Dependencies
Blocked by: 2504 (prep: none).
Blocks: 2506, 2507, 2512 (promotable digest).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Backlog says PR #112 open; Plane notes it merged to dev. Update the backlog status at the next tracker pass.
