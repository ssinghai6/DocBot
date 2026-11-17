# DOCBOT-1613 Prod smoke regression (G1)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S2 | P0 | 2 | ai-engineer | - | Open |

## User story
As the founder deciding whether to cut over to AWS, I want a scripted smoke run against prod that covers every user-facing path, so that I have a dated log proving the product works before G1 is closed and design-partner data moves.

## Why now
G1 (EPIC-16 security gate) is required before the prod cutover 2519 in mid-January, and the founder checklist item 11 asks for "1613 prod smoke log". Design partners start using prod in December, so a regression on PDF, CSV or billing paths costs a paying logo. Today only `infra/aws/scripts/smoke.sh` on `origin/dev` exists, and it checks health, CSP, `/v1` 401, unsigned webhook and demo init, not the product paths.

## Scope
In:
- One repeatable smoke script (extend `infra/aws/scripts/smoke.sh` or a pytest file under `tests/external/` marked `@pytest.mark.external`) covering PDF Q&A with citation, CSV query via E2B, SQL connection query, hybrid, autopilot, demo init, EDGAR search, and signup to verified to credit debit.
- A dated run log saved for Railway prod now and re-run on AWS prod at cutover (2519).
- Bugs found are filed as EPIC-23 tickets, not fixed here.

Out:
- Fixing the bugs found (EPIC-23).
- Load or performance testing.

## Acceptance criteria
- [ ] Given a base URL and a test account, when the smoke is run, then it exercises PDF, CSV, SQL, hybrid, autopilot, demo, EDGAR and signup-to-pay paths and prints pass/fail per path.
- [ ] Given a run on Railway prod, when it finishes, then a log with date, commit SHA, base URL and per-path result is saved and linked from the 1613 Plane issue.
- [ ] Given any failing path, when the run completes, then an EPIC-23 ticket exists for it with the failing step and response code.
- [ ] Given the same script, when it is pointed at `https://api-dev.docbot.space`, then it runs unchanged (only the base URL differs), so it can be reused at 2519.
- [ ] Given the log, when it is reviewed, then no secret value, token or connection string appears in it.

## Deliverable and approach
- Smoke script plus a short README section on how to run it with env var names only (test account credentials come from the founder's shell).
- Reuse the 2541 eval patterns for the PDF and hybrid checks; reuse `smoke.sh` for health and guard checks.
- Prod run is triggered by the founder; the agent writes and dry-runs the script against dev.

## Test / verification
- Script runs green on the AWS dev env (2505) and on Railway prod.
- Log file and Plane comment are the evidence; CI only runs the offline parts.

## Dependencies
Blocked by: none for the script; 2505 (dev env) to dry-run on AWS.
Blocks: G1, 1614 (uses the log as evidence), 2519 (cutover go/no-go).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Is a dedicated prod test org acceptable, and should its credit debits be reset after each run?
- Does the signup-to-pay path use Stripe test mode on prod, or is it stopped at checkout creation?
