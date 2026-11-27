# DOCBOT-2502 Fast-forward dev to main
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 1 | founder | 11-20 | Open (Phase 0, Wave A) |

## User story
As the founder deploying the first AWS image from `dev`, I want `dev` to contain everything on `main`, so that the AWS dev image runs the same app code as Railway prod and the smoke test (2505) means something.

## Why now
Shipped record. After #86 merged, `dev` was behind `main` (it lacked `docker-entrypoint.sh` and recent security fixes). A fast-forward was no longer possible, so `main` was merged into `dev` on 2026-11-17 (commit 724ac6c, "Merge branch 'main' into dev (DOCBOT-2502)").

## Scope
In:
- Merge `main` into `dev` with no conflicts left unresolved.

Out:
- Any later drift between `main` and `dev` (each Wave A PR keeps `dev` current).

## Acceptance criteria
- [ ] Given merge commit 724ac6c, when `git log -1 --format=%P 724ac6c` is read, then its second parent was the tip of `origin/main` at merge time on 2026-11-17 (dev contained all of main).
- [ ] Given `origin/dev`, when `git show origin/dev:docker-entrypoint.sh` is run, then the file exists, so the image built from `dev` has the entrypoint.
- [ ] Given CI on `dev` after the merge, when the run is checked, then unit and integration jobs are green.

## Deliverable and approach
Shipped: merge commit 724ac6c on `origin/dev`, done by the devops agent because a fast-forward was impossible after #86 (founder checklist item 12 records this).

## Test / verification
`git log origin/dev --oneline | grep 724ac6c`; CI run on that commit.

## Dependencies
Blocked by: 2501.
Blocks: 2503, 2505.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- `main` has moved since (for example PR #99, #121). Before 2505's first deploy, re-merge `main` into `dev` so the smoke runs current code; is that a standing step in each Wave A PR or a one-off?
