# DOCBOT-2579 NEW: Merge dev into main before prod apply (needed for 2576)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S1 | P1 | 1 | founder | - | New (2026-11-17) |

## User story
As the founder about to apply AWS prod, I want `dev` (with `infra/aws/**` and `.github/workflows/deploy-aws-prod.yml`) merged into `main`, so that the `aws-prod` GitHub environment, which only accepts branch `main`, has a prod deploy workflow to run.

## Why now
The founder checklist (item 22, corrected 2026-11-17) sets the `aws-prod` deployment branch to `main`, and `deploy-aws-prod.yml` runs from `refs/heads/main`. Today the AWS code (PRs #86, #106 to #117) lives only on `dev`. Without this merge 2576 cannot finish and the soak (2513) cannot start in S1.

## Scope
In:
- Re-author the 8 lost AWS docs first, or confirm the PRs that restore them are merged to `dev` (2505, 2507, 2509, 2510, 2515, 2517, 2518), so `main` does not ship dangling links.
- Merge `dev` into `main` (PR, founder approves and merges).
- Quiet-window timing and a golden-path smoke on Railway prod afterwards, because a push to `main` also deploys Railway prod.

Out:
- Any `terraform apply` (2576), any change to Railway config.

## Acceptance criteria
- [ ] Given the merge PR, when its diff is reviewed, then it contains only commits already on `dev` (`git log origin/main..origin/dev`), and CI is green.
- [ ] Given the merge to `main` triggers a Railway prod deploy, when the merge is done, then it happens in a quiet window (no design-partner session in the last 30 minutes, not during a scheduled demo) chosen by the founder.
- [ ] Given the Railway deploy finishes, when the golden-path smoke runs (`infra/aws/scripts/smoke.sh <railway prod url>` plus PDF chat with a citation, CSV query, login and a billing debit), then every check passes.
- [ ] Given `main` after the merge, when `git ls-tree origin/main infra/aws .github/workflows/deploy-aws-prod.yml` is run, then both exist.
- [ ] Given the merge, when `git rev-list --count origin/main..origin/dev` is run, then it prints 0.

## Deliverable and approach
Founder-only merge (it ships to prod). Devops prepares the PR description listing the merged dev PRs and confirms no app code differs in ways not already smoke-tested on api-dev (2505). If the smoke fails, the founder reverts the merge commit on `main` and Railway redeploys the previous build.

## Test / verification
CI on the merge PR; Railway deploy log status; smoke output pasted (no secrets).

## Dependencies
Blocked by: none (dev PRs #106 to #117 merged; lost AWS docs restored is preferred).
Blocks: 2576.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should this become a standing rule (merge `dev` to `main` once per sprint) so `main` and `dev` do not drift again?
- Adding 2579 puts S1 at 29 pts (cap 28) and the founder lane at 5; accept, or move 3513 (2 pts) to S2?
