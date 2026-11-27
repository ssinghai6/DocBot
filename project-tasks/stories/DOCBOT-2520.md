# DOCBOT-2520 Railway sunset
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S5 | P1 | 2 | founder | - | Open (Wave C) |

## User story
As the founder, I want Railway shut down and archived 14 days after a clean cutover, so that I stop paying two hosts while keeping a recoverable copy of the old data.

## Why now
Stopping the Railway bill is the reason for the migration. 2520 follows 2519 by 14 days (S5 gate: Railway cancelled).

## Scope
In:
- Railway warm and read-only for 14 days after 2519 with no rollback.
- Final `pg_dump` and `/data` tarball archived to the S3 migration bucket (CMK, 90-day retention).
- Railway services and volume deleted, plan cancelled.
- Rename `RAILWAY_BACKEND_URL` to a provider-neutral name: Vercel env (founder) and code in `next.config.ts` and any `src/` reference (senior-dev).

Out:
- Changing the 90-day retention.

## Acceptance criteria
- [ ] Given 14 days have passed since 2519 with no rollback, when the founder checks the S3 bucket, then the final dump and `/data` tarball exist with SSE-KMS and a 90-day expiry rule.
- [ ] Given the Railway dashboard, when the project is opened, then no services or volumes remain and the plan shows cancelled.
- [ ] Given the repo, when `git grep RAILWAY_BACKEND_URL` is run on `main` after the rename PR, then it returns nothing, and Vercel Production uses the new name with the same value.
- [ ] Given prod after the rename deploy, when `smoke.sh https://api.docbot.space` and a login plus chat click-through run, then both pass.

## Deliverable and approach
Founder-only for archive, deletion and Vercel (checklist item 28). Senior-dev ships a small PR renaming the variable with a fallback read of the old name for one release so the order of Vercel and code changes does not matter.

## Test / verification
S3 `head-object` output (names only); grep result; smoke pass.

## Dependencies
Blocked by: 2519 (+14 days).
Blocks: 2526 timing (SOC 2 evidence after hosting is single-provider).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- New variable name: `API_BACKEND_URL` or `BACKEND_URL`?
