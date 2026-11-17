# DOCBOT-2518 Rollback plan + drill
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S3 | P1 | 3 | devops | - | Open (Wave B) |

## User story
As the founder cutting over with paying design partners on the system, I want written rollback triggers and a drilled 15-minute flip back to Railway, so that a bad cutover costs minutes, not a lost customer.

## Why now
Gate G4 requires a rollback drill of 15 minutes or less before 2519. The plan is merged to dev (PR #117), but `infra/aws/docs/rollback.md` was lost to the `.gitignore` bug and no drill has run.

## Scope
In:
- Re-author `infra/aws/docs/rollback.md`.
- Triggers: 5xx over 2% for 10 minutes, failed health, p95 latency over 2x the Railway baseline, data mismatch in the post-restore verify.
- Drill: flip a test hostname (for example `api-rollback-test.docbot.space`) between the ALB and Railway and time it.
- Data-loss window for writes made on AWS before a rollback, stated and accepted by the founder.

Out:
- Reverse data sync from RDS to Railway (the accepted window replaces it).

## Acceptance criteria
- [ ] Given `rollback.md`, when read, then each trigger has a metric, a threshold, a duration and where to see it (2511 alarm or CloudWatch query).
- [ ] Given a test hostname with TTL 60 s pointing at the ALB, when the founder flips it to Railway following the doc, then 3 public resolvers return Railway and `/api/health` passes within 15 minutes of the decision, timed.
- [ ] Given the doc's data-loss section, when the founder reads it, then it names which tables can receive writes on AWS (sessions, messages, credits, audit_log) and the founder's acceptance is recorded with a date.
- [ ] Given `rollback.md`, when `git check-ignore` is run, then nothing prints.

## Deliverable and approach
Devops writes the doc and drill steps; the founder makes the DNS changes (agents never touch DNS). Railway stays warm and read-only for 14 days after cutover (2520), which keeps the rollback target alive.

## Test / verification
Drill log with timestamps; resolver outputs; founder acceptance line.

## Dependencies
Blocked by: 2515, 2516, 2517.
Blocks: 2519 (G4 part).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Credit debits made on AWS before a rollback would be lost; refund manually from `credit_ledger` export, or keep a short write freeze after the flip until the 1-hour watch passes?
