# DOCBOT-2514 Cutover runbook + go/no-go
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S2 | P1 | 3 | devops | - | Write-freeze code merged (PR #111); runbook open |

## User story
As the founder executing the cutover alone, I want one timed runbook with an explicit go/no-go list, so that I can move prod from Railway to AWS in a single window without forgetting a callback URL or a data step.

## Why now
2515, 2516 and 2518 rehearse against this runbook, and the founder must dry-read and sign it before mid-January. A draft exists at `infra/aws/runbooks/cutover.md` on `origin/dev` and the write-freeze code is merged (PR #111), but it references `infra/aws/docs/dns-cutover.md` and the rollback doc, both lost to the `.gitignore` bug.

## Scope
In:
- Timed steps: T-7d stable hostname (2517), T-48h TTL 60 s, T-0 write freeze, final dump and restore, verify, DNS flip, 1-hour watch, T+14d sunset (2520).
- Owner per step (F founder, D devops, S senior-dev), comms template for design partners.
- Go/no-go checklist including external callbacks (Stripe webhook, GitHub and Google OAuth, SAML ACS) and G1/G4 status.
- Founder dry-read and sign-off.

Out:
- Executing the cutover (2519), the rollback mechanics (2518).

## Acceptance criteria
- [ ] Given `infra/aws/runbooks/cutover.md`, when read top to bottom, then every step has a time offset, an owner letter and a verification command or check.
- [ ] Given the go/no-go list, when compared with founder checklist item 8, then every external callback URL host is listed with its new value as `api.docbot.space`.
- [ ] Given the runbook links, when each linked doc path is checked with `git ls-files`, then every link resolves (dns-cutover, rollback, db-access, backup-restore).
- [ ] Given a design-partner comms template, when read, then it states the freeze window, expected duration and a support contact, with no SOC 2 or "enterprise-grade" claim.
- [ ] Given the dry-read, when the founder signs, then the sign-off date is in the runbook header.

## Deliverable and approach
Devops edits the runbook on a branch from `dev`; links must resolve once the lost docs are re-authored in 2509, 2510, 2515, 2517, 2518. PR #118 (runbook link to `docs/db-access.md`) lands after `db-access.md` is restored.

## Test / verification
Markdown link check on `infra/aws/**`; founder sign-off recorded.

## Dependencies
Blocked by: 2508.
Blocks: 2515, 2516, 2518, 2519.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Freeze window length: plan for 60 minutes based on the 2515 rehearsal duration, or measure first?
