# DOCBOT-2509 Secrets rotation
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S2 | P1 | 5 | devops | - | Open (Wave B) |

## User story
As the founder running DocBot on AWS prod, I want every secret (the RDS master password and the app keys) to rotate without an outage or a lost key, so that a leaked or expiring credential can be replaced before and after cutover without breaking design partners' saved connections.

## Why now
Gate G4 (cutover readiness) needs rotation proven on the prod env before 2519 in mid-January. RDS rotates its master password weekly by default; without the forced redeploy a running task keeps the old password and the API returns 503 until ECS replaces it. Code is merged to dev (PR #110), but nothing is applied and `infra/aws/docs/secrets-rotation.md` was lost to the root `docs/` rule in `.gitignore`, so the runbook must be re-authored.

## Scope
In:
- Re-author `infra/aws/docs/secrets-rotation.md` (lost doc) covering RDS-managed rotation, app keys (Groq, Gemini, E2B, `OWNER_KEY`, Stripe, OAuth) and `DB_ENCRYPTION_KEY` via the 2578 dual-key path.
- Verify `infra/aws/modules/app/rotation.tf` (EventBridge to Step Functions `ecs:UpdateService`) on the prod env after 2576.
- Manual force-new-deployment fallback for app secrets until CloudTrail exists (2524), because `PutSecretValue` events are only delivered with a trail.
- Forced-rotation test on the prod env (no customer traffic yet) with timings recorded.

Out:
- Rotating any secret on Railway prod (founder does that only at cutover).
- Building MultiFernet itself (2578).
- CloudTrail (2524).

## Acceptance criteria
- [ ] Given the prod env is applied (2576), when the founder runs "Rotate secret immediately" on the RDS-managed master secret, then an EventBridge rule starts the redeploy state machine and `/api/health` returns `ok` with db ok within 5 minutes, with the timestamps recorded in the doc.
- [ ] Given an app secret such as `groq_api_key` is updated in the console, when the documented manual step (`aws ecs update-service --force-new-deployment`) is run, then the new task uses the new value and the old task drains with no 5xx above the 2511 alarm threshold.
- [ ] Given 2578 is merged, when `DB_ENCRYPTION_KEY` is rotated by prepending a new key and running the re-encrypt script, then a saved connection created before the rotation still decrypts and connects, and after the old key is removed it still decrypts.
- [ ] Given the runbook, when a reader follows it for each of Groq, Gemini, E2B, `OWNER_KEY`, Stripe and OAuth secrets, then each has an owner (founder), the console path, the redeploy step and a verification step, and no secret value appears anywhere in the doc.
- [ ] Given the change is complete, when `git check-ignore infra/aws/docs/secrets-rotation.md` is run, then it prints nothing (the doc is tracked).

## Deliverable and approach
- `infra/aws/docs/secrets-rotation.md` re-authored from `rotation.tf` comments and the README "Secrets" section on `origin/dev`.
- Evidence section in the doc: date, env, rotation trigger time, deploy start, health-ok time. Command output pasted without values.
- Devops writes the doc and any Terraform fix; the founder clicks rotate and fills values. Agents never apply Terraform and never see secret values.
- If the EventBridge rule does not fire in prod, fix in `modules/app/rotation.tf` and re-plan; dev must also show 0 unexpected changes.

## Test / verification
- `terraform fmt -check` and `terraform validate` in `infra/aws/envs/prod` and `envs/dev` (CI).
- Founder-run forced rotation on prod env; `aws stepfunctions list-executions` shows a SUCCEEDED run; health check timings in the doc.
- 2578 unit tests (old-only, both, new-only keys) green in CI.

## Dependencies
Blocked by: 2508 (shared module + envs/prod), 2578 (MultiFernet dual key), 2576 (prod apply) for the live test.
Blocks: 2519 (cutover, G4 part).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Keep RDS rotation weekly in prod, or lengthen the interval (founder checklist item 9b recommends keeping it with the forced redeploy)?
- Should `OWNER_KEY` rotation also invalidate existing owner-unlock cookies, or is waiting for TTL acceptable?
