# Rollback plan (DOCBOT-2518)

Two different rollbacks:

| | What | How | Data loss |
|---|---|---|---|
| **A. Bad release on AWS** | a promoted image misbehaves | `deploy-aws-prod.yml` rolls back automatically when the smoke test fails; otherwise re-run "Deploy AWS prod" with the previous good SHA | none |
| **B. Cutover rollback** | AWS prod as a whole is not fit to serve | DNS back to Railway (below), <= 15 min | writes made on AWS after the flip (see "Data-loss window") |

Cutover order and the flip itself: [../runbooks/cutover.md](../runbooks/cutover.md),
[dns-cutover.md](dns-cutover.md). The founder decides and does every DNS, Railway and
console step; the devops agent watches alarms and logs and calls a trigger.

## A. Release rollback

```bash
gh workflow run "Deploy AWS prod" --ref main -f git_sha=<PREVIOUS_GOOD_SHA>   # approve in Actions
```

The previous task definition is printed in each promote run's summary. Images tagged
`prod-<sha>` are kept (last 30) by the ECR lifecycle rule, so earlier prod digests stay
available. Tasks that never get healthy are already rolled back by the ECS deployment
circuit breaker.

## B. Triggers (during the 1-hour watch and the 14 days before Railway sunset)

| Trigger | Source | Threshold |
|---|---|---|
| 5xx rate | alarm `docbot-prod-alb-5xx-rate` | > 2% of requests for 2 x 5 min (prod sets `alarm_5xx_percent = 2`; ignored under 20 req / 5 min) |
| No healthy task | alarms `docbot-prod-running-below-desired`, `docbot-prod-unhealthy-targets` | not cleared within 10 min by ECS replacing the task |
| Database | alarms `docbot-prod-rds-cpu-high`, `-rds-free-storage-low`, `-rds-connections-high` | sustained and user-visible |
| Smoke / spot checks | runbook T+40m checks | any failure that a fix-forward cannot resolve within 15 min |
| Data | wrong or missing user data on AWS | any confirmed case: roll back at once |

Before the flip (go/no-go #2 fails) there is nothing to roll back: unset
`WRITE_FREEZE` on Railway and stop.

## B. Flip back to Railway (target <= 15 min)

| Min | Step | Who |
|---|---|---|
| 0 | Decide: trigger confirmed, rollback called | F |
| 0-3 | Vercel DNS: edit `api` CNAME from `<ALB_DNS_NAME>` back to `<RAILWAY_CNAME_TARGET>` | F |
| 0-5 | Railway: unset `WRITE_FREEZE` (service redeploys; writes resume) | F |
| 3-8 | `dig +short api.docbot.space @1.1.1.1` / `@8.8.8.8` / `@9.9.9.9` all show Railway | D |
| 5-10 | `infra/aws/smoke-test.sh https://api.docbot.space https://www.docbot.space`, then `infra/aws/scripts/smoke.sh https://www.docbot.space` | D |
| 10-15 | Send template C (runbook). Record times below. | F |
| after | Scale AWS prod to 0 once resolvers show Railway and AWS traffic stops: `aws ecs update-service --cluster docbot-prod --service docbot-prod-api --desired-count 0` | F |

Vercel prod's `BACKEND_URL` stays `https://api.docbot.space` throughout, so no
Vercel redeploy is needed. Railway keeps the `api.docbot.space` custom domain until
sunset, which is what makes this a single record edit.

## Data-loss window

Between the flip and a rollback, writes go to RDS only (sign-ups, chats, uploads,
credit ledger, audit log, Chroma on EFS). After a rollback Railway has none of
them. There is **no reverse migration script**; recovering them would be a manual,
per-table copy from RDS. Practical exposure: the window from flip to rollback,
<= the 1-hour watch for most triggers. Stripe: payments made on AWS in that window
must be checked in the Stripe dashboard and credited by hand on Railway.

The founder accepts this window at go/no-go #1 (sign-off table in the runbook).
After the 1-hour watch, the bar for rollback is higher: prefer fix-forward.

## Drill (dev, before go/no-go)

Proves the alert path and the <= 15 min DNS flip without touching prod:

1. Alarm path: `aws cloudwatch set-alarm-state --alarm-name docbot-dev-alb-5xx-rate --state-value ALARM --state-reason drill`;
   time until the SNS email arrives.
2. DNS flip timing: create a throwaway record `rollback-drill` CNAME -> the dev ALB DNS
   name, wait > 60 s, then edit it to `<RAILWAY_CNAME_TARGET>` and time until all three
   resolvers show the new value. Delete the record after.
3. Release rollback: re-run an older green "Deploy AWS dev" run (`gh run rerun <run-id>`);
   it redeploys that commit's image, already in ECR, without rebuilding. Time until
   `/api/health` is 200 on the old image.

## Record

| Date | Drill or real | Trigger | Called at | DNS edited | Resolvers OK | Smoke OK | Total min | Writes lost |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
