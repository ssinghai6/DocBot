# Prod cutover: Railway to AWS (DOCBOT-2514)

Moves the DocBot API from Railway to the AWS prod stack (`infra/aws/envs/prod`).
The frontend stays on Vercel; it reaches the API through `api.docbot.space`.

This runbook owns the **data** steps and the order of everything. Two sections
live with DevOps and are referenced, not repeated:

- DNS: `infra/aws/docs/dns-cutover.md` (DOCBOT-2517): stable hostname, TTL, the flip, resolver checks.
- Rollback: the DOCBOT-2518 rollback plan: triggers, the flip back, the data-loss window.

Owners: **F** = founder (Railway, Vercel, DNS, AWS console access), **D** = devops
agent, **S** = senior-dev agent. Agents never see connection strings.

## Scripts

| Script | What it does |
|---|---|
| `infra/aws/scripts/pg_migrate.sh` | pg_dump Railway, restore into RDS in one transaction, diff per-table counts and checksums, check triggers and sequences. Exit 0 = verified. |
| `infra/aws/scripts/data_copy.sh` | Copy `/data/chroma` Railway to EFS with sha256 checks at both ends (ADR 0001). |
| `infra/aws/scripts/smoke.sh <url>` | Health + db, CSP, `/v1` 401, unsigned webhook rejected, demo init. |

Connection strings go in env vars in the founder's own shell, never in chat,
tickets or files in the repo. Prefer a leading space (`HISTCONTROL=ignorespace`)
or `read -s` so they stay out of shell history:

```bash
read -rs SOURCE_DATABASE_URL && export SOURCE_DATABASE_URL   # Railway: Postgres service > Connect > public URL
read -rs TARGET_DATABASE_URL && export TARGET_DATABASE_URL   # RDS via the port forward: postgresql://USER:PASS@localhost:15432/docbot?sslmode=require
```

RDS is not publicly reachable. The founder's laptop reaches it through the
access path DevOps provides for DOCBOT-2515 (an SSM port forward on
`localhost:15432`). The RDS user and password come from its Secrets Manager
secret in the console.

## Go / no-go (all must be true at T-1h; F signs)

- [ ] G1 security gate closed; G3 cost gate open; G4: 14-day staging soak ≥ 99.5% (2513), restore drill passed (2510), rollback drill ≤ 15 min (2518).
- [ ] Rehearsal within the last 7 days: `pg_migrate.sh` printed `RESULT: VERIFIED` against prod RDS, and `data_copy.sh` import + 5-session spot check passed (2515, 2516). Durations recorded below.
- [ ] Railway Postgres major ≤ RDS major (`pg_migrate.sh` prints both; it refuses otherwise).
- [ ] AWS prod secrets filled; `DB_ENCRYPTION_KEY` is the **existing** Railway prod key, unchanged (a saved DB connection decrypts in the rehearsal).
- [ ] AWS prod task env matches Railway prod for everything that is not a host detail (auth mode, billing, demo, rate limits, `ALLOWED_ORIGINS`, OAuth/SAML/Stripe settings). `TRUSTED_PROXY_HOPS` set from the `xff-shape:` log line seen on staging.
- [ ] `api.docbot.space` has pointed at Railway for ≥ 7 days and Vercel prod uses it (2517). TTL is 60s (set ≥ 48h ago).
- [ ] AWS prod service at **0 tasks**, deploy pinned to the image digest staging soaked.
- [ ] Comms drafted (below); a 60-minute window chosen at the lowest-traffic hour.
- [ ] Rollback owner on hand for the whole window plus 1 hour.

## Timeline

| When | Step | Owner | Done when | Rehearsal time |
|---|---|---|---|---|
| T-7d | Stable hostname on Railway, Vercel prod switched to it (see dns-cutover.md) | F | site works via `api.docbot.space` | |
| T-48h | Lower `api.docbot.space` TTL to 60s (dns-cutover.md) | F | `dig` shows TTL 60 | |
| T-24h | Send the maintenance notice (template A) | F | sent | |
| T-1h | Go/no-go list above signed | F | signed | |
| T-0 | **Write freeze:** set `WRITE_FREEZE=true` on the Railway service (redeploys; reads keep working, every write gets 503) | F | `curl -X POST https://api.docbot.space/api/demo/init` returns 503 | |
| T+5m | Final Postgres copy: `infra/aws/scripts/pg_migrate.sh` with `DUMP_FILE=./final.dump` | F | `RESULT: VERIFIED`, exit 0 | |
| T+5m (parallel) | Final Chroma copy: `data_copy.sh export chroma.tgz`, then `AWS_CLUSTER=… AWS_SERVICE=… S3_URI=s3://…/chroma-final.tgz data_copy.sh import chroma.tgz` | F (D watches) | both print OK | |
| T+20m | Scale AWS prod service to 1 (never above 1, ADR 0001); wait for healthy target | D/F | ECS steady state | |
| T+25m | `CONNECT_TO=<alb-dns-name> infra/aws/scripts/smoke.sh https://api.docbot.space` (hits the ALB before DNS moves) | D | `SMOKE OK` | |
| T+25m | Data spot checks on AWS: sign in as the founder account, open 3 existing sessions, ask one question each, open one saved DB connection | F | cited answers; connection decrypts | |
| T+30m | **Go/no-go #2.** Any failure above: stop here, unset `WRITE_FREEZE` on Railway, nothing else changed (rollback is free before the flip) | F | decision | |
| T+30m | Flip `api.docbot.space` to the ALB (dns-cutover.md) | F | 3 resolvers show the ALB | |
| T+40m | `smoke.sh https://<vercel prod url>` through Vercel; the prod checks from 2519: PDF chat, CSV, hybrid, login, billing debit | F/D | all pass | |
| T+40m → T+100m | Watch for 1 hour: ALB 5xx, target health, p95, logs. Any 2518 trigger → rollback plan | D | no trigger fired | |
| T+100m | Send the all-clear (template B). Leave Railway running with `WRITE_FREEZE=true` (read-only, warm for rollback) | F | sent | |
| T+14d | Railway sunset (2520): archive `final.dump` and `chroma.tgz` to S3 under KMS (90-day retention), then delete Railway services | F | archived, Railway deleted | |

Writes made on AWS after the flip are **not** on Railway. Rolling back after
the flip therefore loses them unless they are copied back; the 2518 plan states
the window and the founder accepts it at go/no-go.

## Notes

- `pg_migrate.sh` restores in one transaction: a failure rolls back and the target is unchanged, so it is safe to re-run. It refuses a target that is not RDS (so swapped env vars cannot wipe Railway).
- The dump and the tarball hold customer data (credentials stay Fernet-encrypted). Keep them on an encrypted disk, never in the repo, and delete local copies once archived.
- Stripe webhooks rejected during the freeze are retried by Stripe and arrive on AWS after the flip. Check the Stripe dashboard for failed deliveries at T+100m.
- If `WRITE_FREEZE` is not available on the deployed Railway build (PR #111 not merged), the fallback is to block writes at the edge for the window; do not cut over without a freeze, the row-count diff will fail.

## Comms templates

**A. Notice (T-24h)**
> DocBot maintenance: <date>, <start>–<end> <tz>. During this window you can read existing chats and documents, but uploads, new questions and sign-ups are paused. Nothing you have saved will be lost. We will post here when it is done.

**B. All clear**
> Maintenance is complete and DocBot is fully back. If anything looks wrong, reply to this message.

**C. Rolled back**
> We paused the maintenance and DocBot is running on the previous setup. Everything is available again. Changes made between <time> and <time> may need to be redone; we will contact affected users directly.

## Sign-off

| Item | Name | Date |
|---|---|---|
| Runbook dry-read | | |
| Go/no-go #1 (T-1h) | | |
| Go/no-go #2 (before flip) | | |
| Rollback data-loss window accepted (2518) | | |
