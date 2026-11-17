# DOCBOT-2515 Postgres rehearsal Railway to prod RDS
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S3 | P1 | 5 | senior-dev | - | Open (Wave B) |

## User story
As the founder moving design partners' data, I want a rehearsed, verified copy of the Railway Postgres into prod RDS, so that on cutover day every row, sequence, trigger and encrypted credential arrives intact and I know how long it takes.

## Why now
The cutover (2519) is only safe if the dump and restore have been run end to end before. Scripts are merged to dev (PR #116: `infra/aws/scripts/pg_migrate.sh`, migration bucket in `infra/aws/account/migration_bucket.tf`), but `infra/aws/docs/db-access.md` (the laptop to RDS port forward through the ops task) was lost to the `.gitignore` bug and #118 is held until it is back.

## Scope
In:
- Re-author `infra/aws/docs/db-access.md` (ECS Exec port forward to `localhost:15432`).
- Rehearsal: `pg_dump -Fc` from Railway, `pg_restore --no-owner` into prod RDS (never dev) in one transaction, per-table counts and checksums diffed by `pg_migrate.sh`.
- Checks: audit-log immutability trigger present and blocks UPDATE; sequences above max(id); an encrypted `db_connections` row decrypts with the carried-over `DB_ENCRYPTION_KEY`.
- Duration recorded; rehearsal DB dropped before cutover.

Out:
- Chroma data (2516), the real cutover (2519).

## Acceptance criteria
- [ ] Given connection strings in the founder's shell only (`read -rs`), when `pg_migrate.sh` runs, then it exits 0 with per-table row counts and checksums equal between Railway and RDS.
- [ ] Given the restored DB, when an UPDATE is attempted on `audit_log`, then the immutability trigger rejects it.
- [ ] Given every table with a serial id, when `last_value` is compared with `max(id)`, then each sequence is greater.
- [ ] Given a saved connection restored from Railway, when the prod ops task calls the decrypt path with the prod key, then it decrypts and a test query runs.
- [ ] Given the rehearsal, when it finishes, then wall-clock duration of dump plus restore plus verify is written in `infra/aws/runbooks/cutover.md`, and the rehearsal database is dropped (confirmed by `\l`).

## Deliverable and approach
Senior-dev owns `infra/aws/scripts/**` fixes; devops supports the port forward. The founder runs anything needing Railway credentials and hands agents only the S3 location of the dump (checklist item 24). Agents never see connection strings.

## Test / verification
`pg_migrate.sh` output (counts and checksums, no values from rows); `bash -n` and shellcheck on the script in CI.

## Dependencies
Blocked by: 2514, 2576.
Blocks: 2518, 2519.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Railway Postgres major version vs RDS `db_engine_version` (default 16): confirm before the rehearsal, since `pg_migrate.sh` refuses a downgrade.
