# DOCBOT-2510 Prod restore drill
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S2 | P1 | 3 | devops | - | Open (Wave B) |

## User story
As the founder about to put design partners' books on AWS, I want a timed restore of prod RDS and EFS from backup, so that I know the real recovery time and data-loss window before any customer data depends on them.

## Why now
Gate G4 needs a restore drill before cutover (2519). Backup config (RDS PITR 14 days, EFS daily via AWS Backup for 35 days) is merged to dev (PR #108), but nothing has been restored, and `infra/aws/docs/backup-restore.md` was lost to the `.gitignore` bug.

## Scope
In:
- Restore prod RDS to a new instance with point-in-time recovery; compare per-table row counts.
- Restore EFS from AWS Backup; mount on the `docbot-prod-ops` task; check a restored session answers.
- Record RTO and RPO; delete drill resources.
- Re-author `infra/aws/docs/backup-restore.md`.

Out:
- Restoring prod data into dev (never allowed).
- Multi-region backups.

## Acceptance criteria
- [ ] Given prod RDS has data (the 2515 rehearsal restore or seed rows), when it is restored by PITR to a new instance, then per-table row counts equal the source at the chosen restore time.
- [ ] Given an EFS recovery point, when it is restored and mounted on the ops task with `CHROMA_DIR` pointed at it, then a sampled session returns the same top-5 chunk IDs as before.
- [ ] Given the drill, when it ends, then RTO (start to verified) and RPO (restore point vs last write) are written in `backup-restore.md`, and the restored instance and EFS are deleted (verified with `describe-db-instances` and `describe-file-systems`).
- [ ] Given `backup-restore.md`, when `git check-ignore` is run, then nothing prints.

## Deliverable and approach
Devops runs the drill through the ops task and AWS CLI with founder approval (checklist item 24); the founder approves creating billable resources. No dump file or connection string is shared in chat.

## Test / verification
Doc evidence: commands, timestamps, row-count table, chunk-ID comparison for 3 sessions, deletion confirmation.

## Dependencies
Blocked by: 2576.
Blocks: 2519 (G4 part).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Run the drill after 2515 so it restores realistic data, or on seed data in S2 as scheduled? (S2 is before 2515 in S3; seed data is the default.)
