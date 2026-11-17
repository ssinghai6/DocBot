# Backups and restore drill (DOCBOT-2510)

What Terraform sets up (`modules/app/data.tf`):

| Data | Backup | Retention |
|---|---|---|
| RDS Postgres | Automated backups + PITR | dev 3 days, prod 14 days (`db_backup_retention_days`) |
| RDS on delete | prod: final snapshot `docbot-prod-final`, automated backups kept (`delete_automated_backups = false`); dev: none | until deleted by hand |
| EFS (Chroma, `/data/chroma`) | `aws_efs_backup_policy` ENABLED -> AWS Backup default plan, vault `aws/efs/automatic-backup-vault` | daily, 35 days |

The drill below proves both restores work and records RTO/RPO. The founder runs it
(admin credentials); it creates billable resources that it deletes at the end. Run
it on dev first, then on prod before the cutover go/no-go.

Uses the ops task (`docbot-<env>-ops`, EFS root mounted at `/efs`) and the laptop
port forward from [db-access.md](db-access.md).

```bash
ENV=dev
CLUSTER=docbot-$ENV
NETCFG=$(aws ecs describe-services --cluster "$CLUSTER" --services "docbot-$ENV-api" \
  --query 'services[0].networkConfiguration' --output json)
```

## 1. RDS point-in-time restore

Restore into a **new** instance; the live one is untouched.

```bash
DB_SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=docbot-$ENV-db \
  --query 'SecurityGroups[0].GroupId' --output text)
aws rds describe-db-instances --db-instance-identifier docbot-$ENV \
  --query 'DBInstances[0].LatestRestorableTime'           # RPO reference point

date -u +%FT%TZ                                            # record: restore start
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier docbot-$ENV \
  --target-db-instance-identifier docbot-$ENV-drill \
  --use-latest-restorable-time \
  --db-subnet-group-name docbot-$ENV \
  --vpc-security-group-ids "$DB_SG" \
  --db-instance-class db.t4g.micro \
  --no-publicly-accessible \
  --manage-master-user-password \
  --no-deletion-protection
aws rds wait db-instance-available --db-instance-identifier docbot-$ENV-drill
date -u +%FT%TZ                                            # record: restore end (RTO)
```

`--manage-master-user-password` gives the drill instance its own RDS-managed secret
(Secrets Manager, name starts with `rds!db-`); read the password in the console.

### Row counts: live vs restored

Open two port forwards ([db-access.md](db-access.md) step 2), one with
`DB_HOST` = `docbot-$ENV` on `localPortNumber` 15432, one with `docbot-$ENV-drill`
on 15433. Then, for each port:

```bash
psql "postgresql://docbot@localhost:15432/docbot?sslmode=require" -At > live.txt <<'SQL'
SELECT format('SELECT %L, count(*) FROM %I.%I', table_name, table_schema, table_name)
FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name \gexec
SQL
# same with port 15433 > drill.txt
diff live.txt drill.txt
```

Differences are only expected in tables written after the restore point.

### Clean up

```bash
aws rds delete-db-instance --db-instance-identifier docbot-$ENV-drill \
  --skip-final-snapshot --delete-automated-backups
```

## 2. EFS restore (Chroma)

Restore a recovery point into a directory of the **same** file system; the live
`/docbot` tree is untouched. AWS Backup writes it to `/aws-backup-restore_<timestamp>/`
at the EFS root, which only the ops task sees (at `/efs`).

1. Console: AWS Backup -> Protected resources -> the `docbot-<env>-chroma` file system
   -> newest recovery point -> Restore -> **Restore to directory in source file system**
   (full restore). Use the default service role. Record start time.
2. Wait for the restore job to complete (AWS Backup -> Jobs -> Restore jobs). Record end time.
3. Compare file counts with an ops task:

```bash
aws ecs run-task --cluster "$CLUSTER" --task-definition docbot-$ENV-ops --launch-type FARGATE \
  --network-configuration "$NETCFG" \
  --overrides '{"containerOverrides":[{"name":"ops","command":["sh","-c","ls -d /efs/aws-backup-restore_*; for d in /efs/docbot/chroma /efs/aws-backup-restore_*/docbot/chroma; do echo \"$d $(find $d -type f | wc -l) files $(du -sh $d | cut -f1)\"; done"]}]}' \
  --query 'tasks[0].taskArn' --output text
aws logs tail /ecs/docbot-$ENV-api --since 10m | grep -A5 aws-backup-restore
```

The log stream is `api/ops/<task-id>` in `/ecs/docbot-<env>-api`.

### Real restore (not part of the drill)

Scale the API to 0 (`aws ecs update-service --cluster $CLUSTER --service docbot-$ENV-api --desired-count 0`),
then run an ops task that moves `/efs/docbot/chroma` aside and copies the restored
`chroma` directory into `/efs/docbot/chroma` (`cp -a`, keep uid/gid 1000), then scale
back to 1. Chroma is SQLite: never copy while the API is running.

### Clean up

Delete the `/efs/aws-backup-restore_*` directory with an ops task (`rm -rf`).

## 3. Application check

After a real restore (not needed for the drill): sign in, open 3 existing sessions,
ask one question each and confirm cited answers; open one saved DB connection
(proves `DB_ENCRYPTION_KEY` still decrypts).

## Record

| Date | Env | Restore | Restore point | Start | End | RTO | RPO | Row diff / file diff | By |
|---|---|---|---|---|---|---|---|---|---|
| | dev | RDS PITR | | | | | | | |
| | dev | EFS | | | | | | | |
| | prod | RDS PITR | | | | | | | |
| | prod | EFS | | | | | | | |

RPO: RDS PITR is ~5 minutes behind live; EFS is up to 24 h (daily backup).
