# RDS access from a laptop (DOCBOT-2515)

RDS is private (`publicly_accessible = false`, SG `docbot-<env>-db` allows 5432 only
from the task SG). To reach it from a laptop (for `scripts/pg_migrate.sh`, row counts
in the [restore drill](backup-restore.md), or a one-off `psql`), run a short-lived
**ops task** with ECS Exec and port-forward through it with SSM:

```
laptop localhost:15432 --SSM--> ops task (task SG) --5432--> RDS
```

What Terraform provides (`modules/app/ecs.tf`):

- task definition `docbot-<env>-ops`, container `ops`, same image/env/secrets as the API;
- task role `docbot-<env>-ops-task` with only the four `ssmmessages:*Channel` actions ECS Exec needs.
  The API task role has no permissions and no ECS Exec.

The founder runs every command below with their own admin (SSO) credentials. Agents
never run them and never see the DB password.

## Prerequisites

- AWS CLI v2 and the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`session-manager-plugin --version`).
- `psql` (for `pg_migrate.sh`: the same major as RDS, or set `PG_BIN`).
- Every app secret for the env has a value: the ops task injects the same secrets as the API and will not start otherwise.

## 1. Start an ops task with ECS Exec

```bash
ENV=dev                                   # or prod
CLUSTER=docbot-$ENV
# Same subnets/SG/public IP as the API service, so the DB SG lets it in.
NETCFG=$(aws ecs describe-services --cluster "$CLUSTER" --services "docbot-$ENV-api" \
  --query 'services[0].networkConfiguration' --output json)

TASK=$(aws ecs run-task --cluster "$CLUSTER" --task-definition "docbot-$ENV-ops" \
  --launch-type FARGATE --enable-execute-command \
  --network-configuration "$NETCFG" \
  --overrides '{"containerOverrides":[{"name":"ops","command":["sleep","3600"]}]}' \
  --query 'tasks[0].taskArn' --output text)
aws ecs wait tasks-running --cluster "$CLUSTER" --tasks "$TASK"

# ExecuteCommandAgent must be RUNNING before the port forward works (~30 s after start).
aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK" \
  --query 'tasks[0].containers[0].managedAgents[0].lastStatus' --output text
```

`sleep 3600` keeps the task alive for an hour; it stops itself after that.

## 2. Port-forward localhost:15432 to RDS

```bash
RUNTIME_ID=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK" \
  --query 'tasks[0].containers[0].runtimeId' --output text)
DB_HOST=$(aws rds describe-db-instances --db-instance-identifier "docbot-$ENV" \
  --query 'DBInstances[0].Endpoint.Address' --output text)

aws ssm start-session \
  --target "ecs:${CLUSTER}_${TASK##*/}_${RUNTIME_ID}" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters "{\"host\":[\"$DB_HOST\"],\"portNumber\":[\"5432\"],\"localPortNumber\":[\"15432\"]}"
```

Leave this running in its own terminal. For a restored drill instance, use that
instance's identifier for `DB_HOST` (it shares the DB SG).

## 3. Connect

User and database are both `docbot`. The password lives only in the RDS-managed
secret (`terraform output rds_master_secret_arn`); read it in the console
(Secrets Manager -> that secret -> Retrieve secret value), never in chat or a file.

```bash
psql "postgresql://docbot@localhost:15432/docbot?sslmode=require"     # prompts for the password

# pg_migrate.sh target (cutover runbook): URL-encode the password if it has
# reserved characters; leading-space or read -s keeps it out of shell history.
read -rs TARGET_DATABASE_URL && export TARGET_DATABASE_URL
#   postgresql://docbot:<PASSWORD>@localhost:15432/docbot?sslmode=require
```

`sslmode=require` encrypts without verifying the hostname, which is expected here
because the certificate names the RDS endpoint, not `localhost`. `pg_migrate.sh`
checks that the target has `rds_superuser`, which still works through the tunnel.

## 4. Clean up

```bash
aws ecs stop-task --cluster "$CLUSTER" --task "$TASK" --reason "db access done" >/dev/null
unset TARGET_DATABASE_URL
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `TargetNotConnected` | ECS Exec agent not running yet (step 1 check), or the task was started without `--enable-execute-command`. |
| Task stops right away with `ResourceInitializationError` | An app secret is still empty. |
| `psql` hangs | `DB_HOST` is wrong, or the task did not get the service's network config (needs the task SG). |
| Password rejected | RDS rotated it; re-read the secret. |
