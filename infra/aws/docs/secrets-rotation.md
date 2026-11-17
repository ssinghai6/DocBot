# Secret rotation (DOCBOT-2509)

ECS injects secrets when a task **starts**. A running task keeps the old values
until it is replaced, so every secret change needs a new deployment.
`modules/app/rotation.tf` automates that:

```
EventBridge rule ──> Step Functions docbot-<env>-redeploy-on-secret-change
                     (ecs:UpdateService ForceNewDeployment, 3 retries)
```

| Rule | Fires on | Works today? |
|---|---|---|
| `docbot-<env>-rds-rotated` | `RotationSucceeded` for the RDS-managed master secret (AWS service event) | Expected yes: service events need no CloudTrail trail. **Unverified until the first forced test below.** |
| `docbot-<env>-app-secret-changed` | `PutSecretValue` / `UpdateSecret` on `docbot-<env>/*` (API-call event) | **No**, until a CloudTrail trail exists (DOCBOT-2524). Redeploy by hand (below). |

IAM is scoped: the state machine can only `ecs:UpdateService` the one API service;
EventBridge can only start the one state machine.

The founder runs every command here with admin credentials.

```bash
ENV=dev
CLUSTER=docbot-$ENV; SERVICE=docbot-$ENV-api
SFN=$(aws stepfunctions list-state-machines \
  --query "stateMachines[?name=='docbot-$ENV-redeploy-on-secret-change'].stateMachineArn" --output text)
```

## Manual redeploy (fallback for any secret)

```bash
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --force-new-deployment >/dev/null
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"
```

## Forced RDS rotation test (acceptance: health back in < 5 min)

RDS rotates the master password itself (`manage_master_user_password = true`).
Running tasks keep working on open connections, but new connections fail with the
old password until tasks are replaced, so health can dip briefly.

```bash
RDS_SECRET=$(cd infra/aws/envs/$ENV && terraform output -raw rds_master_secret_arn)
date -u +%T
aws secretsmanager rotate-secret --secret-id "$RDS_SECRET"

# Did EventBridge start the redeploy?
aws stepfunctions list-executions --state-machine-arn "$SFN" --max-results 3 \
  --query 'executions[].[status,startDate]' --output table
aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].deployments[].[status,rolloutState,createdAt]' --output table

until curl -fsS https://api-$ENV.docbot.space/api/health; do sleep 10; done; date -u +%T
```

For prod use `https://api.docbot.space` (only after the cutover).

If no execution appears within ~2 minutes, the RDS event does not reach EventBridge
in this account: run the manual redeploy, and record that RDS rotation needs the
manual step (or a trail, DOCBOT-2524) in the table below.

## Rotating an app secret (`docbot-<env>/<name>`)

1. Get the new value from the provider (Groq, Gemini, E2B, Stripe, ...).
2. `aws secretsmanager put-secret-value --secret-id docbot-$ENV/<name> --secret-string '<value>'`
   (leading space on the command line keeps it out of history).
3. Manual redeploy (above). Not automatic until DOCBOT-2524.
4. `curl https://api-$ENV.docbot.space/api/health`, then exercise the feature that uses the key.
5. Revoke the old key at the provider.

`OWNER_KEY`: generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Anything holding the old value must be updated at the same time.

## DB_ENCRYPTION_KEY cannot be rotated

`api/utils/encryption.py` decrypts with a single `Fernet(DB_ENCRYPTION_KEY)`; there is
no `MultiFernet` key list. Changing the value makes every saved DB connection and
CSV/credential blob undecryptable. **Do not rotate it.** Prod must keep the existing
Railway value unchanged. Rotation needs an app change first (MultiFernet with old +
new key, then a re-encrypt job): open a ticket for the senior-dev lane.

## Record

| Date | Env | Secret | Trigger (auto/manual) | Rotation start | Health OK at | Notes |
|---|---|---|---|---|---|---|
| | dev | RDS master | | | | |
