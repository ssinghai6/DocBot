# Founder checklist: wave A (DOCBOT-2502 / 2503 / 2504)

Goal: AWS dev live at `https://api-dev.docbot.space`. **The founder runs every
command and console step here**; agents prepared the code and do not hold AWS
credentials. Total ~4 h, most of it waiting on RDS and DNS. Background for each
step is in [../README.md](../README.md) ("Founder steps, in order").

Placeholders: `<ACCOUNT_ID>` (12-digit AWS account), `<ALB_DNS_NAME>`, `<ARN>`.
Never paste real values into tickets, PRs or chat.

| # | Step | Est. | Done |
|---|---|---|---|
| 0 | Tools + credentials | 20 min | [ ] |
| 1 | `dev` synced with `main` (2502): re-check before every deploy | 2 min | [ ] |
| 2 | Bootstrap: state bucket + lock table | 15 min | [ ] |
| 3 | Account: OIDC, ECR, budgets, anomaly alerts, migration bucket (2504 budgets) | 30 min | [ ] |
| 4 | Dev: ACM cert + validation CNAMEs | 40 min | [ ] |
| 5 | Dev: full apply | 30 min | [ ] |
| 6 | Dev secrets | 20 min | [ ] |
| 7 | `api-dev` CNAME | 15 min | [ ] |
| 8 | GitHub variables + first deploy | 30 min | [ ] |
| 9 | Alarm email, smoke, manual checks | 40 min | [ ] |
| | **Total** | **~4 h** | |

## 0. Tools and credentials (20 min)

Terraform >= 1.9, AWS CLI v2, Session Manager plugin, `gh`, `vercel` CLI (or the
Vercel dashboard). Admin SSO for the target account, region `us-east-1`.

```bash
aws sso login --profile <PROFILE> && export AWS_PROFILE=<PROFILE>
```

**Verify:** `aws sts get-caller-identity --query Account --output text` prints the
intended account; `terraform version` >= 1.9.

## 1. dev synced with main (DOCBOT-2502): re-check before every deploy

**Verify:** `git fetch origin && git diff --stat origin/dev...origin/main -- api src infra Dockerfile requirements.txt next.config.ts`
prints nothing (no app or infra change on `main` missing from `dev`). Docs/CI-only
commits on `main` (e.g. Plane sync) do not matter for the deploy; if this prints
anything, repeat README step 0 (merge `origin/main` into `dev`). `dev` drifted 115
commits behind once (synced 2026-11-27); merge `main` into `dev` weekly, never rebase.

## 2. Bootstrap (15 min)

```bash
cd infra/aws/bootstrap && terraform init
terraform apply -var aws_region=us-east-1 -var state_bucket_name=docbot-tfstate-<ACCOUNT_ID>
terraform output
```

Move `terraform.tfstate` out of the repo afterwards (local state, no secrets, but
needed to change the bucket later).

**Verify:** `aws s3api get-bucket-versioning --bucket docbot-tfstate-<ACCOUNT_ID>`
shows `Enabled`; `aws dynamodb describe-table --table-name docbot-tfstate-lock --query Table.TableStatus` is `ACTIVE`.

## 3. Account stack (30 min)

Fill `bucket` and `region` in `infra/aws/account/backend.hcl` from step 2 outputs.
`cp terraform.tfvars.example terraform.tfvars`; set `budget_email`. If
`aws ce get-anomaly-monitors` already lists a services monitor, set
`anomaly_monitor_arn`. If `aws iam list-open-id-connect-providers` already shows
`token.actions.githubusercontent.com`, set `create_github_oidc_provider = false`.
If PR #86's flat `infra/aws/dev` was ever applied, run the imports in the README
first ("Already applied PR #86's flat infra/aws/dev?").

```bash
cd infra/aws/account && terraform init -backend-config=backend.hcl
terraform plan -out=tfplan && terraform apply tfplan
```

Then: Billing -> Cost allocation tags -> activate `Environment` (per-env budgets
read $0 without it). Click the cost-anomaly confirmation email.

**Verify:** `aws ecr describe-repositories --repository-names docbot-api --query 'repositories[0].imageTagMutability'`
prints `IMMUTABLE_WITH_EXCLUSION`; `aws budgets describe-budgets --account-id <ACCOUNT_ID> --query 'Budgets[].BudgetName'`
lists `docbot-account-monthly-net`, `-gross`, `docbot-dev-monthly-gross`, `docbot-prod-monthly-gross`.

## 4. Dev: ACM certificate (40 min, mostly DNS wait)

Fill `infra/aws/envs/dev/backend.hcl`; `cp terraform.tfvars.example terraform.tfvars`;
set `alarm_email`, and `frontend_url` / `allowed_origins` to the dev frontend
(the Vercel Preview URL for branch `dev`, unless `dev.docbot.space` exists).

```bash
cd infra/aws/envs/dev && terraform init -backend-config=backend.hcl
terraform apply -target=module.app.aws_acm_certificate.api
terraform output acm_validation_records
```

Add each record in Vercel DNS for `docbot.space` (name = record name without the
trailing `.docbot.space.`):

```bash
vercel dns add docbot.space <_TOKEN>.api-dev CNAME <_VALUE>.acm-validations.aws.
```

**Verify:** `dig +short CNAME <_TOKEN>.api-dev.docbot.space @1.1.1.1` returns the value, then
`aws acm list-certificates --query "CertificateSummaryList[?DomainName=='api-dev.docbot.space'].Status"` shows `ISSUED`
(usually 5-30 min).

## 5. Dev: full apply (30 min; RDS takes ~10-15)

```bash
terraform plan -out=tfplan    # review: VPC, ALB, ECS, RDS, EFS, secrets, alarms, rotation, deploy role
terraform apply tfplan
terraform output
```

The ECS service starts at `desired_count = 1` but its tasks cannot start until step 6
(empty secrets fail closed). That is expected.

**Verify:** `aws rds describe-db-instances --db-instance-identifier docbot-dev --query 'DBInstances[0].DBInstanceStatus'`
is `available`; `terraform output -raw alb_dns_name` prints a `*.elb.amazonaws.com` name.

## 6. Dev secrets (20 min)

`terraform output app_secret_arns` lists `groq_api_key`, `GEMINI_API_KEY`,
`E2B_API_KEY`, `DB_ENCRYPTION_KEY`, `OWNER_KEY`. Dev gets a **new** Fernet key, never
prod's. Start each command with a space to keep values out of shell history.

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # DB_ENCRYPTION_KEY (dev only)
python -c "import secrets; print(secrets.token_urlsafe(32))"                               # OWNER_KEY
 aws secretsmanager put-secret-value --secret-id <ARN> --secret-string '<VALUE>'
```

**Verify** (no values printed): for each ARN,
`aws secretsmanager list-secret-version-ids --secret-id <ARN> --query 'length(Versions)'` is `1`.
Then `aws ecs describe-services --cluster docbot-dev --services docbot-dev-api --query 'services[0].runningCount'` reaches `1`
(or wait for the first deploy in step 8).

## 7. api-dev CNAME (15 min)

```bash
vercel dns add docbot.space api-dev CNAME <ALB_DNS_NAME>
```

**Verify:** `dig +short api-dev.docbot.space @8.8.8.8` returns the ALB name, and
`curl https://api-dev.docbot.space/api/health` returns `{"status":"ok",...,"db":"ok"}`
once a task is running (a 503 before then is the ALB with no healthy target).

## 8. GitHub variables and first deploy (30 min)

```bash
gh variable set AWS_DEV_DEPLOY_ROLE_ARN --body "$(cd infra/aws/envs/dev && terraform output -raw deploy_role_arn)"
gh variable set AWS_DEV_REGION --body us-east-1
gh variable set AWS_DEV_API_URL --body https://api-dev.docbot.space
gh workflow run "Deploy AWS dev" --ref dev
```

Details: [first-deploy.md](first-deploy.md).

**Verify:** the run is green and ends with the notice `Promotable to prod: <sha> -> <digest>`;
`aws ecr describe-images --repository-name docbot-api --image-ids imageTag=tested-<sha>` succeeds.

## 9. Alarms, smoke, manual checks (40 min)

Click the SNS confirmation email (alarms are silent until then).

```bash
curl https://api-dev.docbot.space/api/health
infra/aws/smoke-test.sh https://api-dev.docbot.space
infra/aws/scripts/smoke.sh https://api-dev.docbot.space
aws cloudwatch set-alarm-state --alarm-name docbot-dev-rds-cpu-high --state-value ALARM --state-reason drill
```

**Verify:** both smoke scripts print PASS / exit 0; the drill alarm email arrives
(the alarm returns to OK by itself at the next evaluation). Then the manual list in
[first-deploy.md](first-deploy.md) section 4 (PDF chat, CSV, 120 s SSE, EFS
persistence, `xff-shape` log line).

## After wave A

Not in this checklist: prod apply, `aws-prod` GitHub environment, restore drill
([backup-restore.md](backup-restore.md)), rotation test
([secrets-rotation.md](secrets-rotation.md)), cutover ([dns-cutover.md](dns-cutover.md)).
To save money while dev is idle: [cost.md](cost.md) (scale to zero).
