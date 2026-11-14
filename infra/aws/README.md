# DocBot on AWS

Terraform for the DocBot API on AWS (ECS Fargate + RDS Postgres + EFS for Chroma), one module used by two environments.

```
infra/aws/
  bootstrap/      S3 state bucket + DynamoDB lock table (local state, apply once)
  account/        account-wide: GitHub OIDC provider, ECR repo docbot-api, budgets, cost anomaly alerts
  modules/app/    network, ALB + ACM, ECS service, RDS, EFS, secrets, deploy role
  envs/dev/       api-dev.docbot.space, deployed on push to `dev`
  envs/prod/      api.docbot.space, promoted by digest with manual approval
```

Production stays on Railway until the DNS cutover (`docs/dns-cutover.md`). Nothing here creates DNS records or touches Railway or Vercel.

## Architecture (per environment)

```
Internet ──HTTPS──> ALB (<api_domain>, ACM cert, HTTP->HTTPS, idle 300s for SSE)
                     │  SG: 80/443 from anywhere
                     v
              ECS Fargate service (1 task, 1 vCPU / 2 GB, public IP, no NAT)
                     │  SG: 8000 only from the ALB SG
          ┌──────────┼──────────────────────┐
          v          v                      v
   RDS Postgres 16   EFS (/docbot via       Secrets Manager
   SG: 5432 from     access point uid 1000) (app keys + RDS-managed master secret)
   task SG only      mounted at /data, CHROMA_DIR=/data/chroma
```

| | dev | prod |
|---|---|---|
| Fargate | Spot (interruptible) | on-demand only |
| RDS | db.t4g.micro, 3-day backups, no final snapshot | db.t4g.small, 14-day PITR, final snapshot, deletion protection |
| Multi-AZ RDS | off | **off** (cost; see `envs/prod/main.tf` for the trade-off) |
| ALB deletion protection | var (default on) | on |
| Logs | 14 days | 30 days |
| VPC | 10.40.0.0/16 | 10.41.0.0/16 |
| Deploy role trusts | `ref:refs/heads/dev` | `environment:production` |
| Image | built on push to `dev` | the same digest, promoted (no rebuild) |

- **Network:** 2 AZs, public subnets only, no NAT gateway (saves ~$33/mo per env). Tasks get a public IP for egress (ECR, Secrets Manager, LLM and E2B APIs) and accept traffic only from the ALB. Private subnets: DOCBOT-2521.
- **Chroma on EFS** is SQLite: keep `desired_count <= 1` (enforced by a variable validation) until the vector store moves to pgvector.

### How DATABASE_URL is built

RDS creates and rotates the master password itself (`manage_master_user_password = true`) and keeps it in a Secrets Manager secret holding JSON `{username, password}`. The task injects `DB_USER`/`DB_PASSWORD` via ECS JSON-key secret references; the container command URL-encodes them into `DATABASE_URL` and `exec`s the same uvicorn command as the Dockerfile. `PGSSLMODE=require` because RDS Postgres 16 requires TLS. Nobody copies or sees the password.

### Secrets

Terraform creates each secret in `app_secret_names` **empty** (`docbot-<env>/<name>`) and never reads or writes values, so nothing secret is in state. A task cannot start until every listed secret has a value (fails closed).

- dev: `groq_api_key`, `GEMINI_API_KEY`, `E2B_API_KEY`, `DB_ENCRYPTION_KEY` (a **new** dev Fernet key, never prod's), `OWNER_KEY` (new random value).
- prod: the list in `envs/prod/variables.tf`. Trim it to what Railway prod actually sets **before** the first prod apply. `DB_ENCRYPTION_KEY` must be the **existing** Railway prod key, unchanged, or saved DB connections cannot be decrypted.

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # dev DB_ENCRYPTION_KEY only
python -c "import secrets; print(secrets.token_urlsafe(32))"                               # OWNER_KEY
aws secretsmanager put-secret-value --secret-id <arn> --secret-string '<value>'
```

Non-secret env vars that differ from the defaults go in `extra_env` in `terraform.tfvars`.

## Spending guardrails (`account/budgets.tf`)

- `docbot-account-monthly-net`: default cost types (credits applied), `monthly_budget_usd` (default $150).
- `docbot-account-monthly-gross`: same limit, `include_credit = false`, so it shows true burn while the $5K credits pay the bill.
- Both email `budget_email` at 50%, 80%, 100% actual and 100% forecasted.
- Cost Anomaly Detection: AWS-services monitor + daily email when an anomaly's impact is >= `anomaly_threshold_usd` ($20). Accounts allow one services monitor; if `aws ce get-anomaly-monitors` already lists one, set `anomaly_monitor_arn` instead of creating another.
- AWS asks each email address to confirm the anomaly subscription; budgets email without confirmation.

## Founder steps, in order

Prerequisites: Terraform >= 1.9, AWS CLI, admin credentials (SSO) for the chosen account, region `us-east-1`. Every `backend.hcl` needs `bucket` and `region` from the bootstrap outputs.

1. **Bootstrap** (once, local state; keep `bootstrap/terraform.tfstate` outside the repo):
   ```bash
   cd infra/aws/bootstrap && terraform init
   terraform apply -var aws_region=us-east-1 -var state_bucket_name=docbot-tfstate-<account-id>
   ```
2. **Account:** fill `account/backend.hcl`, copy `terraform.tfvars.example` to `terraform.tfvars`, then
   ```bash
   cd infra/aws/account && terraform init -backend-config=backend.hcl && terraform plan -out=tfplan && terraform apply tfplan
   ```
3. **Dev:** fill `envs/dev/backend.hcl` and `terraform.tfvars`, then
   ```bash
   cd infra/aws/envs/dev && terraform init -backend-config=backend.hcl
   terraform apply -target=module.app.aws_acm_certificate.api
   terraform output acm_validation_records      # add these CNAMEs at the DNS host
   terraform plan -out=tfplan && terraform apply tfplan   # waits for ACM ISSUED
   ```
   Then: CNAME `api-dev.docbot.space` -> `terraform output -raw alb_dns_name`; fill the secrets (`terraform output app_secret_arns`); set GitHub repo **variables** `AWS_DEV_DEPLOY_ROLE_ARN`, `AWS_DEV_REGION` (and `AWS_DEV_API_URL=https://api-dev.docbot.space` for the post-deploy smoke test); push to `dev` or run "Deploy AWS dev".
4. **Prod:** same as dev in `envs/prod` (cert for `api.docbot.space`; do **not** point the CNAME at the ALB yet, that is the cutover). Set repo variables `AWS_PROD_DEPLOY_ROLE_ARN`, `AWS_PROD_REGION`, `AWS_PROD_API_URL`, and create the GitHub environment `production` (required reviewer: founder; deployment branches: `main` only).

### Already applied PR #86's flat `infra/aws/dev`?

The `moved` blocks in `envs/dev/moved.tf` carry the state into `module.app` with no changes. The ECR repo and OIDC provider move to `account/`: dev forgets them (`removed`, no destroy), so import them into `account` **before** its first apply:

```bash
cd infra/aws/account && terraform init -backend-config=backend.hcl
terraform import aws_ecr_repository.api docbot-api
terraform import aws_ecr_lifecycle_policy.api docbot-api
terraform import 'aws_iam_openid_connect_provider.github[0]' arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com
```

The dev plan then shows only the old `docbot-dev-monthly` budget being destroyed (account budgets replace it).

### Changing the task definition

`terraform apply` registers a new revision, but the service keeps the revision the workflow deployed (`ignore_changes = [task_definition]`). Re-run the deploy workflow to roll it out; it starts from the latest revision of the family.
