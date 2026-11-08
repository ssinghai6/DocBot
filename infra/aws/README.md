# DocBot on AWS: dev environment

This sets up a small, cheap AWS **dev** copy of the DocBot API, deployed from the `dev` branch.

**Production is not touched.** `www.docbot.space` stays on Railway (API) and Vercel (frontend). Nothing here creates DNS records or changes Railway or Vercel production settings. The deploy workflow never runs on `main`, and its IAM role only trusts the `dev` branch.

## Architecture

```
Internet ──HTTPS──> ALB (api-dev.docbot.space, ACM cert, HTTP->HTTPS, idle 300s for SSE)
                     │  SG: 80/443 from anywhere
                     v
              ECS Fargate service (1 task, 1 vCPU / 2 GB, Spot by default, public IP, no NAT)
                     │  SG: 8000 only from the ALB SG
          ┌──────────┼──────────────────────┐
          v          v                      v
   RDS Postgres 16   EFS (/docbot via       Secrets Manager
   db.t4g.micro      access point uid 1000) (app keys + RDS-managed master secret)
   SG: 5432 from     mounted at /data,
   task SG only      CHROMA_DIR=/data/chroma
```

- **Network:** one VPC across 2 AZs with public subnets only. There is no NAT gateway, which saves about $33/mo. Tasks get a public IP so they can reach ECR, Secrets Manager and the LLM and E2B APIs. Inbound traffic reaches them only through the ALB.
- **Image:** the existing `Dockerfile`, pushed to ECR repo `docbot-api` (scan on push, keeps the last 20 images). It is tagged with the git SHA, and `dev` is moved to the latest build.
- **Deploys:** `.github/workflows/deploy-aws-dev.yml` runs on push to `dev` or by manual dispatch. It uses GitHub OIDC (no stored AWS keys), builds and pushes the image, registers a new task-definition revision and waits for the service to become stable. The ECS deployment circuit breaker rolls back automatically if new tasks fail the `/api/health` check.
- **Logs:** CloudWatch log group `/ecs/docbot-dev-api`, kept for 14 days.
- **Budget:** an AWS Budgets monthly budget (default $60) emails you at 80% of actual spend and at 100% of forecast spend.

### How DATABASE_URL is built

RDS creates and rotates the master password itself (`manage_master_user_password = true`) and stores it in a Secrets Manager secret holding JSON `{username, password}`. The task definition injects `username` and `password` as `DB_USER` and `DB_PASSWORD` using ECS JSON-key secret references. The container command then:

1. builds `DATABASE_URL=postgresql://user:<url-encoded password>@<endpoint>/docbot`, and
2. `exec`s the same `uvicorn api.index:app` command as the Dockerfile.

No code change is needed, and nobody ever has to copy or see the password. `PGSSLMODE=require` is set because RDS Postgres 16 requires TLS by default.

**Rotation caveat:** RDS rotates the password every 7 days by default, and a running task keeps the old one. Connections that are already open keep working. Once a new connection fails, `/api/health` returns 503, the ALB marks the task unhealthy and ECS replaces it. The replacement reads the new password. In dev this means a short blip about once a week at most. If that becomes annoying, run `aws ecs update-service --force-new-deployment` after rotation, or lengthen the rotation schedule on the RDS secret.

### Secrets

Terraform creates these secrets **empty**, with no value and no placeholder. It never reads or writes their values, so no secret value ends up in Terraform state. A task cannot start until every one of them has a value. This is intentional: it fails closed, and there is never a guessable placeholder `OWNER_KEY`.

| Secret (`docbot-dev/<name>`) | Value |
|---|---|
| `groq_api_key` | Groq key |
| `GEMINI_API_KEY` | Gemini key |
| `E2B_API_KEY` | E2B key |
| `DB_ENCRYPTION_KEY` | **A new Fernet key made only for dev.** Never reuse the production key. Generate it with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `OWNER_KEY` | A new random value: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |

To add more, for example `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `STRIPE_SECRET_KEY` or `RESEND_API_KEY`: add the name to `app_secret_names` in `terraform.tfvars`, apply, fill the value and redeploy. If you list a secret and leave it empty, tasks will not start.

Plain environment variables set by Terraform: `APP_AUTH_MODE=required`, `BILLING_ENFORCE=true`, `PUBLIC_DEMO_MODE=false`, `APP_ENV=production` (so dev-only escape hatches such as anonymous sandbox stay off on this public host), `ALLOWED_ORIGINS`, `FRONTEND_URL`, `APP_BASE_URL=https://<api_domain>`, `CHROMA_DIR=/data/chroma`, `PORT=8000`, `PGSSLMODE=require`.

## Monthly cost estimate (us-east-1 on-demand prices, other regions are within about 10–20%)

| Item | Running 24/7 (Spot) | Idle (`desired_count = 0`) |
|---|---|---|
| Fargate 1 vCPU / 2 GB, FARGATE_SPOT (about $36 on regular FARGATE) | ~$11 | $0 |
| ALB (hourly charge plus light LCU use) | ~$18 | ~$17 |
| RDS db.t4g.micro single-AZ + 20 GB gp3 + backups | ~$14 | ~$14 |
| Public IPv4 (2 for the ALB, 1 for the task, $3.65 each) | ~$11 | ~$7 |
| Secrets Manager (6 secrets × $0.40) | ~$2.4 | ~$2.4 |
| ECR (up to 20 images of about 1 GB each) | ~$1–2 | ~$1–2 |
| EFS, CloudWatch Logs, S3 state bucket, DynamoDB lock table | ~$1–2 | ~$1 |
| **Total** | **~$58–60** | **~$43–45** |

With `use_fargate_spot = false` the 24/7 total is about $85. You can stop RDS for up to 7 days at a time (`aws rds stop-db-instance`) to save another ~$12 while idle. AWS starts it again automatically after 7 days.

## Founder steps, in order

Prerequisites: Terraform >= 1.9, the AWS CLI, and admin credentials for the chosen account. Never use the production Railway account or keys.

1. **Choose the AWS account and region.** All commands below use that region.
2. **Bootstrap remote state** (one time, local state):
   ```bash
   cd infra/aws/bootstrap
   terraform init
   terraform apply -var aws_region=<region> -var state_bucket_name=docbot-tfstate-<account-id>
   ```
   Keep `infra/aws/bootstrap/terraform.tfstate` safe (it is gitignored).
3. **Fill in `infra/aws/dev/backend.hcl`** with the `state_bucket`, `region` and `lock_table` values from the bootstrap outputs.
4. **Create `infra/aws/dev/terraform.tfvars`** from `terraform.tfvars.example`. Set `aws_region`, `budget_email` and `frontend_url`/`allowed_origins` (the Vercel Preview URL for branch `dev`). If the account already has a GitHub OIDC provider, set `create_github_oidc_provider = false`.
5. **Init, then create the certificate first** so you can get the validation records:
   ```bash
   cd infra/aws/dev
   terraform init -backend-config=backend.hcl
   terraform apply -target=aws_acm_certificate.api
   terraform output acm_validation_records
   ```
6. **Add the ACM validation CNAME(s)** from step 5 at your DNS host. DNS is not on Route53.
7. **Plan and apply everything:**
   ```bash
   terraform plan -out=tfplan
   terraform apply tfplan
   ```
   The apply waits at `aws_acm_certificate_validation` until ACM sees the CNAME, usually a few minutes. The ECS service is created, but its task cannot start yet because there is no image and the secrets are empty. That is expected.
8. **Point `api-dev` at the ALB:** at your DNS host, add a CNAME from `api-dev.docbot.space` to `terraform output -raw alb_dns_name`.
9. **Fill in the secrets** (see the Secrets section above). Use a **new** dev `DB_ENCRYPTION_KEY`:
   ```bash
   terraform output app_secret_arns
   aws secretsmanager put-secret-value --secret-id <arn> --secret-string '<value>'   # once per secret
   ```
10. **Set the GitHub repo variables** (Settings → Secrets and variables → Actions → Variables. These are variables, not secrets):
    - `AWS_DEV_DEPLOY_ROLE_ARN` = `terraform output -raw deploy_role_arn`
    - `AWS_DEV_REGION` = `terraform output -raw aws_region`
11. **Push to `dev`**, or run the "Deploy AWS dev" workflow on the `dev` branch. It builds, pushes and deploys, then waits for the service to become stable.
12. **Verify:** `curl https://api-dev.docbot.space/api/health` should return `{"status":"ok", ..., "db":"ok"}`.
13. **Point the Vercel Preview environment for branch `dev`** at this API by setting `RAILWAY_BACKEND_URL=https://api-dev.docbot.space` on Preview, scoped to branch `dev` only. Do not change Production env vars.
14. **Scale to 0 when idle:** set `desired_count = 0` in `terraform.tfvars` and run `terraform apply`. Set it back to 1 to resume. You can also run `aws ecs update-service --cluster docbot-dev --service docbot-dev-api --desired-count 0`, but the next `terraform apply` resets it to the tfvars value.

### Notes

- **Changing the task definition in Terraform** (env vars, CPU, secrets): `terraform apply` registers a new revision, but the service keeps running the revision the workflow deployed (`ignore_changes = [task_definition]`). Run the workflow again (workflow_dispatch on `dev`) to roll it out. The workflow always starts from the latest revision of the family.
- **Tearing down:** set `deletion_protection = false`, apply, then run `terraform destroy`. The RDS instance is deleted without a final snapshot. Secrets have a 7-day recovery window.
- **Chroma on EFS:** Chroma uses SQLite, which is safe on EFS with a single task. Keep `desired_count <= 1` unless the vector store moves off local files.
- **Spot interruptions:** with `use_fargate_spot = true`, AWS can reclaim the task with 2 minutes' notice. ECS starts a replacement automatically, but in-flight streams are cut. Set it to `false` if that disrupts demos.
