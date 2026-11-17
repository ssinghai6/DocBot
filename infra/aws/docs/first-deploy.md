# First deploy to AWS dev (DOCBOT-2505)

Prerequisite: `envs/dev` applied, ACM certificate issued, `api-dev.docbot.space`
CNAME pointing at the ALB, every dev secret filled. Steps and verification for
those: [founder-wave-a-checklist.md](founder-wave-a-checklist.md). The founder
does all console, DNS and `terraform apply` steps.

## 1. GitHub repository variables

Settings -> Secrets and variables -> Actions -> **Variables** (not secrets; none of these are secret):

| Variable | Value | Used by |
|---|---|---|
| `AWS_DEV_DEPLOY_ROLE_ARN` | `terraform output -raw deploy_role_arn` (`arn:aws:iam::<ACCOUNT_ID>:role/docbot-dev-github-deploy`) | OIDC login |
| `AWS_DEV_REGION` | `us-east-1` (`terraform output -raw aws_region`) | OIDC login |
| `AWS_DEV_API_URL` | `https://api-dev.docbot.space` | smoke test; without it the run skips the smoke test **and** never tags `tested-<sha>`, so nothing becomes promotable to prod |
| `AWS_DEV_FRONTEND_ORIGIN` | optional; the dev frontend origin (must be in `allowed_origins`) | CORS check in the smoke test |

```bash
gh variable set AWS_DEV_DEPLOY_ROLE_ARN --body "arn:aws:iam::<ACCOUNT_ID>:role/docbot-dev-github-deploy"
gh variable set AWS_DEV_REGION --body us-east-1
gh variable set AWS_DEV_API_URL --body https://api-dev.docbot.space
```

Without the first two, "Deploy AWS dev" ends with a "skipped" notice instead of failing.

## 2. Run the deploy

The workflow runs on every push to `dev`, or by hand (only from `dev`: the role trusts `ref:refs/heads/dev` only):

```bash
gh workflow run "Deploy AWS dev" --ref dev
gh run watch "$(gh run list --workflow 'Deploy AWS dev' --limit 1 --json databaseId -q '.[0].databaseId')"
```

What it does (`.github/workflows/deploy-aws-dev.yml`): builds `linux/amd64`, pushes
`docbot-api:<sha>` and `:dev`, renders the latest `docbot-dev-api` task definition
with the new image, deploys and waits for stability (circuit breaker rolls back
tasks that never get healthy), runs `infra/aws/smoke-test.sh`, then tags the
same digest `tested-<sha>` and prints `Promotable to prod: <sha> -> <digest>`.

The first deploy is slow: the image build is cold and the task has a 180 s health
grace period.

## 3. Automated smoke checks

Both run without credentials, from any laptop:

```bash
infra/aws/smoke-test.sh https://api-dev.docbot.space            # health+db, http->https, latency < 3 s, CORS
infra/aws/scripts/smoke.sh https://api-dev.docbot.space         # health, CSP, /v1 401, unsigned webhook rejected, demo init
```

## 4. Manual checks (rest of the 2505 acceptance)

Need a signed-in user (`APP_AUTH_MODE=required` on AWS) through the dev frontend.

- [ ] Upload a PDF, ask a question, get a cited answer.
- [ ] Upload a CSV, ask an aggregate question (E2B sandbox path), get a table or chart.
- [ ] A long Autopilot answer streams for 120 s+ without being cut (ALB idle timeout is 300 s).
- [ ] EFS persistence: note a session id, force a new deployment
      (`aws ecs update-service --cluster docbot-dev --service docbot-dev-api --force-new-deployment`),
      wait for stability, then the same session still answers from its document.
- [ ] Logs: `aws logs tail /ecs/docbot-dev-api --since 30m` shows no tracebacks at startup.
- [ ] `TRUSTED_PROXY_HOPS`: find the `xff-shape:` line
      (`aws logs tail /ecs/docbot-dev-api --since 1h | grep xff-shape`) for a request through
      the Vercel frontend; `chosen_index` must point at the client IP. If not, change
      `trusted_proxy_hops` in `envs/dev/terraform.tfvars` and re-apply.
- [ ] Alarm email confirmed (SNS subscription), if `alarm_email` is set.

## Troubleshooting

| Symptom | Look at |
|---|---|
| Tasks stop with `ResourceInitializationError` | An app secret is empty: `aws secretsmanager list-secret-version-ids --secret-id <arn> --query 'length(Versions)'` is 0 for it (ARNs from `terraform output app_secret_arns`; this never prints values). |
| Deploy waits, then circuit breaker rolls back | `aws ecs describe-services --cluster docbot-dev --services docbot-dev-api --query 'services[0].events[:5]'` and the task logs. |
| `/api/health` returns 503 `db: error` | RDS SG/endpoint or the RDS secret; check `DB_HOST` in the task definition. |
| Workflow: `Not authorized to perform sts:AssumeRoleWithWebIdentity` | Run not on `dev`, or the role ARN variable is wrong. |
| Changed Terraform task definition not live | `ignore_changes = [task_definition]`: re-run the workflow. |
