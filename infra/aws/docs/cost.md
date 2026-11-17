# AWS cost: estimate and monthly evidence (DOCBOT-2507)

Estimates are us-east-1 on-demand list prices for what `envs/dev` and `envs/prod`
create, before credits. They are not a quote: the monthly Cost Explorer numbers
below are the evidence. Re-check list prices when instance sizes change.

## Estimate per month

| Item | dev 24/7 | dev idle (`desired_count = 0`) | prod |
|---|---|---|---|
| ALB (hours + low LCU) | ~$18 | ~$18 | ~$18 |
| Fargate 1 vCPU / 2 GB, 1 task | ~$11 (Spot) | $0 | ~$36 (on-demand) |
| RDS instance | ~$12 (db.t4g.micro) | ~$12 | ~$23 (db.t4g.small, single-AZ) |
| RDS storage 20 GB gp3 + backups | ~$3 | ~$3 | ~$4 (14-day PITR) |
| Public IPv4 ($0.005/h each: 2 ALB + 1 task) | ~$11 | ~$7 | ~$11 |
| Secrets Manager ($0.40 each) | ~$2.40 (5 + RDS) | ~$2.40 | ~$6 (14 + RDS) |
| EFS + AWS Backup, CloudWatch logs + alarms, SNS | ~$2 | ~$1 | ~$3 |
| **Total** | **~$60** | **~$43** | **~$101** |

Shared (`account/` + bootstrap): S3 state, DynamoDB lock, ECR storage (~30 images
kept per lifecycle rule), migration bucket: **~$3**.

Planned run rate: dev idle or 24/7 + prod + shared = **~$148-165/mo**. The $5K AWS
credits cover roughly 30 months at that rate. Budgets in `account/budgets.tf`:
account `monthly_budget_usd` default $250 (net and gross), dev $60 gross, prod $150
gross (`env_budgets_usd`).

Not in the estimate: data transfer out (small at current traffic), Multi-AZ RDS
(`db_multi_az = true` roughly doubles the RDS line), NAT gateway (none; ~$33/mo per
env if private subnets arrive, DOCBOT-2521).

## Scale dev to zero

`desired_count` in `envs/dev/terraform.tfvars` (module validation: 0 or 1). The
founder applies:

```bash
cd infra/aws/envs/dev
terraform plan -var desired_count=0 -out=tfplan && terraform apply tfplan   # or set it in terraform.tfvars
```

At 0 the `running-below-desired` alarm is not created (it would fire forever). ALB,
RDS, EFS and IPv4 still bill. Scale back with `desired_count = 1`. A quick
temporary stop without Terraform: `aws ecs update-service --cluster docbot-dev
--service docbot-dev-api --desired-count 0` (the next apply resets it to the tfvars value).

## Monthly evidence (gross, credits excluded)

`RECORD_TYPE` excludes credits and refunds so the number is real burn even while
credits pay. Activate `Environment` as a cost allocation tag first (Billing -> Cost
allocation tags), or the per-env split shows everything as untagged. Read-only calls;
Cost Explorer API costs $0.01 per request.

```bash
START=2026-12-01; END=2027-01-01
FILTER='{"Not":{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit","Refund"]}}}'

# by environment tag (dev / prod / shared)
aws ce get-cost-and-usage --time-period Start=$START,End=$END --granularity MONTHLY \
  --metrics UnblendedCost --filter "$FILTER" --group-by Type=TAG,Key=Environment \
  --query 'ResultsByTime[0].Groups[].[Keys[0],Metrics.UnblendedCost.Amount]' --output table

# by service
aws ce get-cost-and-usage --time-period Start=$START,End=$END --granularity MONTHLY \
  --metrics UnblendedCost --filter "$FILTER" --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[0].Groups[].[Keys[0],Metrics.UnblendedCost.Amount]' --output table

# credits used this month
aws ce get-cost-and-usage --time-period Start=$START,End=$END --granularity MONTHLY \
  --metrics UnblendedCost --filter '{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit"]}}'
```

| Month | dev | prod | shared | Total gross | Credits used | Credits left | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
