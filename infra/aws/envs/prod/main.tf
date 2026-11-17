# Production. Differences from dev, on purpose:
#   - on-demand Fargate only (no Spot interruptions mid-stream)
#   - RDS: 14-day PITR backups, deletion protection, final snapshot on destroy
#   - ALB deletion protection
#   - Multi-AZ RDS is OFF to save ~$25/mo at db.t4g.small. A single-AZ outage means
#     restoring from PITR into the other AZ (RTO ~30-60 min). Flip db_multi_az
#     to true when revenue justifies it; the change is applied in place with a
#     short failover.
module "app" {
  source = "../../modules/app"

  environment     = "prod"
  vpc_cidr        = "10.41.0.0/16"
  api_domain      = var.api_domain
  frontend_url    = var.frontend_url
  allowed_origins = var.allowed_origins

  desired_count       = var.desired_count
  use_fargate_spot    = false
  deletion_protection = true
  task_cpu            = var.task_cpu
  task_memory         = var.task_memory

  db_instance_class        = var.db_instance_class
  db_multi_az              = var.db_multi_az
  db_backup_retention_days = 14
  db_skip_final_snapshot   = false
  log_retention_days       = 30

  alarm_email      = var.alarm_email
  app_secret_names = var.app_secret_names
  extra_env        = var.extra_env

  ecr_repository_url = data.aws_ecr_repository.api.repository_url
  ecr_repository_arn = data.aws_ecr_repository.api.arn
  image_tag          = "prod" # never pushed: deploy-aws-prod.yml registers api + ops revisions by digest. After an apply that changes a task def, re-run the last promote before using ops tasks.

  github_oidc_provider_arn = data.aws_iam_openid_connect_provider.github.arn
  # Only jobs that run in the GitHub environment "aws-prod" (required reviewer,
  # deployment branch main) can assume this role. Not "production": Vercel already
  # owns that environment and it has no protection rules.
  github_oidc_subjects   = ["repo:${var.github_repo}:environment:aws-prod"]
  deploy_can_push_images = false # prod never builds; it promotes a dev-tested digest
}
