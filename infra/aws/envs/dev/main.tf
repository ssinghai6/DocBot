module "app" {
  source = "../../modules/app"

  environment     = "dev"
  vpc_cidr        = "10.40.0.0/16"
  api_domain      = var.api_domain
  frontend_url    = var.frontend_url
  allowed_origins = var.allowed_origins

  desired_count       = var.desired_count
  use_fargate_spot    = var.use_fargate_spot
  deletion_protection = var.deletion_protection

  db_instance_class        = "db.t4g.micro"
  db_backup_retention_days = 3
  db_skip_final_snapshot   = true # dev data; deletion_protection is the guard
  log_retention_days       = 14

  alarm_email        = var.alarm_email
  db_engine_version  = var.db_engine_version
  trusted_proxy_hops = var.trusted_proxy_hops
  app_secret_names   = var.app_secret_names
  extra_env          = var.extra_env

  ecr_repository_url = data.aws_ecr_repository.api.repository_url
  ecr_repository_arn = data.aws_ecr_repository.api.arn
  image_tag          = "dev"

  github_oidc_provider_arn = data.aws_iam_openid_connect_provider.github.arn
  github_oidc_subjects     = ["repo:${var.github_repo}:ref:refs/heads/dev"]
  deploy_can_push_images   = true
}
