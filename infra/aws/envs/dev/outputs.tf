output "alb_dns_name" {
  description = "CNAME var.api_domain to this at your DNS host."
  value       = module.app.alb_dns_name
}

output "acm_validation_records" {
  value = module.app.acm_validation_records
}

output "deploy_role_arn" {
  description = "GitHub repo variable AWS_DEV_DEPLOY_ROLE_ARN."
  value       = module.app.deploy_role_arn
}

output "aws_region" {
  description = "GitHub repo variable AWS_DEV_REGION."
  value       = var.aws_region
}

output "ecr_repository_url" {
  value = data.aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  value = module.app.ecs_cluster_name
}

output "ecs_service_name" {
  value = module.app.ecs_service_name
}

output "task_definition_family" {
  value = module.app.task_definition_family
}

output "app_secret_arns" {
  value = module.app.app_secret_arns
}

output "rds_master_secret_arn" {
  value = module.app.rds_master_secret_arn
}

output "rds_instance_id" {
  value = module.app.rds_instance_id
}

output "efs_file_system_id" {
  value = module.app.efs_file_system_id
}

output "ops_task_definition_family" {
  description = "One-off ops/drill tasks (docs/backup-restore.md)."
  value       = module.app.ops_task_definition_family
}
