output "alb_dns_name" {
  description = "CNAME var.api_domain to this at your DNS host."
  value       = aws_lb.api.dns_name
}

output "acm_validation_records" {
  description = "Add these CNAMEs at your DNS host so ACM can issue the certificate."
  value = [for o in aws_acm_certificate.api.domain_validation_options : {
    name  = o.resource_record_name
    type  = o.resource_record_type
    value = o.resource_record_value
  }]
}

output "deploy_role_arn" {
  value = aws_iam_role.deploy.arn
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}

output "task_definition_family" {
  value = aws_ecs_task_definition.api.family
}

output "app_secret_arns" {
  description = "Fill each with: aws secretsmanager put-secret-value --secret-id <arn> --secret-string '<value>'"
  value       = { for k, s in aws_secretsmanager_secret.app : k => s.arn }
}

output "rds_master_secret_arn" {
  description = "RDS-managed master credentials (rotated by RDS)."
  value       = aws_db_instance.main.master_user_secret[0].secret_arn
}

output "rds_instance_id" {
  value = aws_db_instance.main.identifier
}

output "efs_file_system_id" {
  value = aws_efs_file_system.chroma.id
}
