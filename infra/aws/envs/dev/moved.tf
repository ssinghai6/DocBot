# DOCBOT-2508: state written by PR #86's flat infra/aws/dev moves into module.app
# with no changes. No-ops if dev was never applied. Delete after the first apply.
moved {
  from = aws_vpc.main
  to   = module.app.aws_vpc.main
}
moved {
  from = aws_internet_gateway.main
  to   = module.app.aws_internet_gateway.main
}
moved {
  from = aws_subnet.public
  to   = module.app.aws_subnet.public
}
moved {
  from = aws_route_table.public
  to   = module.app.aws_route_table.public
}
moved {
  from = aws_route_table_association.public
  to   = module.app.aws_route_table_association.public
}
moved {
  from = aws_default_security_group.default
  to   = module.app.aws_default_security_group.default
}
moved {
  from = aws_security_group.alb
  to   = module.app.aws_security_group.alb
}
moved {
  from = aws_security_group.task
  to   = module.app.aws_security_group.task
}
moved {
  from = aws_security_group.db
  to   = module.app.aws_security_group.db
}
moved {
  from = aws_security_group.efs
  to   = module.app.aws_security_group.efs
}
moved {
  from = aws_acm_certificate.api
  to   = module.app.aws_acm_certificate.api
}
moved {
  from = aws_acm_certificate_validation.api
  to   = module.app.aws_acm_certificate_validation.api
}
moved {
  from = aws_lb.api
  to   = module.app.aws_lb.api
}
moved {
  from = aws_lb_target_group.api
  to   = module.app.aws_lb_target_group.api
}
moved {
  from = aws_lb_listener.http
  to   = module.app.aws_lb_listener.http
}
moved {
  from = aws_lb_listener.https
  to   = module.app.aws_lb_listener.https
}
moved {
  from = aws_db_subnet_group.main
  to   = module.app.aws_db_subnet_group.main
}
moved {
  from = aws_db_instance.main
  to   = module.app.aws_db_instance.main
}
moved {
  from = aws_efs_file_system.chroma
  to   = module.app.aws_efs_file_system.chroma
}
moved {
  from = aws_efs_mount_target.chroma
  to   = module.app.aws_efs_mount_target.chroma
}
moved {
  from = aws_efs_access_point.chroma
  to   = module.app.aws_efs_access_point.chroma
}
moved {
  from = aws_cloudwatch_log_group.api
  to   = module.app.aws_cloudwatch_log_group.api
}
moved {
  from = aws_secretsmanager_secret.app
  to   = module.app.aws_secretsmanager_secret.app
}
moved {
  from = aws_iam_role.execution
  to   = module.app.aws_iam_role.execution
}
moved {
  from = aws_iam_role_policy_attachment.execution_managed
  to   = module.app.aws_iam_role_policy_attachment.execution_managed
}
moved {
  from = aws_iam_role_policy.execution_secrets
  to   = module.app.aws_iam_role_policy.execution_secrets
}
moved {
  from = aws_iam_role.task
  to   = module.app.aws_iam_role.task
}
moved {
  from = aws_ecs_cluster.main
  to   = module.app.aws_ecs_cluster.main
}
moved {
  from = aws_ecs_cluster_capacity_providers.main
  to   = module.app.aws_ecs_cluster_capacity_providers.main
}
moved {
  from = aws_ecs_task_definition.api
  to   = module.app.aws_ecs_task_definition.api
}
moved {
  from = aws_ecs_service.api
  to   = module.app.aws_ecs_service.api
}
moved {
  from = aws_iam_role.deploy
  to   = module.app.aws_iam_role.deploy
}
moved {
  from = aws_iam_role_policy.deploy
  to   = module.app.aws_iam_role_policy.deploy
}

# Now owned by infra/aws/account: forget them here without destroying, then
# import them there (see infra/aws/README.md "Already applied PR #86?").
removed {
  from = aws_ecr_repository.api
  lifecycle {
    destroy = false
  }
}
removed {
  from = aws_ecr_lifecycle_policy.api
  lifecycle {
    destroy = false
  }
}
removed {
  from = aws_iam_openid_connect_provider.github
  lifecycle {
    destroy = false
  }
}
# aws_budgets_budget.monthly (old dev budget) is intentionally destroyed: the
# account budgets in infra/aws/account replace it.
