resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}-api"
  retention_in_days = var.log_retention_days
}

# Created empty on purpose: no placeholder value, so a task cannot start (and no
# guessable OWNER_KEY exists) until the founder puts real values. Terraform never
# reads or writes the values, so nothing secret lands in state.
resource "aws_secretsmanager_secret" "app" {
  for_each                = var.app_secret_names
  name                    = "${local.name}/${each.key}"
  recovery_window_in_days = 7
}

# ------------------------------------------------------------------- IAM roles
data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "read-app-secrets"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = concat(
        [for s in aws_secretsmanager_secret.app : s.arn],
        [aws_db_instance.main.master_user_secret[0].secret_arn],
      )
    }]
  })
}

# The app needs no AWS API access (EFS uses SG + access point, not IAM auth).
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

# ----------------------------------------------------------------- ECS service
resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
}

locals {
  db_secret_arn = aws_db_instance.main.master_user_secret[0].secret_arn

  # Assemble DATABASE_URL from the RDS-managed secret at container start.
  # Password is URL-encoded (it can contain URL-reserved punctuation).
  # Dockerfile CMD is replaced, but runs the same uvicorn command.
  start_command = <<-SH
    set -e
    DATABASE_URL="$(python -c 'import os, urllib.parse as u; e = os.environ; q = lambda k: u.quote(e[k], safe=""); print("postgresql://%s:%s@%s/%s" % (q("DB_USER"), q("DB_PASSWORD"), e["DB_HOST"], e["DB_NAME"]))')"
    export DATABASE_URL
    exec uvicorn api.index:app --host 0.0.0.0 --port "$PORT"
  SH

  plain_env = merge(var.extra_env, {
    PORT             = tostring(local.container_port)
    APP_AUTH_MODE    = "required"
    BILLING_ENFORCE  = "true"
    PUBLIC_DEMO_MODE = "false"
    APP_ENV          = "production" # hosted deploy: keeps DEV-ONLY escape hatches off (rbac_service.is_deployed_environment)
    ALLOWED_ORIGINS  = var.allowed_origins
    FRONTEND_URL     = var.frontend_url
    APP_BASE_URL     = "https://${var.api_domain}"
    CHROMA_DIR       = "/data/chroma"
    DB_HOST          = aws_db_instance.main.endpoint # host:port
    DB_NAME          = aws_db_instance.main.db_name
    PGSSLMODE        = "require" # read by asyncpg; RDS PG16 forces SSL
  })
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64" # msodbcsql18 is amd64-only
  }

  volume {
    name = "data"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.chroma.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.chroma.id
        iam             = "DISABLED"
      }
    }
  }

  container_definitions = jsonencode([{
    name         = local.container_name
    image        = "${var.ecr_repository_url}:${var.image_tag}"
    essential    = true
    command      = ["sh", "-c", local.start_command]
    portMappings = [{ containerPort = local.container_port, protocol = "tcp" }]
    environment  = [for k, v in local.plain_env : { name = k, value = v }]
    secrets = concat(
      [for k, s in aws_secretsmanager_secret.app : { name = k, valueFrom = s.arn }],
      [
        { name = "DB_USER", valueFrom = "${local.db_secret_arn}:username::" },
        { name = "DB_PASSWORD", valueFrom = "${local.db_secret_arn}:password::" },
      ],
    )
    mountPoints = [{ sourceVolume = "data", containerPath = "/data", readOnly = false }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = data.aws_region.current.region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name                              = "${local.name}-api"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = var.desired_count
  health_check_grace_period_seconds = 180

  capacity_provider_strategy {
    capacity_provider = var.use_fargate_spot ? "FARGATE_SPOT" : "FARGATE"
    weight            = 1
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = true # no NAT gateway
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = local.container_name
    container_port   = local.container_port
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [aws_lb_listener.https, aws_ecs_cluster_capacity_providers.main, aws_efs_mount_target.chroma]

  lifecycle {
    # The GitHub workflow registers new revisions; don't roll them back on apply.
    ignore_changes = [task_definition]
  }
}
