# DOCBOT-2509: a running task keeps the secret values it started with. When a
# secret changes, force a new ECS deployment so tasks pick up the new value.
# EventBridge -> Step Functions (SDK integration ecs:UpdateService), no code.
#
# Triggers:
#  - RotationSucceeded on the RDS-managed master secret ("AWS Service Event via
#    CloudTrail": delivered without a trail).
#  - PutSecretValue/UpdateSecret on docbot-<env>/* app secrets ("AWS API Call via
#    CloudTrail": only delivered once a CloudTrail trail exists, DOCBOT-2524;
#    until then run the manual force-new-deployment in docs/secrets-rotation.md).

data "aws_iam_policy_document" "states_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "redeploy" {
  name               = "${local.name}-redeploy"
  assume_role_policy = data.aws_iam_policy_document.states_assume.json
}

resource "aws_iam_role_policy" "redeploy" {
  name = "update-service"
  role = aws_iam_role.redeploy.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["ecs:UpdateService"], Resource = [aws_ecs_service.api.arn] }]
  })
}

resource "aws_sfn_state_machine" "redeploy" {
  name     = "${local.name}-redeploy-on-secret-change"
  role_arn = aws_iam_role.redeploy.arn
  definition = jsonencode({
    Comment = "Force a new ECS deployment so tasks re-read Secrets Manager"
    StartAt = "ForceNewDeployment"
    States = {
      ForceNewDeployment = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:ecs:updateService"
        Parameters = {
          Cluster            = aws_ecs_cluster.main.name
          Service            = aws_ecs_service.api.name
          ForceNewDeployment = true
        }
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 30, MaxAttempts = 3, BackoffRate = 2 }]
        End   = true
      }
    }
  })
}

data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "events_redeploy" {
  name               = "${local.name}-events-redeploy"
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

resource "aws_iam_role_policy" "events_redeploy" {
  name = "start-redeploy"
  role = aws_iam_role.events_redeploy.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["states:StartExecution"], Resource = [aws_sfn_state_machine.redeploy.arn] }]
  })
}

locals {
  secret_change_patterns = {
    rds-rotated = {
      source        = ["aws.secretsmanager"]
      "detail-type" = ["AWS Service Event via CloudTrail"]
      detail = {
        eventName           = ["RotationSucceeded"]
        additionalEventData = { SecretId = [local.db_secret_arn] }
      }
    }
    app-secret-changed = {
      source        = ["aws.secretsmanager"]
      "detail-type" = ["AWS API Call via CloudTrail"]
      detail = {
        eventName = ["PutSecretValue", "UpdateSecret"]
        # secretId is whatever the caller passed: a name or an ARN.
        requestParameters = { secretId = [{ prefix = "${local.name}/" }, { wildcard = "arn:aws:secretsmanager:*:secret:${local.name}/*" }] }
      }
    }
  }
}

resource "aws_cloudwatch_event_rule" "secret_change" {
  for_each      = local.secret_change_patterns
  name          = "${local.name}-${each.key}"
  event_pattern = jsonencode(each.value)
}

resource "aws_cloudwatch_event_target" "secret_change" {
  for_each = local.secret_change_patterns
  rule     = aws_cloudwatch_event_rule.secret_change[each.key].name
  arn      = aws_sfn_state_machine.redeploy.arn
  role_arn = aws_iam_role.events_redeploy.arn
}
