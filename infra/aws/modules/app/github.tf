# GitHub Actions OIDC -> per-env deploy role. Only the exact sub claims in
# var.github_oidc_subjects can assume it (dev: refs/heads/dev; prod:
# environment:production). The OIDC provider itself is account-wide (infra/aws/account).

data "aws_iam_policy_document" "deploy_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.github_oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = var.github_oidc_subjects
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${local.name}-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.deploy_assume.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"] # does not support resource scoping
    resources = ["*"]
  }
  statement {
    # Read + retag (PutImage with an existing manifest) is all a promote needs.
    sid = "EcrOneRepo"
    actions = concat(
      ["ecr:BatchGetImage", "ecr:DescribeImages", "ecr:PutImage", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer"],
      var.deploy_can_push_images ? ["ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"] : [],
    )
    resources = [var.ecr_repository_arn]
  }
  statement {
    sid       = "EcsService"
    actions   = ["ecs:DescribeServices", "ecs:UpdateService"]
    resources = [aws_ecs_service.api.arn]
  }
  statement {
    # Register/DescribeTaskDefinition do not support resource-level permissions.
    sid       = "EcsTaskDefinition"
    actions   = ["ecs:RegisterTaskDefinition", "ecs:DescribeTaskDefinition"]
    resources = ["*"]
  }
  statement {
    sid       = "PassTaskRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy-${local.name}"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}
