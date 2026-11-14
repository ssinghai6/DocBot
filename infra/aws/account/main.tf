# Account-wide, env-independent resources shared by envs/dev and envs/prod:
# the GitHub OIDC provider (one per account) and the ECR repo (dev builds an
# image once; prod promotes the same digest). Apply after bootstrap, before envs.

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  backend "s3" {}
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = { Project = "docbot", Environment = "shared", ManagedBy = "terraform" }
  }
}

variable "aws_region" {
  type = string
}

variable "create_github_oidc_provider" {
  type        = bool
  default     = true
  description = "False if the account already has token.actions.githubusercontent.com (then nothing to create; envs look it up)."
}

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_github_oidc_provider ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

resource "aws_ecr_repository" "api" {
  name                 = "docbot-api"
  image_tag_mutability = "MUTABLE" # dev moves the `dev` tag; deploys pin SHA tags / digests
  image_scanning_configuration {
    scan_on_push = true
  }
  lifecycle {
    prevent_destroy = true
  }
}

# Rule 1 protects anything prod ever ran (tagged prod-<sha> by deploy-aws-prod):
# an image matched by a higher-priority rule is never expired by a lower one.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 30 images prod ran (rollback targets)"
        selection    = { tagStatus = "tagged", tagPrefixList = ["prod-"], countType = "imageCountMoreThan", countNumber = 30 }
        action       = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep the last 30 other images"
        selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 30 }
        action       = { type = "expire" }
      },
    ]
  })
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "github_oidc_provider_arn" {
  value = one(aws_iam_openid_connect_provider.github[*].arn)
}
