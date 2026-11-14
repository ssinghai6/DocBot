terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  # Partial config: values come from backend.hcl (terraform init -backend-config=backend.hcl).
  backend "s3" {}
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = { Project = "docbot", Environment = "prod", ManagedBy = "terraform" }
  }
}

# Created by infra/aws/account.
data "aws_ecr_repository" "api" {
  name = "docbot-api"
}

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}
