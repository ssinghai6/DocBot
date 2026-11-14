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
    tags = { Project = "docbot", Environment = "dev", ManagedBy = "terraform" }
  }
}

locals {
  name           = "docbot-dev"
  container_name = "api"
  container_port = 8000
}

data "aws_availability_zones" "available" {
  state = "available"
}
