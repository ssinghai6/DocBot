terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

locals {
  name           = "docbot-${var.environment}"
  container_name = "api"
  container_port = 8000
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_region" "current" {}
