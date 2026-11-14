variable "environment" {
  type        = string
  description = "Short env name (dev, prod). Resource names are docbot-<environment>."
}

variable "vpc_cidr" {
  type        = string
  description = "VPC CIDR; use a different /16 per env."
}

variable "api_domain" {
  type        = string
  description = "Public API hostname (ACM cert + APP_BASE_URL)."
}

variable "frontend_url" {
  type        = string
  description = "FRONTEND_URL."
}

variable "allowed_origins" {
  type        = string
  description = "ALLOWED_ORIGINS for CORS (comma-separated)."
}

variable "desired_count" {
  type        = number
  description = "Number of API tasks. 0 stops compute. Keep <= 1 while Chroma lives on EFS (SQLite)."
  validation {
    condition     = var.desired_count >= 0 && var.desired_count <= 1
    error_message = "desired_count must be 0 or 1 while Chroma is on EFS."
  }
}

variable "use_fargate_spot" {
  type        = bool
  description = "FARGATE_SPOT (cheaper, interruptible). Never for prod."
}

variable "task_cpu" {
  type    = number
  default = 1024
}

variable "task_memory" {
  type    = number
  default = 2048
}

variable "deletion_protection" {
  type        = bool
  description = "Deletion protection on RDS and the ALB."
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "db_allocated_storage" {
  type    = number
  default = 20
}

variable "db_max_allocated_storage" {
  type        = number
  default     = 100
  description = "RDS storage autoscaling ceiling (GB)."
}

variable "db_multi_az" {
  type        = bool
  default     = false
  description = "Multi-AZ RDS (~2x DB cost)."
}

variable "db_backup_retention_days" {
  type        = number
  description = "RDS automated backup / PITR window (days)."
  validation {
    condition     = var.db_backup_retention_days >= 1 && var.db_backup_retention_days <= 35
    error_message = "Must be 1-35 days."
  }
}

variable "db_skip_final_snapshot" {
  type = bool
}

variable "log_retention_days" {
  type    = number
  default = 14
}

variable "app_secret_names" {
  type        = set(string)
  description = "Secrets Manager secrets injected as env vars of the same name. Created empty; values filled by hand."
}

variable "extra_env" {
  type        = map(string)
  default     = {}
  description = "Extra non-secret env vars (e.g. EMAIL_FROM, STRIPE_PACKS_JSON). Never put secrets here."
}

variable "ecr_repository_url" {
  type = string
}

variable "ecr_repository_arn" {
  type = string
}

variable "image_tag" {
  type        = string
  description = "Tag in the initial task definition. The deploy workflow replaces it on every deploy."
}

variable "github_oidc_provider_arn" {
  type = string
}

variable "github_oidc_subjects" {
  type        = list(string)
  description = "Exact OIDC sub claims allowed to assume the deploy role (no wildcards)."
}

variable "deploy_can_push_images" {
  type        = bool
  description = "Let the deploy role push new images (dev builds; prod only promotes)."
}

variable "efs_backup" {
  type        = bool
  default     = true
  description = "Daily AWS Backup of the Chroma EFS (35-day retention)."
}
