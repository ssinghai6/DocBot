variable "aws_region" {
  type        = string
  description = "AWS region for the dev environment."
}

variable "api_domain" {
  type    = string
  default = "api-dev.docbot.space"
}

variable "frontend_url" {
  type        = string
  default     = "https://dev.docbot.space"
  description = "Set to the Vercel Preview URL used for branch dev."
}

variable "allowed_origins" {
  type    = string
  default = "https://dev.docbot.space"
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Scale-to-zero toggle: 0 stops Fargate compute (ALB, RDS, EFS still bill)."
}

variable "use_fargate_spot" {
  type    = bool
  default = true
}

variable "deletion_protection" {
  type        = bool
  default     = true
  description = "RDS + ALB deletion protection. Set false (and apply) before terraform destroy."
}

variable "app_secret_names" {
  type    = set(string)
  default = ["groq_api_key", "GEMINI_API_KEY", "E2B_API_KEY", "DB_ENCRYPTION_KEY", "OWNER_KEY"]
}

variable "extra_env" {
  type    = map(string)
  default = {}
}

variable "github_repo" {
  type    = string
  default = "ssinghai6/DocBot"
}

variable "alarm_email" {
  type        = string
  default     = ""
  description = "CloudWatch alarm email (SNS). Empty = no alarms."
}
