variable "aws_region" {
  type = string
}

variable "api_domain" {
  type        = string
  default     = "api.docbot.space"
  description = "Stable API hostname; DNS cutover flips its CNAME from Railway to the ALB (docs/dns-cutover.md)."
}

variable "frontend_url" {
  type    = string
  default = "https://www.docbot.space"
}

variable "allowed_origins" {
  type    = string
  default = "https://www.docbot.space,https://docbot.space"
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Keep 1: Chroma on EFS (SQLite) supports a single writer."
}

variable "task_cpu" {
  type    = number
  default = 1024
}

variable "task_memory" {
  type    = number
  default = 2048
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.small"
}

variable "db_multi_az" {
  type    = bool
  default = false
}

# Mirror every secret Railway prod has set (`railway variables`). Any name listed
# here but left empty keeps tasks from starting (fails closed). Drop the ones prod
# does not use before the first apply.
variable "app_secret_names" {
  type = set(string)
  default = [
    "groq_api_key",
    "GEMINI_API_KEY",
    "E2B_API_KEY",
    "DB_ENCRYPTION_KEY", # the EXISTING Railway prod key, unchanged (DOCBOT-2515)
    "OWNER_KEY",
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "RESEND_API_KEY",
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "SENTRY_DSN",
    "TURNSTILE_SECRET_KEY",
  ]
}

variable "extra_env" {
  type        = map(string)
  default     = {}
  description = "Non-secret Railway vars to carry over (EMAIL_FROM, STRIPE_PACKS_JSON, PLATFORM_ADMIN_USER_IDS, ...)."
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
