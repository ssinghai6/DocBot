variable "aws_region" {
  type        = string
  description = "AWS region for the dev environment."
}

variable "api_domain" {
  type        = string
  default     = "api-dev.docbot.space"
  description = "Public hostname of the dev API. CNAME it to the ALB at your DNS host."
}

variable "frontend_url" {
  type        = string
  default     = "https://dev.docbot.space"
  description = "Dev frontend URL (FRONTEND_URL). Set to the Vercel Preview URL used for branch dev."
}

variable "allowed_origins" {
  type        = string
  default     = "https://dev.docbot.space"
  description = "ALLOWED_ORIGINS for CORS (comma-separated)."
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Number of API tasks. Set 0 to stop compute while idle."
}

variable "use_fargate_spot" {
  type        = bool
  default     = true
  description = "Run the task on FARGATE_SPOT (~70% cheaper, can be interrupted with 2 min notice; ECS restarts it)."
}

variable "deletion_protection" {
  type        = bool
  default     = true
  description = "RDS deletion protection. Set false (and apply) before terraform destroy."
}

variable "app_secret_names" {
  type        = set(string)
  description = "Secrets Manager secrets injected into the container as env vars of the same name. Values are filled by hand, never by Terraform."
  default = [
    "groq_api_key",
    "GEMINI_API_KEY",
    "E2B_API_KEY",
    "DB_ENCRYPTION_KEY",
    "OWNER_KEY",
  ]
}

variable "github_repo" {
  type    = string
  default = "ssinghai6/DocBot"
}

variable "github_deploy_branch" {
  type    = string
  default = "dev"
}

variable "create_github_oidc_provider" {
  type        = bool
  default     = true
  description = "Create the GitHub Actions OIDC provider. Set false if the account already has one (it is account-wide, only one allowed)."
}

variable "monthly_budget_usd" {
  type    = number
  default = 60
}

variable "budget_email" {
  type        = string
  description = "Email that receives AWS Budgets alerts (80% actual, 100% forecast)."
}
