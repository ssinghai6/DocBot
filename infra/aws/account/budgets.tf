# Spending guardrails for the whole account (dev + prod + shared).
#
# Two budgets because of the $5K promotional credits:
#   net   - default cost types (credits applied). Stays near $0 while credits
#           last, then becomes the real bill.
#   gross - include_credit = false, so it shows true burn rate even while
#           credits pay for it. This is the one to watch during the credit period.

variable "budget_email" {
  type        = string
  description = "Receives budget and cost-anomaly alerts."
}

variable "monthly_budget_usd" {
  type    = number
  default = 250
}

variable "env_budgets_usd" {
  type        = map(number)
  default     = { dev = 60, prod = 150 }
  description = "Per-env gross monthly budgets, filtered on the Environment tag (activate it as a cost allocation tag in Billing first)."
}

variable "anomaly_threshold_usd" {
  type        = number
  default     = 20
  description = "Alert when an anomaly's total impact is at least this many USD."
}

variable "anomaly_monitor_arn" {
  type        = string
  default     = ""
  description = "Existing AWS-services anomaly monitor ARN (accounts allow only one; check `aws ce get-anomaly-monitors`). Empty = create one."
}

locals {
  budgets = {
    net   = true  # include_credit
    gross = false # credits excluded
  }
  # [threshold %, ACTUAL|FORECASTED]
  budget_alerts = [[50, "ACTUAL"], [80, "ACTUAL"], [100, "ACTUAL"], [100, "FORECASTED"]]
}

resource "aws_budgets_budget" "monthly" {
  for_each     = local.budgets
  name         = "docbot-account-monthly-${each.key}"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = each.value
  }

  dynamic "notification" {
    for_each = local.budget_alerts
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value[0]
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.value[1]
      subscriber_email_addresses = [var.budget_email]
    }
  }
}

# Gross (credits excluded) per env, so dev/prod burn shows even while credits pay.
resource "aws_budgets_budget" "env" {
  for_each     = var.env_budgets_usd
  name         = "docbot-${each.key}-monthly-gross"
  budget_type  = "COST"
  limit_amount = tostring(each.value)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = false
  }

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Environment$%s", each.key)] # TagKeyValue syntax: user:<key>$<value>
  }

  dynamic "notification" {
    for_each = local.budget_alerts
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value[0]
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.value[1]
      subscriber_email_addresses = [var.budget_email]
    }
  }
}

resource "aws_ce_anomaly_monitor" "services" {
  count             = var.anomaly_monitor_arn == "" ? 1 : 0
  name              = "docbot-services"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"
}

resource "aws_ce_anomaly_subscription" "daily" {
  name             = "docbot-anomalies"
  frequency        = "DAILY" # email subscribers require DAILY or WEEKLY
  monitor_arn_list = [coalesce(var.anomaly_monitor_arn, one(aws_ce_anomaly_monitor.services[*].arn))]

  subscriber {
    type    = "EMAIL"
    address = var.budget_email
  }

  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      match_options = ["GREATER_THAN_OR_EQUAL"]
      values        = [tostring(var.anomaly_threshold_usd)]
    }
  }
}
