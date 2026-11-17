# CloudWatch alarms -> SNS email (DOCBOT-2511). Disabled when alarm_email is "".
# ~$0.10 per alarm metric per month.

locals {
  alarms_on = var.alarm_email != ""
  alb_dims  = { LoadBalancer = aws_lb.api.arn_suffix, TargetGroup = aws_lb_target_group.api.arn_suffix }
  # name => [namespace, metric, statistic, comparison, threshold, period, eval periods, dimensions, missing data]
  simple_alarms = {
    unhealthy-targets = ["AWS/ApplicationELB", "UnHealthyHostCount", "Maximum", "GreaterThanThreshold", 0, 60, 3, local.alb_dims, "notBreaching"]
    # Healthy targets below desired == "ECS running < desired" without paying for Container Insights.
    running-below-desired = ["AWS/ApplicationELB", "HealthyHostCount", "Minimum", "LessThanThreshold", var.desired_count, 60, 5, local.alb_dims, "breaching"]
    rds-cpu-high          = ["AWS/RDS", "CPUUtilization", "Average", "GreaterThanThreshold", 80, 300, 3, { DBInstanceIdentifier = aws_db_instance.main.identifier }, "missing"]
    rds-free-storage-low  = ["AWS/RDS", "FreeStorageSpace", "Minimum", "LessThanThreshold", 2 * 1024 * 1024 * 1024, 300, 1, { DBInstanceIdentifier = aws_db_instance.main.identifier }, "missing"]
    rds-connections-high  = ["AWS/RDS", "DatabaseConnections", "Maximum", "GreaterThanThreshold", var.alarm_db_connections, 300, 2, { DBInstanceIdentifier = aws_db_instance.main.identifier }, "missing"]
    # Bursting EFS: running out of credits throttles Chroma reads to baseline.
    efs-burst-credits-low = ["AWS/EFS", "BurstCreditBalance", "Minimum", "LessThanThreshold", 500 * 1024 * 1024 * 1024, 300, 2, { FileSystemId = aws_efs_file_system.chroma.id }, "missing"]
  }
}

resource "aws_sns_topic" "alarms" {
  count = local.alarms_on ? 1 : 0
  name  = "${local.name}-alarms"
}

resource "aws_sns_topic_subscription" "alarms_email" {
  count     = local.alarms_on ? 1 : 0
  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email" # AWS emails a confirmation link; alarms are silent until it is clicked
  endpoint  = var.alarm_email
}

resource "aws_cloudwatch_metric_alarm" "simple" {
  # running-below-desired would fire forever while scaled to zero on purpose.
  for_each = { for k, v in local.simple_alarms : k => v if local.alarms_on && !(k == "running-below-desired" && var.desired_count == 0) }

  alarm_name          = "${local.name}-${each.key}"
  namespace           = each.value[0]
  metric_name         = each.value[1]
  statistic           = each.value[2]
  comparison_operator = each.value[3]
  threshold           = each.value[4]
  period              = each.value[5]
  evaluation_periods  = each.value[6]
  dimensions          = each.value[7]
  treat_missing_data  = each.value[8]
  alarm_actions       = [aws_sns_topic.alarms[0].arn]
  ok_actions          = [aws_sns_topic.alarms[0].arn]
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx_rate" {
  count               = local.alarms_on ? 1 : 0
  alarm_name          = "${local.name}-alb-5xx-rate"
  alarm_description   = "Target + ELB 5xx > ${var.alarm_5xx_percent}% of requests (ignored under 20 req / 5 min)"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_5xx_percent
  evaluation_periods  = 2
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]
  ok_actions          = [aws_sns_topic.alarms[0].arn]

  metric_query {
    id          = "rate"
    expression  = "IF(req >= 20, 100 * (FILL(t5xx, 0) + FILL(e5xx, 0)) / req, 0)"
    label       = "5xx percent"
    return_data = true
  }
  dynamic "metric_query" {
    for_each = { t5xx = "HTTPCode_Target_5XX_Count", e5xx = "HTTPCode_ELB_5XX_Count", req = "RequestCount" }
    content {
      id = metric_query.key
      metric {
        namespace   = "AWS/ApplicationELB"
        metric_name = metric_query.value
        stat        = "Sum"
        period      = 300
        dimensions  = { LoadBalancer = aws_lb.api.arn_suffix }
      }
    }
  }
}
