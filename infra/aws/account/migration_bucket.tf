# Private bucket for cutover artifacts: the Chroma tarball handed to the ECS
# import task (data_copy.sh, via a presigned URL) and the Railway sunset archive
# (final pg_dump + /data tarball, DOCBOT-2520). import/ expires after 90 days;
# archive/ is kept.
# SSE-S3 today; moves to a KMS CMK in DOCBOT-2523.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "migration" {
  bucket = "docbot-migration-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "migration" {
  bucket                  = aws_s3_bucket.migration.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "migration" {
  bucket = aws_s3_bucket.migration.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "migration" {
  bucket = aws_s3_bucket.migration.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "migration" {
  bucket = aws_s3_bucket.migration.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "migration" {
  bucket = aws_s3_bucket.migration.id
  # Only import/ expires; archive/ (Railway sunset pg_dump) and anything else are kept.
  rule {
    id     = "expire-import-90d"
    status = "Enabled"
    filter {
      prefix = "import/"
    }
    expiration {
      days = 90
    }
  }
  rule {
    id     = "noncurrent-and-multipart"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_policy" "migration_tls_only" {
  bucket = aws_s3_bucket.migration.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.migration.arn, "${aws_s3_bucket.migration.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.migration]
}

output "migration_bucket" {
  description = "S3_URI prefix for data_copy.sh import and the 2520 archive."
  value       = "s3://${aws_s3_bucket.migration.bucket}"
}
