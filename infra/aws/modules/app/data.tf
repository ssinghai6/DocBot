# ---------------------------------------------------------------- RDS Postgres
resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.public[*].id
}

resource "aws_db_instance" "main" {
  identifier     = local.name
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "docbot"
  username = "docbot"
  # RDS creates and owns the password in a Secrets Manager secret (JSON with
  # username/password). The task reads both keys; see local.start_command.
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = var.db_multi_az

  backup_retention_period    = var.db_backup_retention_days
  copy_tags_to_snapshot      = true
  auto_minor_version_upgrade = true
  deletion_protection        = var.deletion_protection
  skip_final_snapshot        = var.db_skip_final_snapshot
  final_snapshot_identifier  = var.db_skip_final_snapshot ? null : "${local.name}-final"
}

# ------------------------------------------------- EFS for Chroma (CHROMA_DIR)
resource "aws_efs_file_system" "chroma" {
  encrypted = true
  tags      = { Name = "${local.name}-chroma" }
}

resource "aws_efs_mount_target" "chroma" {
  count           = 2
  file_system_id  = aws_efs_file_system.chroma.id
  subnet_id       = aws_subnet.public[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# The image runs as appuser (uid/gid 1000); the access point forces that
# identity and creates /docbot owned by it, so CHROMA_DIR is writable.
resource "aws_efs_access_point" "chroma" {
  file_system_id = aws_efs_file_system.chroma.id
  posix_user {
    uid = 1000
    gid = 1000
  }
  root_directory {
    path = "/docbot"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "750"
    }
  }
}
