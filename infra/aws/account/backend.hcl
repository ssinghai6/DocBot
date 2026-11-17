# Same bucket/table as envs; fill from `terraform output` in infra/aws/bootstrap.
#   terraform init -backend-config=backend.hcl
bucket         = "REPLACE_ME_STATE_BUCKET"
key            = "docbot/account/terraform.tfstate"
region         = "REPLACE_ME_REGION"
dynamodb_table = "docbot-tfstate-lock"
encrypt        = true
