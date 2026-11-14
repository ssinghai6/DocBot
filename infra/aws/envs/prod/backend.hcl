# Fill in from `terraform output` in infra/aws/bootstrap, then:
#   terraform init -backend-config=backend.hcl
bucket         = "REPLACE_ME_STATE_BUCKET"
key            = "docbot/prod/terraform.tfstate"
region         = "REPLACE_ME_REGION"
dynamodb_table = "docbot-tfstate-lock"
encrypt        = true
