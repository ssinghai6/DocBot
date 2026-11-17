# DOCBOT-2508 Shared module + envs/prod
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 8 | devops | merge by 11-30, **apply after Dec 1** | PR #106 open (merge by 11-30, apply after) |

## User story
As the founder moving prod off Railway, I want dev and prod built from one shared Terraform module with prod-only safety settings, so that prod is the same stack dev already proved, with backups, deletion protection and encryption fixed from day one.

## Why now
Prod RDS and EFS must be created in private DB subnets and on one customer-managed KMS key from the start, because neither can be changed in place later. PR #106 is merged to dev; the prod apply waits until after Dec 1 (2576).

## Scope
In:
- `infra/aws/modules/app` (alb, ecs, data, network, alarms, github, rotation) used by `envs/dev` and `envs/prod`.
- `infra/aws/envs/dev/moved.tf` so the refactor is a no-op for dev.
- Prod: own state key, `use_fargate_spot=false`, RDS `backup_retention_period` at least 7 (README says 14), `deletion_protection=true`, `skip_final_snapshot=false`, RDS and EFS in private DB subnets, one CMK with rotation, EFS backup policy, OIDC role trusted only for `environment:aws-prod`, `desired_count <= 1`.

Out:
- The prod apply (2576), private task subnets (2521), remaining CMKs (2523).

## Acceptance criteria
- [ ] Given dev was applied from the flat layout, when `terraform plan` runs in `infra/aws/envs/dev` after the refactor, then it shows 0 to add, 0 to change, 0 to destroy.
- [ ] Given `infra/aws/envs/prod`, when `terraform validate` and `terraform plan` (founder, after Dec 1) run, then RDS shows `deletion_protection = true`, `skip_final_snapshot = false`, backup retention at least 7, and a KMS key ARN on both RDS and EFS.
- [ ] Given the prod OIDC role, when its trust policy is read, then the only allowed `sub` is `repo:<repo>:environment:aws-prod`.
- [ ] Given `desired_count = 2` in prod tfvars, when `terraform validate` runs, then the variable validation fails (Chroma on EFS is single-writer).
- [ ] Given the prod RDS and EFS mount targets, when their subnets are inspected in the plan, then they are the private DB subnets with no route to an internet gateway.

## Deliverable and approach
Shipped as code on `origin/dev` (PR #106). Remaining: founder dev re-plan proof, and prod plan review in 2576. Any fix found in plan goes in a follow-up PR by devops; agents never apply.

## Test / verification
CI `terraform fmt -check` / `validate` on both envs; founder pastes plan summary lines only.

## Dependencies
Blocked by: 2501.
Blocks: 2509, 2511, 2512, 2514, 2576.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- RDS single-AZ `db.t4g.small` at launch (checklist item 10); revisit Multi-AZ (about $25/mo more) after cutover?
