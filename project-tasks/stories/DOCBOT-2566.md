# DOCBOT-2566 Multi-org membership
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S5 | - | 5 | senior-dev | - | To do |

## User story
As a contractor CFO, I want to belong to my own personal org and to a firm org at the same time and switch between them, so that I keep my own clients separate from the firm's.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. `users.org_id` is a single column (`api/index.py` users table) and `users.email` is unique, so accepting an invite (2565) has to move the user out of their personal org.

## Scope
In:
- `org_members` (org_id, user_id, role, created_at, removed_at) as source of truth; backfilled from `users.org_id`.
- Active org chosen per session; `CallerScope.org_id` and role come from the membership row.
- Org switcher in the sidebar next to the workspace switcher.

Out:
- Per-org SAML (2572).

## Acceptance criteria
- [ ] Given existing users, when the app boots, then each has exactly one membership row matching `users.org_id` and the backfill is idempotent.
- [ ] Given a user with two memberships, when they switch org, then `CallerScope.org_id` and role change and data lists show only that org.
- [ ] Given a request naming an org the user is not a member of, when it is processed, then it returns 404.
- [ ] Given role changes, when an admin edits a member, then the last-admin protection (DOCBOT-1620) still holds per org.
- [ ] Given the 2401 sweep, when run with a two-org member as caller, then no cross-org access appears.

## Deliverable and approach
- Keep `users.org_id` as the default org for compatibility; read membership in the CallerScope dependency.
- Update `get_user_org` in `api/org_service.py` callers.

## Test / verification
- Integration tests for backfill and switching; sweep extension.

## Dependencies
Blocked by: 2565
Blocks: 2567

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Does `require_role` read the role per active org everywhere (rbac_service.py), including `/v1` keys?
- Points: moving the role source to `org_members` touches `require_role`, `/v1` API keys and the CallerScope dependency; 5 may be low. Proposed to the founder for re-sizing, not changed here.
