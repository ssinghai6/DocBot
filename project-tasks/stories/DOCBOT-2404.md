# DOCBOT-2404 Per-org quotas
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-24 Stage A | 1 | S7 | P1 | 3 | senior-dev | - | Not on main |

## User story
As the founder, I want per-org caps on documents, connections, concurrent sandboxes and E2B minutes, so that one tenant cannot exhaust shared capacity or run up costs for everyone.

## Why now
Limits today are per org or user for sandbox runs per hour (`enforce_sandbox_quota` in `api/utils/request_guard.py:554`) and credits per org (`api/credit_service.py`), but there is no cap on stored documents, connections or concurrent sandboxes. Firm orgs with many client workspaces arrive in S5-S6.

## Scope
In:
- Caps on documents, connections, concurrent sandboxes and E2B minutes per org, configured by tier (free, Pro, Firm, design partner).
- 429 with a reason string naming the limit.
- Defaults through env vars, documented in `.env.example`.

Out:
- RLS, Alembic, pgvector, BYOK and KMS (Stage A exclusions: rebuilt in the platform).
- UI for quota usage (show the 429 reason only).

## Acceptance criteria
- [ ] Given a free-tier org at its document cap, when it uploads one more document, then the API returns 429 with a reason naming the document limit and no file is stored.
- [ ] Given an org with N concurrent sandboxes running where N is the cap, when another sandbox call starts, then it returns 429 and the running ones are unaffected.
- [ ] Given 50 concurrent requests at cap minus 1, when they race, then exactly one succeeds (security gate 5 from yc-10-day-plan section 4).
- [ ] Given a Firm org, when its tier is set, then its caps come from the tier config without code changes.
- [ ] Given the owner key, when it is used, then quotas are bypassed as today.

## Deliverable and approach
- Extend `api/utils/request_guard.py` with a counter per (org, resource) using the existing Postgres-backed limiter from DOCBOT-1606.
- Tier lookup via the billing context (`api/utils/billing_context.py`).
- Concurrent sandboxes: count in-flight with a row per run, released in `finally`.

## Test / verification
- Unit tests for each cap and the race (asyncio.gather of 50).
- Route-guard test unchanged.

## Dependencies
Blocked by: 2401
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Starting numbers per tier? Default proposal: free 10 docs / 2 connections / 1 sandbox; Pro 200 / 10 / 2; Firm per workspace.
