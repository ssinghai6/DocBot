# DOCBOT-2506 Vercel Preview env for dev
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 0 | - | P1 | 1 | founder | 11-24 | Open (Phase 0, Wave A) |

## User story
As the founder, I want Vercel Preview builds of branch `dev` to talk to `api-dev.docbot.space`, so that I can click through signup, login and chat on AWS end to end while production keeps using Railway.

## Why now
After the 2505 smoke, the only missing piece of a full AWS test env is the frontend. One Vercel variable scoped to Preview and branch `dev` does it with no prod risk.

## Scope
In:
- Vercel env var `RAILWAY_BACKEND_URL=https://api-dev.docbot.space`, environment Preview, branch `dev` only.
- Click-through check on the preview URL.

Out:
- Any Production env change (that happens once in 2517, step 18).

## Acceptance criteria
- [ ] Given the Vercel dashboard, when the founder lists env vars for Production before and after, then the Production list is unchanged.
- [ ] Given a Preview deploy of `dev`, when the founder signs up, logs in and asks a PDF question, then all three work and the browser network tab shows requests proxied to api-dev.
- [ ] Given the api-dev `ALLOWED_ORIGINS`/`frontend_url`, when the preview URL calls the API, then no CORS error appears.

## Deliverable and approach
Founder-only (checklist item 17). If CORS fails, devops adds the preview URL to `allowed_origins` in `envs/dev/terraform.tfvars` and the founder re-applies.

## Test / verification
Founder confirms the three click-throughs in the tracker row; no values shared.

## Dependencies
Blocked by: 2505.
Blocks: none (nice-to-have for 2513 manual checks).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Preview URLs change per deploy; is a stable branch alias (for example the `dev` branch domain) needed in `allowed_origins`?
