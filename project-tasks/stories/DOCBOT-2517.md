# DOCBOT-2517 DNS cutover plan, stable api hostname
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S2 | P1 | 3 | founder | - | Open (Wave B; start early) |

## User story
As the founder, I want the frontend and every third-party callback to use a stable `api.docbot.space` hostname well before cutover, so that moving prod to AWS is one CNAME change instead of re-registering Stripe, OAuth and SAML URLs on the day.

## Why now
The hostname must point at Railway and carry real traffic for at least 7 days before 2519, and the TTL must drop to 60 s at least 48 hours before. The plan is merged to dev (PR #117), but `infra/aws/docs/dns-cutover.md` was lost to the `.gitignore` bug.

## Scope
In:
- Re-author `infra/aws/docs/dns-cutover.md` (devops writes; founder executes).
- Railway custom domain `api.docbot.space`, Vercel Production `RAILWAY_BACKEND_URL` switched to it, callbacks moved (Stripe webhook, GitHub and Google OAuth, SAML ACS, `OAUTH_REDIRECT_BASE_URL`).
- TTL 60 s at least 48 hours before cutover; post-flip resolver check.

Out:
- The flip itself (2519), rollback (2518).

## Acceptance criteria
- [ ] Given the Railway custom domain is live, when `curl -s https://api.docbot.space/api/health` is run, then it returns ok with a valid certificate, at least 7 days before the cutover date.
- [ ] Given Vercel Production, when the founder signs up, logs in, chats and triggers a Stripe test event, then all work through `api.docbot.space`.
- [ ] Given the callback list from checklist item 8, when each provider console is checked, then every callback host is `api.docbot.space`.
- [ ] Given T-48h, when `dig api.docbot.space` is run, then the TTL is 60 or less.
- [ ] Given the post-flip check, when 3 public resolvers (for example 1.1.1.1, 8.8.8.8, 9.9.9.9) are queried, then all return the ALB within 10 minutes of the flip.

## Deliverable and approach
Founder executes checklist items 18 and 26 using the doc. Devops writes the doc and the resolver-check commands. No DNS change is made by an agent.

## Test / verification
Health curl output and dig outputs pasted with dates; founder confirms the click-through.

## Dependencies
Blocked by: none (plan doc pairs with 2514).
Blocks: 2518, 2519.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- If `docbot.space` DNS is on Cloudflare, `api` must be DNS-only (grey cloud) or SSE and ACM validation break; confirm the DNS host (checklist item 6).
