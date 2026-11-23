# DOCBOT-3905 Developer portal /apis with Scalar, test key and sandbox quickstart
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API, SDK and MCP Server | 1 | S1 | P1 | 5 | frontend | - | New (scheduled from parking lot) |

## User story
As the founder, I want a token-gated `docbot.space/apis` page to read the docs and try the API, so that I and early developers can play with the API and show it to others.

## Why now
This is the founder's ask. It needs the filtered spec (3901), test keys (2598) and sandbox data (2597).

## Scope
In:
- `src/app/apis/page.tsx` renders Scalar from `/v1/openapi.json`.
- Signed-out visitors see read-only docs + a sign-in call to action.
- "Get test key": key held in React memory only (no URL, no storage), injected as bearer auth.
- Quickstart: sandbox session, hybrid question, lineage.

Out:
- Live key management UI.

## Acceptance criteria
- [ ] Given a signed-out visitor, when they open `/apis`, then they see read-only docs and a sign-in call to action.
- [ ] Given a signed-in user, when they click "Get test key" and use Try it, then requests go same-origin through the Vercel `/v1` rewrite with the key as bearer auth.
- [ ] Given the page after a reload, when local storage and the URL are inspected, then the key is in neither.
- [ ] Given a new user, when they follow the quickstart, then they get a first hybrid answer in under 10 minutes.
- [ ] Given the nonce CSP from `src/proxy.ts`, when the page loads in light and dark mode, then Scalar renders with no CSP errors and passes WCAG 2.1 AA basics.

## Deliverable and approach
Scalar via its React package; no custom docs renderer. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Manual regression list (Playwright smoke once e2e exists): sign in, get key, run quickstart.

## Dependencies
Blocked by: 3901, 2597, 2598.
Blocks: 3906, 3907.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Scalar theme to match the 3-color palette?
