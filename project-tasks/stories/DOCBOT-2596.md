# DOCBOT-2596 NEW: Close default FastAPI /docs, /redoc, /openapi.json in prod
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S1 | P1 | 1 | senior-dev | - | New (2026-11-23) |

## User story
As the founder, I want the default FastAPI docs and full OpenAPI spec closed on the production backend, so that internal `/api` routes are not published to anyone who finds the backend host.

## Why now
Today `/docs`, `/redoc` and `/openapi.json` are open on the backend host (`api/index.py:50`) and list every internal route. The public spec must be the filtered `/v1/openapi.json` from DOCBOT-3901.

## Scope
In:
- `docs_url`, `redoc_url`, `openapi_url` are `None` unless `ENV=dev`.
- The CSP exemption for the docs page in `api/utils/security_headers.py` is removed or dev-only.
- Unit test for both modes.

Out:
- The filtered `/v1/openapi.json` (DOCBOT-3901).

## Acceptance criteria
- [ ] Given `ENV` is not `dev`, when `/docs`, `/redoc` or `/openapi.json` is requested, then the backend returns 404.
- [ ] Given `ENV=dev`, when `/docs` is requested, then it returns 200 as today.
- [ ] Given the prod security headers, when the docs exemption is checked, then it applies only in dev.
- [ ] Given the route guard test `tests/unit/test_route_guards.py`, when it runs, then it stays green.

## Deliverable and approach
Read `ENV` once at app construction in `api/index.py` and pass the three URLs; gate the CSP exemption on the same flag. No new module. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
`tests/unit/test_docs_exposure.py` (404 when `ENV!=dev`, 200 in dev).

## Dependencies
Blocked by: none.
Blocks: 3901.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Does the Vercel `/v1` rewrite need any change once `/openapi.json` is gone? (Expected no.)
