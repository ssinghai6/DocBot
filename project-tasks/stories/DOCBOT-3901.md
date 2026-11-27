# DOCBOT-3901 Public API v1 contract: v1_routes module, filtered OpenAPI, error envelope, versioning, per-key rate limits
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API, SDK and MCP Server | 1 | S1 | P1 | 5 | senior-dev | - | New (scheduled from parking lot) |

## User story
As an API client, I want a stable, documented `/v1` contract, so that I can build against DocBot without reading its source.

## Why now
Every playground story builds on this: the portal renders the filtered spec, and rate limits must not sleep the server.

## Scope
In:
- `/v1` handlers moved from `api/index.py` to `api/routes/v1_routes.py`; models in `api/v1_models.py` + `src/lib/schemas/v1.ts`; `response_model` on every route.
- `/v1` error envelope + `X-Request-Id`.
- Per-key limiter on the shared counter store: 429 + `Retry-After` + `X-RateLimit-*` (replaces the sleeping in-memory bucket in `api/api_key_service.py`).
- `GET /v1/openapi.json` (filtered) + snapshot test; `GET /v1/me`; versioning and deprecation policy doc; keys in the query string rejected (400 `key_in_url`).

Out:
- Async jobs (3902), SDKs (3906, 3907).

## Acceptance criteria
- [ ] Given any `/v1` error, when it is returned, then it uses the envelope with a `type`, a `code` and the `X-Request-Id`.
- [ ] Given a key over its rate limit, when it calls again, then it gets 429 with `Retry-After` and `X-RateLimit-*` and the server does not sleep.
- [ ] Given `GET /v1/openapi.json`, when it is fetched, then it lists only `/v1` routes and matches the committed snapshot.
- [ ] Given a key passed as a query parameter, when the request arrives, then it gets 400 `key_in_url`.
- [ ] Given the route guard test, when it runs, then every `/v1` route has `require_api_key`.

## Deliverable and approach
Move, do not rewrite: same handlers, new module. The limiter reuses the existing counter store. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
`test_v1_error_envelope.py`, `test_v1_rate_limit.py`, `test_v1_openapi_snapshot.py`; `test_route_guards.py` stays green.

## Dependencies
Blocked by: 1530, 2591, 2596.
Blocks: 2907, 2597, 2599, 3905, 3902.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Rate limit defaults per key mode?
