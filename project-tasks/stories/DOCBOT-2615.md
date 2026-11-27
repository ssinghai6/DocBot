# DOCBOT-2615 NEW: Opt-in SSE streaming on /v1
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P2 | 3 | senior-dev | - | New (2026-11-23) |

## User story
As an API client building a chat UI, I want to stream tokens with `Accept: text/event-stream`, so that users see answers as they are written.

## Why now
`/v1` is buffered only. Streaming must emit the corrected answer, which needs the DOCBOT-2592 fix first.

## Scope
In:
- Streaming on chat, db query, hybrid and autopilot when `Accept: text/event-stream`.
- Named events `metadata`, `token`, `replace`, `step`, `done`, `error`; 15s pings.
- Credits settle on client disconnect; concurrency limit enforced.

Out:
- WebSockets.

## Acceptance criteria
- [ ] Given `Accept: text/event-stream`, when a hybrid question is asked, then named events arrive and exactly one terminal `done` or `error` event ends the stream.
- [ ] Given a correction during the answer, when the stream runs, then a `replace` event carries the corrected text.
- [ ] Given the client disconnects mid-stream, when the server notices, then credits settle for what ran.
- [ ] Given the per-key concurrency limit is reached, when another stream starts, then it gets 429 `concurrency_limited`.

## Deliverable and approach
Reuse the existing SSE generators; translate to named events in one adapter. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests with a fake stream.

## Dependencies
Blocked by: 3901, 2592.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- none
