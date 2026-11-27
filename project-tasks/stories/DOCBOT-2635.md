# DOCBOT-2635 BUG: Hybrid answer replaced by a generic error when the stream breaks off
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 1 | senior-dev | 2026-11-25 | Done (2026-11-24, PR #163) |

## User story
As a fractional CFO running a hybrid tie-out, I want a broken answer stream to keep what already arrived and say why it stopped, so that I never see an unexplained "Please try again" over results that were delivered.

## Why now
Prod 2026-11-24 07:05 UTC, Tidewell demo, "Does segment revenue in Note 12 agree with the books?": the backend sent the full 66,566-byte stream (Railway HTTP log, 200, 25s), the SQL and analysis code rendered, then the browser showed "I encountered an error with the hybrid query. Please try again." That text is only reached when the body read rejects (every per-event error is already skipped), and the real error was never logged. Investor demo is 2026-12-01.

## Scope
In:
- `src/hooks/useChatSubmit.ts` hybrid path: `console.error` the real error; keep streamed text, SQL, code and charts; name the cause in the bubble; `console.warn` any skipped SSE event.
- `src/lib/chatStream.ts`: `keepPartialAnswer`, `streamDroppedNote`.

Out:
- Recovering the finished answer from the backend after a drop (needs run_id early + a read route).
- Docs and DB chat paths.

## Acceptance criteria
- [ ] Given SQL and analysis code arrived and the body read rejects, when the hybrid stream fails, then SQL and code stay and the bubble names the error (e.g. "TypeError: network error").
- [ ] Given answer text arrived and the stream breaks, then the text is kept and the note is appended.
- [ ] Given any hybrid failure, then the real error is logged with `console.error`.

## Deliverable and approach
Two helpers in `src/lib/chatStream.ts`, used by the hybrid catch in `src/hooks/useChatSubmit.ts`. Tests in `tests/unit/test_hybrid_stream_drop_2635.py` (node, captured prod state).

## Test / verification
`pytest tests/unit/test_hybrid_stream_drop_2635.py`; `npm run lint`; `npx tsc --noEmit`. Repro: 8 prod runs (direct, via Vercel HTTP/1.1 and HTTP/2, and in Chrome) all completed; the drop itself did not recur.

## Dependencies
Blocked by: none.
Blocks: investor demo 2026-12-01.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- What cut the browser's stream on 2026-11-24 (network blip, sleep, edge reset)? The new console.error and bubble text will name it next time.
