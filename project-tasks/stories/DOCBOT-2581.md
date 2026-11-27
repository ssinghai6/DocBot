# DOCBOT-2581 NEW: Waitlist instead of payments
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P1 | 3 | senior-dev | 11-25 | In progress (2026-11-17) |

## User story
As the founder, I want people who hit the out-of-credits wall or visit the landing page to join a waitlist, so that I capture interest and reply by hand while I have no company, tax or bank setup for payments.

## Why now
Credit packs are all `purchasable:false` on prod (Stripe not configured), so the out-of-credits modal shows "Coming soon" and a dead end.

## Scope
In:
- `waitlist` table, `POST /api/waitlist` (public, honeypot, validation), `GET /api/admin/waitlist` (platform admin).
- Optional founder email via Resend when `RESEND_API_KEY`, `EMAIL_FROM`, `WAITLIST_NOTIFY_EMAIL` are all set.
- `WaitlistForm` in OutOfCreditsModal (only when no pack is purchasable) and on the landing page.

Out:
- Payments, Stripe config, invite flow, funnel event for waitlist joins.

## Acceptance criteria
- [ ] Given no purchasable pack, when the out-of-credits modal opens, then the waitlist form shows instead of "Coming soon" and the payments line.
- [ ] Given a purchasable pack, when the modal opens, then the Buy path works unchanged.
- [ ] Given a valid submit, when the same email submits again, then one row exists with the newer note and updated_at.
- [ ] Given the honeypot field is filled, when submitted, then the API answers 202 and stores nothing.
- [ ] Given an invalid email or an over-length field, when submitted, then the API answers 422.
- [ ] Given a non-admin or anonymous caller, when they GET the admin list, then they get 401 or 403.
- [ ] Given the notify env vars are unset, when someone joins, then no email is attempted and the 202 is unaffected.

## Deliverable and approach
New router `api/routes/waitlist_routes.py`, models in `api/waitlist_models.py`, service in `api/waitlist_service.py`, Zod mirror `src/lib/schemas/waitlist.ts`. Reuses the global per-IP POST limiter in abuse_guard.

## Test / verification
`tests/unit/test_waitlist.py`, route guard and admin guard sweeps, `npx tsc --noEmit`, `npm run lint`, `npm run build`.

## Dependencies
Blocked by: none.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Which inbox should WAITLIST_NOTIFY_EMAIL point to?
