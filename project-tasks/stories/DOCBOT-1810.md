# DOCBOT-1810 Pro subscription + dunning
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-18 billing (referenced) | 1 | S3 | P2 | 3 | senior-dev | - | PR #103 open (held for founder review) |

## User story
As a fractional CFO on the Pro plan, I want my monthly subscription to renew, retry failed payments and downgrade cleanly if payment finally fails, so that I am billed correctly without anyone at DocBot doing it by hand.

## Why now
Paying logos are the Phase 1 gate ($5K MRR + 6 logos by 02-28). PR #103 (`api/subscription_service.py`, `api/routes/billing_routes.py`, `api/billing_service.py`, `api/billing_models.py`, tests) is rebased on main, passes locally and is held for founder review. Firm billing per workspace (2569) reuses its webhook and lifecycle.

## Scope
In:
- Merge PR #103 after founder review.
- Follow-up 1: price-id filter, `parse_event` only treats a subscription or invoice line as Pro when its price is `STRIPE_PRO_PRICE_ID`.
- Follow-up 2: stale `customer.subscription.deleted` overwrite, ignore events whose `subscription_id` differs from the stored row while that row is active or `past_due`.
- Founder steps by name only: env `STRIPE_PRO_PRICE_ID` (recurring $79/mo, same Stripe mode as `STRIPE_SECRET_KEY`), confirm `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` belong to the same account and mode, `FRONTEND_URL`; webhook subscribes to `customer.subscription.created/updated/deleted`, `invoice.paid`, `invoice.payment_failed`, `checkout.session.completed`, `checkout.session.async_payment_succeeded`; Revenue recovery Smart Retries on with "cancel the subscription" when retries fail.

Out:
- Customer portal and Pro UI (deferred in the PR body).
- Metered overage (1809).
- Credit rollover policy change.

## Acceptance criteria
- [ ] Given a Checkout for the Pro price, when `customer.subscription.created` and `invoice.paid` arrive, then the org's `org_subscriptions` row is active and Pro credits are granted once.
- [ ] Given a subscription on a different recurring price, when its events arrive, then no Pro state or credits change (price-id filter).
- [ ] Given an org that cancelled and re-subscribed, when a late `customer.subscription.deleted` for the old subscription arrives with a newer `created`, then the active row for the new subscription is unchanged.
- [ ] Given `invoice.payment_failed` followed by Stripe cancelling after Smart Retries, when `customer.subscription.deleted` arrives for the current subscription, then the org drops to free.
- [ ] Given a replayed webhook `event.id` or a bad signature, when received, then the first is a no-op and the second returns 400.
- [ ] Given the founder checklist, when it is completed, then each env var and dashboard setting above is confirmed by name in the PR, with no values pasted.

## Deliverable and approach
- Founder reviews and merges PR #103; follow-ups 1 and 2 land in a small follow-up PR (or as commits on #103 if the founder prefers) in `api/subscription_service.py`.

## Test / verification
- Existing `tests/integration/test_stripe_subscription.py` and unit tests in the PR; add cases for the price filter and the stale delete.
- Stripe test-mode run of create, renew, fail, cancel recorded in the PR.

## Dependencies
Blocked by: founder review of PR #103 and the Stripe config.
Blocks: 2569 (Firm billing per workspace), 1614 (billing surface in review).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Which webhook API version does the endpoint use (both invoice shapes are parsed)?
- Do the two follow-ups ship before merge or as a fast follow?
- Points: 3 covers merging PR #103 only; the price-id filter and stale `subscription.deleted` fix add about 2 (proposed 3 to 5, founder to approve).
