# DOCBOT-2569 Firm billing per client workspace
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S6 | - | 5 | senior-dev | - | To do |

## User story
As a firm owner, I want my subscription quantity to follow the number of active client workspaces with proration, so that I pay per client without emailing anyone to change my plan.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Design partners pay by invoice or Payment Link until S6 (decision D10). Credits are per org (`api/credit_service.py:220-258`) and the Pro lifecycle is in PR #103 (DOCBOT-1810: `org_subscriptions`, webhook-driven, one price).

## Scope
In:
- `STRIPE_FIRM_PRICE_ID` (name only; value set by the founder) and a Firm checkout reusing the 1810 webhook and lifecycle.
- Subscription quantity = active workspaces; update with proration on create and archive (2564).
- Price-id filter in the webhook so Pro and Firm events are told apart (PR #103 follow-up 1).
- Ignore stale `customer.subscription.deleted` for a different subscription id while the stored row is active or past_due (PR #103 follow-up 2).

Out:
- Customer portal UI.
- Metered overage (1809).

## Acceptance criteria
- [ ] Given a Firm org with 3 active workspaces, when a 4th is created, then the Stripe subscription quantity becomes 4 with proration, verified in Stripe test mode.
- [ ] Given a workspace is archived, when the change is saved, then quantity drops by 1 and the next invoice shows a proration credit.
- [ ] Given a webhook event for a price that is neither Pro nor Firm, when it arrives, then it is ignored and logged.
- [ ] Given an org re-subscribes after cancel, when a late `subscription.deleted` for the old subscription arrives, then the new active row is not overwritten.
- [ ] Given a bad signature or a replayed `event.id`, when the webhook receives it, then it returns 400 or is a no-op (security gate 4).

## Deliverable and approach
- Extend `api/subscription_service.py` from PR #103 once merged; quantity update called from `api/workspace_service.py` archive/create.
- Founder creates the Firm product and price in Stripe; agents never change Stripe config.

## Test / verification
- `tests/integration/test_stripe_subscription.py` extended with Stripe fixtures (no network).
- Founder test-mode run with screenshots of quantity and proration.

## Dependencies
Blocked by: 1810 (PR #103 merged), 2562, 2564
Blocks: Firm tier self-serve sale

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Minimum quantity (for example 3 workspaces) or pure per-workspace?
- Do design partners on $500/mo migrate to this price after their 6-month term?
