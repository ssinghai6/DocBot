# DOCBOT-1627 Shopify webhook fix
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-16 remainder | 1 | S5 | P1 | 1 | senior-dev | - | Open |

## User story
As a merchant with a connected Shopify store, I want each Shopify webhook processed once and acknowledged quickly, so that Shopify does not retry-storm the endpoint and my sync is not run many times for one event.

## Why now
Finding SEC-20. On main the undefined `_connectors` bug is already gone (`shopify_webhook` in `api/routes/connector_routes.py` uses `_connector_cache`). Two parts remain: there is no dedupe on `X-Shopify-Webhook-Id`, and `sync_connector_data` runs inline on the request path, so a slow 24-hour sync can exceed Shopify's 5-second timeout and trigger retries.

## Scope
In:
- Dedupe on `X-Shopify-Webhook-Id` (Postgres table or existing store with TTL), duplicate returns 200 with no sync.
- Move the sync off the request path (APScheduler one-off job on the existing scheduler, or `asyncio` task with error logging).
- Tests.

Out:
- New Shopify topics or connector features.
- Amazon notifications.

## Acceptance criteria
- [ ] Given a valid signed webhook, when it is received, then the endpoint returns 200 in under 1 second and the sync is queued, not awaited.
- [ ] Given the same `X-Shopify-Webhook-Id` received twice, when the second arrives, then it returns 200 and no second sync is queued.
- [ ] Given a webhook with a bad HMAC, when received, then it still returns 401 and nothing is recorded.
- [ ] Given a queued sync that raises, when it fails, then the error is logged by type only and the webhook response was already 200.

## Deliverable and approach
- Dedupe helper and job queueing in `api/commerce_service.py` or `api/connectors/sync_jobs.py`; route stays thin.

## Test / verification
- Extend `tests/unit/test_shopify_connector.py` or a route test: dedupe, queue, bad signature.
- CI green.

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
- Should unknown shops keep returning 404, or 200 to avoid confirming which shops are connected?
