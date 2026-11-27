# DOCBOT-2573 BYOK 'contact us' (BYOK = customer LLM key, EPIC-19)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S5 | - | 1 | senior-dev | - | To do |

## User story
As a firm evaluating DocBot, I want a 'contact us' option for bringing my own LLM key, so that I can register interest even though BYOK is not built yet.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Only an enum placeholder exists (`KeySource.byok` in `api/billing_models.py:112`). Real BYOK is EPIC-19 and deferred; the pricing page must say 'contact us', never that BYOK works.

## Scope
In:
- CTA on Firm pricing that records interest.
- Org flag `byok_interest_at` set by an authenticated POST; admin metrics count.

Out:
- Storing or using any customer LLM key (EPIC-19).

## Acceptance criteria
- [ ] Given a logged-in admin, when they click the BYOK contact CTA, then the org flag is set once and a confirmation shows.
- [ ] Given the pricing page, when read, then BYOK is labeled 'contact us' and no copy claims BYOK is available.
- [ ] Given admin metrics, when loaded, then the count of orgs with BYOK interest is shown.

## Deliverable and approach
- Column via `ADD COLUMN IF NOT EXISTS`; route with Pydantic; Zod on the frontend.

## Test / verification
- Route unit test; route-guard test.

## Dependencies
Blocked by: none
Blocks: none

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Also notify the founder by email on each interest click?
