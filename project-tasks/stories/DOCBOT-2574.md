# DOCBOT-2574 Priority support
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Firm workspace | 1 | S6 | - | 1 | PM | - | To do |

## User story
As a Firm customer, I want a priority support channel with a written response target, so that a close-week problem gets answered fast.

## Why now
Firm tier ($500-2,000/mo, about $50-75 per client workspace) cannot be sold as described until 2562, 2563, 2565 and 2569 ship (2561 gap analysis). Until then a design partner runs as one firm org with operator-moved users. Nothing exists in code; this is mainly a process item plus an in-app link.

## Scope
In:
- Support channel choice (shared inbox or Slack Connect) and a written response target in the Firm order form.
- In-app contact link that pre-fills org id and plan (no document contents).
- Short runbook for the founder: triage, response, escalation.

Out:
- 24/7 coverage or an SLA with credits.

## Acceptance criteria
- [ ] Given a Firm user, when they click contact support, then the message opens with org id and plan pre-filled and no client data.
- [ ] Given the Firm order form, when read, then it states the response target in business hours.
- [ ] Given the runbook, when the founder follows it on a test ticket, then the response time is logged in the 2551 tracker.

## Deliverable and approach
- PM writes the runbook and copy; senior-dev adds the link (small).

## Test / verification
- Test ticket end to end; screenshot of the pre-filled message.

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
- Response target: next business day or 4 business hours?
