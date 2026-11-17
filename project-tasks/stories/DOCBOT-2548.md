# DOCBOT-2548 Discovery call script
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 0 | - | P0 | 1 | PM | 11-20 | Open (pulled into EPIC-26) |

## User story
As the founder on a 20-minute discovery call with a fractional CFO, I want a script with 5 qualifiers, clear disqualifiers and a notes template, so that every call ends with either a booked free tie-out or a clean no, and produces a quotable pain I can use on the traction slide.

## Why now
First calls are 11-23. EPIC-26 gate needs >= 3 quotable pains and 1 LOI or verbal DP yes by 12-01. Case studies (2553) later need a "before" baseline that only this call can capture.

## Scope
In:
- Call flow (20 min): context (2), current tie-out process (6), pain and cost (5), DocBot in one line (2), next step: book free tie-out on their last closed month (5)
- 5 qualifiers: number of clients (>= 8 for Firm, 3 to 6 for packs/Pro); recurring board/lender/investor reporting; books in QBO or Xero with CSV export available; hours per client per month on tie-out; decision maker = firm owner
- Disqualifiers: no recurring reporting, single client, needs SOC 2 report before any trial, non-US clients (D5), wants a live QBO connector before buying (parked EPIC-37)
- Baseline capture for 2553: hours per close per client, last error that reached a client, tools used
- Notes template and consent question for quoting ("can I quote this without your name?")
Out:
- The live tie-out demo on their docs (2550)
- Pricing negotiation beyond the locked DP offer

## Acceptance criteria
- [ ] Given the script, when read, then it has the 5 timed sections summing to 20 minutes, 5 qualifiers and at least 5 disqualifiers
- [ ] Given the notes template, when filled for a call, then it records client count, hours per client per month, current tools, pain quote, quote consent (yes/no), qualified (yes/no) and next step with date
- [ ] Given a mock call run by someone other than the author, when completed with the script, then the notes template is fully filled in 20 to 25 minutes
- [ ] Given the "DocBot in one line" section, when read, then it uses the 2601 buyer line verbatim
- [ ] Given the closing section, when read, then the only ask is booking a free tie-out on their last closed month (feeds 2550) or, if they already saw one, the DP order form (2533)

## Deliverable and approach
`project-tasks/gtm/2548-discovery-call-script.md` (script + qualifiers + disqualifiers) and `project-tasks/gtm/2548-call-notes-template.md`. Notes from each call roll up to the 2551 tracker (held, qualified, quotable pain).

## Test / verification
One mock call before 11-23; PM checks the template captures every field 2551 and 2553 need.

## Dependencies
Blocked by: 2601  Blocks: 2550, 2553, 2551 (call data)

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
None
