# DOCBOT-2605 Data room lite
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 2 | founder | 11-29 | In progress (EPIC-26, due by 12-01) |

## User story
As a pre-seed investor who liked the 12-01 call, I want one link with the deck, the eval result, architecture, security posture, roadmap, cap table and live metrics, so that I can do diligence in an evening without emailing the founder for each file.

## Why now
Investors ask for a data room within 24 hours of a good call. The 2610 follow-up pipeline sends this link to every new investor.

## Scope
In:
- One shared folder (or single index page) with: deck v2 PDF (2602), 2543 eval output (date, model, N, SHA), architecture diagram, security posture (2531 page link + 2608 diagram), roadmap (Now/Next/Later from the master backlog phases), cap table, metrics (signups, calls, LOIs from 2551)
- Index doc listing every file, its date and its source
Out:
- Financial model beyond the 2607 18-month plan
- Customer data, prospect names, or the 2546 prospect list

## Acceptance criteria
- [ ] Given the data room link, when opened in an incognito window by a link-holder, then all 7 item types listed above are present and open without a login request beyond the share link
- [ ] Given the metrics file, when compared with the 2551 tracker row for the latest Friday, then every number matches
- [ ] Given the eval file, when read, then it is the raw 2543 output with date, model (GROQ_MODEL value), N=10 runs x 12 questions and commit SHA, and no other accuracy number is in the room
- [ ] Given every document in the room, when searched, then no prospect or design-partner name appears without written consent and no secret, key or connection string appears
- [ ] Given the roadmap file, when read, then SOC 2, single-tenant deploy and private model mode are labelled "next" or "roadmap", not "shipped"

## Deliverable and approach
Share folder (Google Drive or DocSend, founder choice) plus index `project-tasks/preseed/2605-data-room-index.md` (item, file name, date, source ticket). Architecture diagram: reuse the CLAUDE.md architecture block redrawn, or the 2608 data-flow diagram.

## Test / verification
Incognito open test; second-reader check against the AC list; `grep -i "sk-\|api_key\|postgresql://"` over exported text files returns nothing.

## Dependencies
Blocked by: 2602, 2543, 2531, 2607, 2608, 2551  Blocks: 2610

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Cap table source: founder only; confirm whether it is shared pre-term-sheet or on request.
