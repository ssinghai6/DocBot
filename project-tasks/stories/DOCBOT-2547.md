# DOCBOT-2547 Outreach sequence drafts
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 0 | - | P0 | 1 | PM | 11-19 | Open (pulled into EPIC-26) |

## User story
As a fractional CFO getting a cold LinkedIn DM, I want a short message that names my actual pain (tying out a client's board pack against the books) and offers a free tie-out of my last closed month, so that replying costs me nothing and I can judge the tool on my own numbers.

## Why now
Founder starts sending on 11-19 (25 per day on 11-19 and 11-20). Hook locked: "free tie-out of your last month".

## Scope
In:
- 3-touch sequence for LinkedIn DM and for email (day 0, day 3, day 7)
- Touch 1: the free tie-out hook; touch 2: trust (no SOC 2 yet, here is where data goes, DPA, read-only); touch 3: status-quo ("an hour of tie-out per client per month, fifteen clients is fifteen hours")
- Each touch tagged with the panel objection it answers (trust 18/20, status quo, CSV not SQL)
- Personalisation slots: {first_name}, {firm}, {client_count}, {one specific detail from source link}
- Reply handling snippets: "send me more", "not now", "is my data safe?", "how much?"
Out:
- Sending (founder only)
- Launch content and 40 DM batch for Phase 1 (2535)

## Acceptance criteria
- [ ] Given the drafts, when counted, then there are 3 DM touches, 3 email touches and 4 reply snippets
- [ ] Given each touch, when read, then it is under 90 words (DM) or 120 words (email) and names one objection it answers
- [ ] Given touch 1, when read, then it contains "free tie-out of your last month" and asks for a 15-minute call, not a signup
- [ ] Given the trust touch, when read, then it says "no SOC 2 report yet" and links the /security page (2531) or DPA, and never says "enterprise-grade"
- [ ] Given the "how much?" snippet, when read, then it quotes only locked pricing ($500/mo x 6 DP offer; free 3,000 credits; packs $20/$50/$150)
- [ ] Given the file header, when read, then it says "DRAFTS: founder sends; agents never send"

## Deliverable and approach
`project-tasks/gtm/2547-outreach-sequence.md`. Inputs: ten hooks in `market-gtm-plan.html` section 6, panel pass reasons in section 4, 2601 lines. CSV path line ("Bring a CSV from QuickBooks or Xero, no database needed") only once 2577 or the 2532 playbook confirms the path works.

## Test / verification
Word counts; banned-word grep; founder approval on 11-19 recorded in file; reply rate tracked in 2551 (target DM to call >= 15%, rewrite if < 8% after 50 sends).

## Dependencies
Blocked by: 2601, 2546 (personalisation)  Blocks: none (2551 measures it)

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Trust touch links /security, but 2531 is due 11-25 and sends start 11-19. Until then, offer the DPA "on request" instead of a link.
