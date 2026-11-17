# DOCBOT-2602 Pitch deck v1 to v2
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 3 | founder | v1 11-22, v2 11-28 | In progress (EPIC-26, due by 12-01) |

## User story
As a pre-seed investor taking a 30-minute call on 2026-12-01, I want a 12-slide deck that shows the tie-out wedge, a measured proof number and real call traction, so that I can judge the founder, the market and the honesty of the claims in one pass.

## Why now
The investor call is fixed for 2026-12-01. v2 must carry quotes from real discovery calls (EPIC-26 gate: >= 3 quotable pains), which only exist after 11-23.

## Scope
In:
- 12 slides in this order: problem, wedge demo moment, product, how it works, proof (2543 number), privacy/deploy options (2608), market (SAM $60M to $240M), business model (locked pricing), traction (2551 numbers + quotes), competition, roadmap, team + ask (2607)
- Speaker notes per slide with the source of each number
- PDF export for the data room (2605)
Out:
- Live demo content (2603), raise terms (decided in 2607)
- Any slide citing FinanceBench, FinDER, FaithEval or the 20/20 extraction baseline

## Acceptance criteria
- [ ] Given deck v1 on 2026-11-22, when opened, then all 12 slides exist in the order above with placeholders only on traction and proof
- [ ] Given deck v2 on 2026-11-28, when each number is checked against its speaker-note source, then every number traces to 2543 output, 2551 tracker, `market-gtm-plan.html` section 1, or the locked pricing table
- [ ] Given the proof slide, when read, then it quotes the 2543 live result with date, model and N=120 and names the corpus as fictional demo filings; no offline 120/120 or 27/27 row appears
- [ ] Given the privacy slide, when read, then tier 1 is "Hosted (now)", tier 2 is "Private model mode (next)" and tier 3 is "Single-tenant in your AWS (roadmap)", and no slide says private mode was demoed
- [ ] Given the traction slide, when compared with the 2551 tracker row for the week of 11-27, then contacted, booked, held, LOI counts match exactly and at least 3 quotes are attributed by role (no names without consent)
- [ ] Given the full deck, when searched, then "enterprise-grade" and "fine-tuning loop" appear nowhere; "eval loop" is used instead

## Deliverable and approach
Source deck in the founder's slide tool; exports to `project-tasks/preseed/docbot-preseed-deck-v1.pdf` and `...-v2.pdf`. Agent drafts the outline and speaker notes in `project-tasks/preseed/2602-deck-outline.md`; founder builds the visuals.
Inputs: 2601 one-pager; `market-gtm-plan.html` sections 1 (SAM math, labelled as inference), 3 (competitor map: Fathom, Jirav, DataSnipper, ChatGPT), 5 (pricing: free 3,000 credits, packs $20/$50/$150, Firm $500 to $2,000 per client workspace, DP $500/mo x 6, Pro $79 later); 2608 diagram; 2607 ask; 2551 tracker.

## Test / verification
Claim-trace review: a second reader (PM agent) ticks each number against its source in the outline file. Mock calls in 2606 use v2 and log any slide that drew a confused question.

## Dependencies
Blocked by: 2601, 2543, 2607, 2608, 2551  Blocks: 2605, 2606

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- If 2543 has not run by 11-28, the proof slide says "measured method, live run pending" and shows the method only. Confirm the founder accepts that fallback.
- SAM is an inference from Lightcast and IBISWorld inputs, not a sourced figure; the slide must label it so.
