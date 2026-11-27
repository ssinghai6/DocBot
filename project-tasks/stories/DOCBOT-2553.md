# DOCBOT-2553 Design-partner case studies (3)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 1 | S6 | P1 | 2 | PM | - | Open |

## User story
As a referral-driven fractional CFO who will not trust a vendor's own claims, I want to read how three peers used DocBot on their month-end close, with hours saved and discrepancies caught, so that I can believe it works on books like my clients'.

## Why now
Panel buyers P005 and P009 flip on peer proof. W12 (02-11) needs >= $4K MRR and the last push to 6 logos runs in S6 and S7 (2535 uses these stories).

## Scope
In:
- 3 case studies, one per paying DP, about 400 words each: firm profile (size, client count, anonymised if asked), before (hours per close per client from the 2548 baseline, tools), what DocBot did (closed months run, documents checked), results (hours per close after, discrepancies found with type, e.g. "board pack revenue $X vs GL $Y"), one quote, what DocBot missed
- Written client approval before publishing, per the 2549 consent clause
- Short versions (2 to 3 lines) for DMs and the deck
Out:
- Logos or names without written consent
- Accuracy percentages other than the 2543 live number

## Acceptance criteria
- [ ] Given each case study, when read, then hours per close before and after are both stated with the source (2548 baseline notes and DP-reported after figure)
- [ ] Given each case study, when read, then at least one real discrepancy found is described with document type and the two values, and one limitation ("what DocBot missed") is stated
- [ ] Given each published case study, when the file is checked, then a dated written approval from the DP is attached or referenced
- [ ] Given the text, when grepped, then no banned claim from 2601 appears and no client-of-client name or confidential figure appears without consent
- [ ] Given 3 approved case studies, when 2535 drafts are reviewed, then each draft that cites a peer links one of them

## Deliverable and approach
`project-tasks/gtm/2553-case-study-dp1.md`, `-dp2.md`, `-dp3.md`, plus `2553-approvals.md` (DP, date, scope of approval). Published versions can go to the site with 2539 content pages. Inputs: 2548 call notes baseline, DP weekly feedback calls, tie-out XLSX exports (DOCBOT-2014).

## Test / verification
DP approval present; second reader checks numbers against call notes and exports; banned-word grep.

## Dependencies
Blocked by: 2549, 2548, 3 paid DPs (W4)  Blocks: 2535, 2539

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- If fewer than 3 DPs consent, ship what is approved and note the gap; anonymised versions still need approval.
