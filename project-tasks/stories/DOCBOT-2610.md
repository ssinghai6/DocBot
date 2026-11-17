# DOCBOT-2610 Investor list + follow-up
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P1 | 1 | founder | 11-30 | In progress (EPIC-26, due by 12-01) |

## User story
As the founder with one investor call booked, I want a list of 15 to 25 more angels and pre-seed funds in fintech and AI infra with a follow-up plan, so that one call becomes a process and a "no" on 12-01 does not end the raise.

## Why now
A single-investor process has no leverage. If the 12-01 call goes well, the ask for intros happens on that call, so the list must exist before it.

## Scope
In:
- 15 to 25 named investors: name, firm, thesis fit (fintech, accounting tech, AI infra), check size, warm path (who can intro), source link
- Follow-up sequence: thank-you + data room within 24h of 12-01; intro-request note for the 12-01 investor; first-touch drafts for the list (founder sends)
- Pipeline columns: stage (identified, intro asked, meeting, diligence, pass/commit), next action, date
Out:
- Sending anything (founder sends)
- Institutional seed funds that require > $1M ARR

## Acceptance criteria
- [ ] Given the list on 11-30, when counted, then it has >= 15 and <= 25 investors, each with a source link and a stated thesis fit
- [ ] Given each row, when read, then it names a warm path or is marked "cold"
- [ ] Given the follow-up drafts, when read, then the 12-01 thank-you includes the 2605 data room link and a specific intro ask
- [ ] Given the pipeline file on 12-08, when checked, then every row has a stage and a next-action date
- [ ] Given the drafts, when grepped, then no banned claim from 2601 appears

## Deliverable and approach
`project-tasks/preseed/2610-investor-pipeline.csv` (gitignored) and `project-tasks/preseed/2610-followup-drafts.md`. Inputs: 2601 lines, 2605 link, 2607 ask.

## Test / verification
Row count and column completeness check; founder confirms drafts approved.

## Dependencies
Blocked by: 2605 (link in follow-ups)  Blocks: none

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
None
