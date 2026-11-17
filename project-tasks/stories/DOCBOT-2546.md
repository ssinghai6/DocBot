# DOCBOT-2546 ICP prospect list
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 0 | - | P0 | 2 | founder | 11-21 | Open (pulled into EPIC-26) |

## User story
As the founder doing all outreach myself, I want a list of 150 named fractional CFOs and small advisory/CAS firm owners with a source link and segment for each, so that every DM goes to someone who runs month-end close and board or lender reporting for multiple clients.

## Why now
EPIC-26 gate by 12-01: 150 contacted, >= 12 calls booked. The only segment that bought in the simulated panel was firm owners with 8 to 15 clients, so the list must be weighted to them.

## Scope
In:
- 150 named by 11-21 (EPIC-26 cut); 400 by end of Phase 1 per the original AC
- Columns: id, name, firm, title, segment (firm owner 2-10 staff / solo fractional CFO / CAS lead at small CPA firm), est. client count, location (US only, decision D5), LinkedIn URL, source link, warm path, status
- Mix target: >= 60% firm owners with 8 to 15 clients; solo CFOs tagged for packs/Pro, not Firm
- Sources: LinkedIn Sales Navigator, fractional CFO directories and listicles (Rework, Knolli, Fractionus, Futureproof), CFO/CAS communities
Out:
- Enterprise targets (2552)
- Sending outreach (2547 drafts, founder sends)

## Acceptance criteria
- [ ] Given the CSV on 11-21, when counted, then >= 150 unique rows with no duplicate LinkedIn URL
- [ ] Given each row, when checked, then name, firm, segment and a source link are filled and the source link resolves
- [ ] Given the segment column, when tallied, then >= 60% are firm owners with an estimated 8+ clients
- [ ] Given a random sample of 10 rows, when a second reader checks the source, then >= 9 confirm the person offers fractional CFO, advisory or CAS services in the US
- [ ] Given the file location, when `git check-ignore` runs on it, then it is ignored (personal data never committed)

## Deliverable and approach
`project-tasks/gtm/2546-icp-prospects.csv` (gitignored folder). Agents may research and fill rows from public pages; founder owns Sales Navigator searches. Feeds 2547 personalisation and the 2551 "contacted" count.

## Test / verification
Script check: row count, uniqueness, required columns non-empty, segment share; 10-row spot check by PM.

## Dependencies
Blocked by: none  Blocks: 2547 (sends), 2551, 2535

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Title still says 400 named; EPIC-26 cut it to 150 by 11-21. Points (2) fit 150; reaching 400 is another 1 to 2 pts of founder time in Phase 1. Keep 2, flag for PM.
