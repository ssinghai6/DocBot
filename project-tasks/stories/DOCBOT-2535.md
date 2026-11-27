# DOCBOT-2535 Launch content + outreach drafts + funnel tracker
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 GTM | 1 | S7 | P1 | 3 | PM | - | Open |

## User story
As the founder sending every post and DM myself, I want ready-to-send launch drafts (3 public 10-K tie-out posts, 40 DMs, 3 emails, a launch post) and a weekly funnel tracker, so that my sales time goes to calls, not writing, and I can see which step of the funnel is broken.

## Why now
Phase 1 gate week (02-24 to 02-28) needs a final push to reach 6 paying logos, with Firm per-workspace billing live from S6. Funnel rates from Dec-Feb tell us which channel to keep.

## Scope
In:
- 3 public tie-out posts using EDGAR 10-K filings (no client data), each with the table of figures that did or did not tie and the method line
- 40 personalised DM drafts for firm owners with 8 to 15 clients (pulled from 2546), each mapped to one of the ten hooks in `market-gtm-plan.html` section 6
- 3 emails to free signups (closed-month offer, honest "what DocBot missed" nurture, Firm launch)
- 1 launch post for Firm tier
- Funnel tab in the 2551 tracker: DM to call (target >= 15%), call to proof-run (>= 50%), proof-run to paid (>= 33%)
Out:
- Sending anything (founder only)
- Paid ads, SEO pages (2539)

## Acceptance criteria
- [ ] Given the drafts folder, when counted, then it contains 3 posts, 40 DMs, 3 emails and 1 launch post, each marked "DRAFT, founder sends"
- [ ] Given each 10-K post, when a reader re-runs the cited figures in DocBot on the same EDGAR filing, then the stated tie/no-tie results reproduce
- [ ] Given each DM, when read, then it names the hook and objection it answers and offers "free tie-out of your last month"
- [ ] Given the funnel tab, when filled for any Friday, then it shows the three conversion rates with targets and a "change something if" threshold (DM to call < 8%, call to proof-run < 30%, proof-run to paid < 20%)
- [ ] Given all drafts, when grepped against the 2601 banned list, then zero matches; any accuracy number is the 2543 live number only

## Deliverable and approach
`project-tasks/gtm/2535-launch-content/` with `posts.md`, `dms.csv` (prospect id from 2546, hook #, text), `emails.md`, `launch-post.md`; funnel as a tab/section of the 2551 tracker. EDGAR runs use `api/edgar_service.py` ingest on prod. Case-study proof from 2553 if approved.

## Test / verification
Second reader reproduces one 10-K post in the app; banned-word grep; founder approval note per batch.

## Dependencies
Blocked by: 2546, 2551, 2543, 2553 (for DP quotes)  Blocks: 2539

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Timing: the GTM 30-day calendar puts 10-K posts and DMs in 11-17 to 12-15, but this ticket sits in S7 (02-24). The pre-Dec 1 DMs are covered by 2547. Recommend pulling the first 10-K post into Phase 0 under 2547 or re-titling this ticket "Firm tier launch content". Backlog sprint kept.
