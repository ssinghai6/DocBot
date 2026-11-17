# DOCBOT-2601 Positioning one-pager
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 1 | founder | 11-18 | In progress (EPIC-26, due by 12-01) |

## User story
As the solo founder pitching a pre-seed investor on 2026-12-01 and selling to fractional CFOs the same week, I want one signed-off page with the wedge, vision, privacy and why-now lines plus a banned-claims list, so that the deck, landing page, outreach and trust page all say the same thing and none of them overclaims.

## Why now
Every other EPIC-26 and GTM artifact (2602, 2531, 2537, 2547, 2606) copies its wording from this page. Two "locked" lines currently disagree (see Open questions); fixing that here is cheaper than fixing it in six places.

## Scope
In:
- The 5 layers from the EPIC-26 story table: wedge, vision, privacy, why now, why you
- One buyer-facing line (for site, DMs) and one investor variant (for deck)
- The claim audit table (phrase, status in code, say instead), copied from the tracker and checked against current `main`
- Banned-claims list with the approved substitute for each
- The only citable accuracy number: the DOCBOT-2543 live tie-out result, with its required wording template
Out:
- Deck slides (2602), landing copy edits (2537, 2534), outreach drafts (2547)
- Any new claim not already backed by shipped code

## Acceptance criteria
- [ ] Given the one-pager, when a reviewer compares each capability claim with the claim-audit table, then every claim maps to a shipped file (`api/hybrid_service.py`, `api/utils/discrepancy_detector.py`, `api/autopilot_service.py`, `api/db_service.py`, `api/sandbox_service.py`) or is marked "next"/"roadmap"
- [ ] Given the banned list, when searched, then it contains at least: "enterprise-grade", "self-improving fine-tuning", "runs fully offline / on your infra" (before 2609 ships), "private model mode demoed" (before 2609 ships), "SOC 2 compliant", and the 20/20 LLM extraction baseline, each with its approved substitute
- [ ] Given the one-pager, when read, then exactly one buyer-facing line is marked "use verbatim" and the other candidate is either dropped or marked investor-only
- [ ] Given the proof section, when read, then it cites only the DOCBOT-2543 live result using the template "X of 120 runs, [date], [model], 12-question benchmark on fictional demo filings", or says "pending 2543" if the run has not happened
- [ ] Given the page, when the founder reviews it, then a dated sign-off line ("Approved by Sanshrit Singhai, 2026-11-xx") is present
- [ ] Given the file, when grepped for the em dash character, then zero matches

## Deliverable and approach
File: `project-tasks/preseed/2601-positioning-one-pager.md` (folder is gitignored; this is a founder working doc). Max one printed page plus the claim table as an appendix.
Outline: 1) Buyer line (verbatim) 2) Investor line 3) Wedge / Vision / Privacy / Why now / Why you 4) Proof (2543 template) 5) Banned claims and substitutes 6) Sign-off.
Inputs: EPIC-26 section 1 and the claim audit in `project-tasks/docbot-v2-project-tracking.md`; section 2 "Positioning alignment" of `project-tasks/exec-plan/market-gtm-plan.html`; claim rule in `project-tasks/docbot-enterprise-roadmap.md` (line 19); `api/utils/llm_provider.py` to confirm the LLM is still Groq/Gemini only.

## Test / verification
Reviewer checklist: (1) each claim traced to a file or tagged "next"; (2) `grep -n -i "enterprise-grade\|fine-tun\|20/20\|offline" project-tasks/preseed/2601-positioning-one-pager.md` hits only inside the banned list; (3) `grep -c $'\xe2\x80\x94'` (em dash bytes) returns 0; (4) founder sign-off line present.

## Dependencies
Blocked by: none (proof line fills in after 2543)  Blocks: 2602, 2531, 2534, 2537, 2547, 2606

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Two lines are both called "locked". EPIC-26 wedge: "DocBot ties out the numbers in a board pack, lender report or 10-K against the client's books, and cites both sides when they disagree." Market-GTM buyer line: "DocBot reconciles your client's PDFs against their live books and flags every number that doesn't tie out, with a citation for each side." Recommend the EPIC-26 wedge for public copy: "flags every number" is a completeness claim the 2543 run may not support (any X below 120 contradicts "every"). Founder to pick.
