# DOCBOT-2606 Investor Q&A prep
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 1 | founder | 11-29 | In progress (EPIC-26, due by 12-01) |

## User story
As the solo founder on the 12-01 investor call, I want written, rehearsed answers to the 20 questions I am most likely to get, so that I answer hard ones (solo founder, ChatGPT risk, no SOC 2) in under a minute without overclaiming.

## Why now
Two mock calls are scheduled 11-28 and 11-29. Answers must exist before the first mock.

## Scope
In:
- 20 questions with 60 to 90 word answers, each tied to a source (2601, 2607, 2551, market-gtm-plan)
- Must include: solo founder risk; why not ChatGPT/Claude/Copilot; data privacy and where client data goes; moat (DataSnipper and Fathom as fast copiers); LLM cost per tie-out and Groq price risk; why fractional CFOs; raise size and use of funds; why no SOC 2; what the eval number means; what happens if Groq changes pricing
- Two mock calls with notes on weak answers
Out:
- The deck itself (2602)
- Term negotiation strategy

## Acceptance criteria
- [ ] Given the Q&A doc, when counted, then it has 20 questions and every answer names its source
- [ ] Given the "why not ChatGPT" answer, when read, then it cites the method (page + ledger row citation, discrepancy flag) and the 2543 number, not the 20/20 extraction baseline
- [ ] Given the privacy answer, when read, then it says today LLM calls go to Groq/Gemini and code runs in E2B, embeddings and reranker run locally, and private model mode is next (not demoed)
- [ ] Given two mock calls held by 11-29, when the notes are read, then each lists the questions answered weakly and the doc shows a revised answer for each
- [ ] Given the doc, when grepped, then no "enterprise-grade" or em dash appears

## Deliverable and approach
`project-tasks/preseed/2606-investor-qa.md`: table of question, short answer, source, mock-call note. Inputs: 2601 claim audit, 2607 ask, `market-gtm-plan.html` sections 3 and 4 (competitors, panel pass reasons: 18/20 on trust), master backlog risk table (Groq price change row).

## Test / verification
Mock call notes attached in the same file; PM agent checks each answer against 2601 banned list.

## Dependencies
Blocked by: 2601, 2607  Blocks: none

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Per-tie-out LLM cost is not in any source file. Need one measured number (from the 2543 run's token usage or `SESSION_COST_CEILING_USD` traces) before answering the cost question with a figure.
