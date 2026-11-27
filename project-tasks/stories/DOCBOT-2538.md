# DOCBOT-2538 FAQ + FAQPage JSON-LD + llms.txt
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 SEO | 1 | S6 | - | 2 | senior-dev | - | To do |

## User story
As a fractional CFO asking ChatGPT or Perplexity "can AI tie out a 10-K against the general ledger?", I want DocBot to show up with a short, quotable answer that states the method and the price, so that I find a tool built for this job without a sales call.

## Why now
AEO plan: answer-shaped FAQ blocks and comparison tables are what LLM search engines quote. Firm tier is sellable in S6, so the FAQ price answer can state it.

## Scope
In:
- 5-question FAQ block on `/`: "What is a financial statement tie-out?", "Can AI tie out a 10-K against the general ledger?", "How does DocBot cite each number?", "What does DocBot cost?", "Does DocBot replace my reviewer?" (answer: no, it flags and a human signs off)
- Each answer 40 to 60 words, opens with the claim, stands alone when quoted
- FAQPage JSON-LD on `/` matching the visible text exactly, using the same JSON-LD pattern 2536 added
- `public/llms.txt`: what DocBot is, who it is for, locked pricing, links to `/`, `/security` and FAQ anchor
Out:
- /tie-out, /fractional-cfo, /compare/* pages (2539)
- Search Console verification (2540)

## Acceptance criteria
- [ ] Given `/` after deploy, when loaded, then 5 FAQ items are visible and each answer is 40 to 60 words
- [ ] Given the page HTML, when pasted into the Schema.org validator or Google Rich Results test, then FAQPage parses with 5 Question entities and no errors, and each `acceptedAnswer.text` equals the visible answer
- [ ] Given the cost answer, when read, then it states only locked pricing: free 3,000 credits/mo after email verification, packs $20 / $50 / $150, Firm $500 to $2,000/mo per client workspace (Pro $79 only if 1810 is live)
- [ ] Given `curl -s https://docbot.space/llms.txt`, when run, then it returns 200 text with the sections above and no banned claims
- [ ] Given the FAQ text, when grepped, then no accuracy percentage, "enterprise-grade" or "fine-tuned" appears
- [ ] Given the nonce CSP (DOCBOT-2337), when the page loads, then the JSON-LD block causes no CSP console error

## Deliverable and approach
Branch `feature/DOCBOT-2538-faq-llms`. Files: `src/app/page.tsx` (FAQ section + JSON-LD script tag), `public/llms.txt`. FAQ copy drafted by PM from `market-gtm-plan.html` "AEO plan", approved by founder before merge.

## Test / verification
Validator screenshot in PR; curl for llms.txt; word count per answer in PR description; banned-word grep.

## Dependencies
Blocked by: 2537  Blocks: 2539, 2540

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Priority blank in backlog (SEO rows have none). Suggest P2: it does not move the Feb gate directly.
- If 1810 (Pro $79) is not live by S6, leave Pro out of the cost answer.
