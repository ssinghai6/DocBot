# DOCBOT-2534 Pitch/landing copy alignment
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 GTM | 1 | S6 | P1 | 2 | senior-dev | - | Open |

## User story
As a fractional CFO who clicks from a LinkedIn post to docbot.space, signs up and opens the app, I want every page I touch to describe the same tie-out job in the same words, so that I trust that the product I tried is the one I was pitched.

## Why now
By S6 the site has the 2537 landing rewrite, the 2531 security page and Firm workspace screens. Copy drift across signup, onboarding, chat empty state, metadata and README is likely, and Firm tier becomes sellable in S6.

## Scope
In:
- Sweep of all user-facing copy outside the landing hero: `src/app/layout.tsx` metadata, `src/app/signup`, `src/app/onboarding`, chat empty state and suggested questions in `src/components/ChatEmptyState.tsx` and `src/components/ChatArea.tsx`, the in-app `src/components/DataHandlingPanel.tsx`, README hero, deck v2 language
- Replace any remaining "enterprise", fine-tuning or accuracy wording with the 2601 approved substitutes ("eval loop", "built-in SSO, RBAC, encryption and audit log")
- Hero line verbatim per 2601
Out:
- Landing H1/badge and /landing redirect (done in 2537)
- New pages (2539)

## Acceptance criteria
- [ ] Given `main` after merge, when `grep -rn -i "enterprise-grade\|enterprise-ready\|enterprise security\|fine-tun\|self-improving\|20/20" src/ README.md` runs, then zero matches in user-facing strings
- [ ] Given the landing hero and the layout description, when compared to 2601, then the buyer line appears verbatim
- [ ] Given the chat empty state, when a new user opens /chat, then the first suggested question is a tie-out question
- [ ] Given any accuracy number in `src/`, when found, then it is the 2543 live number with date and N, or there is none
- [ ] Given CI, when the PR runs, then `npm run build` and lint pass and no snapshot test breaks

## Deliverable and approach
One PR on `feature/DOCBOT-2534-copy-alignment` touching only copy strings. Reuse the 2601 banned list as the grep pattern; consider adding it as a lint step only if trivial.

## Test / verification
Grep command above in PR description with output; manual click-through signup to chat on Vercel preview.

## Dependencies
Blocked by: 2601, 2537, 2543  Blocks: none

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- 2537 already removes the landing "Enterprise" claims and rewrites the H1, which is most of the original 2534 AC. Remaining scope fits 1 pt; backlog says 2, kept. PM to decide whether to resize or fold into 2537.
