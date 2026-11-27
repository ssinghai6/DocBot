# DOCBOT-2537 Landing copy realignment + /landing dedupe
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 SEO | 0 | - | - | 3 | senior-dev | 11-24 | To do (pulled into EPIC-26) |

## User story
As an investor or fractional CFO landing on docbot.space from the deck or a DM, I want the headline to say what the deck says (tie out a board pack or 10-K against the books, with citations) and no enterprise claims, so that the site matches the pitch and does not overclaim on security.

## Why now
Investors visit the site before 12-01. Today the badge says "Enterprise Document Intelligence" and the copy says "Enterprise-grade intelligence" and "Enterprise-ready", which breaks the no-SOC-2 claim rule. `/landing` is a near-duplicate of `/` that splits ranking.

## Scope
In:
- `src/app/page.tsx`: H1 (line 433) and sub-head to the 2601 buyer line; badge (line 430); remove "Enterprise Security" feature title (line 93), "Enterprise-grade intelligence across every data source" (line 614), "Enterprise-ready from day one" (line 784); rename the security card to "Built-in security" with SSO, RBAC, encryption, audit log wording
- 308 redirect `/landing` to `/` in `next.config.ts` `redirects()`; delete `src/app/landing/page.tsx`
- Keep metadata from 2536 (PR #99) consistent with the new H1
Out:
- FAQ and llms.txt (2538), content pages (2539)
- Converting the page to a server component (later perf item)

## Acceptance criteria
- [ ] Given the deployed `/`, when loaded, then the H1 or sub-head contains the 2601 buyer line verbatim
- [ ] Given `src/app/page.tsx`, when `grep -n -i "enterprise" src/app/page.tsx` runs, then zero matches in rendered strings
- [ ] Given `curl -sI https://docbot.space/landing`, when run after deploy, then status is 308 with `location: /`
- [ ] Given the repo, when listed, then `src/app/landing/page.tsx` no longer exists and `src/app/sitemap.ts` and `src/app/robots.ts` comments no longer mention a duplicate
- [ ] Given any accuracy claim on the page, when checked, then it is absent or is the 2543 live number with date and N
- [ ] Given CI, when the PR runs, then `npm run build` and lint pass

## Deliverable and approach
Branch `feature/DOCBOT-2537-landing-copy`. Files: `src/app/page.tsx`, `next.config.ts`, delete `src/app/landing/page.tsx`, touch `src/app/sitemap.ts` comment only. Must merge by 11-25, before the 2604 prod freeze on 11-27.

## Test / verification
Vercel preview: visual check of hero; curl for 308; grep output pasted in PR; founder signs off copy against deck v1.

## Dependencies
Blocked by: 2536 (Done), 2601  Blocks: 2538, 2534

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Tracker lists 3 "Enterprise" claims (L430/614/784); main has a fourth at L93 ("Enterprise Security" card). Included in scope; 3 pts still fits.
- Priority is blank in the backlog (SEO rows have no priority). Given it is pulled into EPIC-26 with an 11-24 due date it behaves as P0; PM to set.
