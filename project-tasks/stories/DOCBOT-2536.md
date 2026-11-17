# DOCBOT-2536 Landing SEO foundation
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 SEO | 0 | - | - | 2 | senior-dev | 11-19 | PR #99 open (held; pulled into EPIC-26) |

## User story
As a fractional CFO searching for a tie-out tool, I want docbot.space to show a clear title, description and preview card, so that I can find the product and understand it from a search result or a shared link.

## Why now
Shipped record. PR #99 merged to main on 2026-11-17, ahead of EPIC-26 outreach, so links sent to 150 prospects render a proper preview. The backlog status text predates the merge; Plane state is Done via the sync's state floor.

## Scope
In (shipped):
- `src/app/layout.tsx`: buyer-line title and description, OG and Twitter tags, self-canonical, `metadataBase` from `NEXT_PUBLIC_SITE_URL` via `src/lib/site.ts`.
- `src/app/robots.ts` (disallows /chat, /api/, /onboarding, /verify) and `src/app/sitemap.ts`.
- `src/app/page.tsx`: Organization and SoftwareApplication JSON-LD (free-tier Offer only).

Out:
- Copy realignment and /landing dedupe (2537), FAQ and llms.txt (2538).

## Acceptance criteria
- [ ] Given production, when `/robots.txt` is fetched, then it disallows /chat, /api/, /onboarding and /verify and links the sitemap.
- [ ] Given production, when `/sitemap.xml` is fetched, then it lists the canonical home URL on the `NEXT_PUBLIC_SITE_URL` host.
- [ ] Given the home page HTML, when inspected, then it has a canonical link, OG and Twitter tags and JSON-LD with only the free-tier Offer.

## Deliverable and approach
- Merged PR #99 (5 files above). No further work.

## Test / verification
- Merge commit on main; founder set `NEXT_PUBLIC_SITE_URL` on Vercel before merge; spot-check of robots, sitemap and page source on prod.

## Dependencies
Blocked by: none.
Blocks: 2537, 2538, 2539, 2540.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- None. Backlog and tracker status text should be updated to "Merged (PR #99)".
