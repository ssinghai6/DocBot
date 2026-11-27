# DOCBOT-2531 Trust package (security page, DPA, questionnaire)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 GTM | 0 | - | P0 | 3 | PM | 11-25 | Open (pulled into EPIC-26, due 11-25) |

## User story
As a fractional CFO who signs off on lender and board numbers for 8 to 15 clients, I want a public security page, a DPA I can sign and pre-filled security questionnaire answers, so that I can say yes to uploading a client's books without a SOC 2 report.

## Why now
Simulated panel: 18 of 20 buyers passed on trust; 19 of 20 said a DPA with retention and deletion terms plus a completed questionnaire would change their answer. Without this, price cannot even be tested. Booked calls start 11-23.

## Scope
In:
- Public page `/security`: SSO (SAML, Google, GitHub), RBAC (viewer/analyst/admin), Fernet-encrypted credentials, append-only audit log, PII masking, read-only DB access (prompt + sqlglot AST + read-only transaction), data retention, sub-processors with data type, the 2608 data-flow diagram, and "No SOC 2 report yet" with the stated plan
- DPA template (processing scope, sub-processors, retention and deletion on request, breach notice, governing law) marked for lawyer review
- Security questionnaire answers (a short-form set of about 40 common SIG-Lite / CAIQ-style questions) with honest "No / Planned" answers where true
Out:
- SOC 2 work (parked EPIC-27; pulled only on a written ask)
- Trust center with NDA flow (parked 2713)
- Any new security feature

## Acceptance criteria
- [ ] Given docbot.space/security after deploy, when loaded, then it shows each item in Scope In, links the DPA request, and contains the sentence "We do not have a SOC 2 report yet" plus the plan wording agreed in Open questions
- [ ] Given the page source and the DPA, when grepped for "enterprise-grade", "SOC 2 compliant" or "certified", then zero matches
- [ ] Given each claim on the page, when traced, then it maps to shipped code (e.g. `api/rbac_service.py`, `api/audit_service.py`, `api/utils/sql_validator.py`, `api/utils/pii_masking.py`, `api/utils/encryption.py`) or is labelled "planned"
- [ ] Given the sub-processor table, when compared with the 2608 diagram, then the lists are identical (Vercel, Railway until cutover, Groq, Google Gemini, E2B, Stripe, Resend, Sentry)
- [ ] Given the DPA template, when reviewed, then it has retention and deletion terms with a number of days and is marked "lawyer reviewed on <date>" (or the gap is logged before first DP signature)
- [ ] Given the questionnaire file, when counted, then every question has an answer and an evidence pointer, and "No" answers are not reworded as "Yes"
- [ ] Given `src/app/sitemap.ts`, when read, then `/security` is listed

## Deliverable and approach
- Page: `src/app/security/page.tsx` (server component, static), add `/security` to `src/app/sitemap.ts`; SVG from 2608 into `public/`. Code by senior-dev or frontend lane, copy by PM.
- DPA: `project-tasks/gtm/2531-dpa-template.md` (export to PDF for signing).
- Questionnaire: `project-tasks/gtm/2531-security-questionnaire.md` (question, answer, evidence file).
- Retention facts: `COMMERCE_RAW_JSON_RETENTION_DAYS` (default 90) in `api/commerce_service.py`; session/document retention must be confirmed in code before stating a number.

## Test / verification
`npm run build` passes; page renders at /security; grep for banned words; senior-dev reviews each technical claim against code; founder reads DPA; footer link added on `/`.

## Dependencies
Blocked by: 2601, 2608  Blocks: 2533, 2549, 2552, 2605

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- AC says "no SOC 2 yet with a roadmap date", but SOC 2 is parked until a written buyer ask (backlog D2). A hard date contradicts the parking rule. Proposed wording: "SOC 2 Type I is planned after our move to AWS (early 2027); we start it when a customer needs it. Ask us for the current status." Founder to confirm.
- Retention period for uploaded PDFs, vector stores and CSV bytes is not documented in code I could find; needs a decision (and maybe a ticket) before the DPA states days.
- 3 pts is tight for page + DPA + questionnaire; keep 3 if the page reuses 2608 content and lawyer review is outside the estimate.
