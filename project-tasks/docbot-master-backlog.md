# DocBot Master Backlog (EPIC-25 to EPIC-40)

**Status:** Trimmed to a selling-first plan on 2026-11-17 (founder decisions 1-8, section 4). **Author:** senior PM agent.
**Inputs:** `project-tasks/docbot-enterprise-roadmap.md`, `project-tasks/docbot-robustness-test-data.md` (RB-01..RB-25), `project-tasks/docbot-v2-project-tracking.md` (EPIC-25, EPIC-26), `project-tasks/yc-10-day-plan.md` (EPIC-16/18/19/23 rows), EPIC-24 Stage A rows from branch `docs/DOCBOT-2400-strategy-roadmap`, `CLAUDE.md`, `git log`.
**Totals:** every point total, sprint load, founder-lane load, dependency order and the parking-lot rule in this file is generated and checked by a Python script (Fibonacci sizes, lanes, capacity per sprint, founder lane <= 8, no story before its dependencies, no EPIC-27..40 story in a sprint except the Phase 1 gate items). No total is hand-typed.

---

## 1. How to use this file

| Rule | Detail |
|---|---|
| One ticket per session | Take the top unblocked ticket from section 6, then follow the section 5 order. Finish, merge, then start the next. |
| Branch | `feature/DOCBOT-XXXX-short-desc` from `main` (docs-only: `docs/DOCBOT-XXXX-...`). One ticket per branch. |
| Definition of done | AC met; tests in `tests/unit` or `tests/integration` (external-API tests marked `@pytest.mark.external`); CI green; no secrets; no em dashes or banned claims in copy. |
| After merge | 1) Mark the row Done here and in `docbot-v2-project-tracking.md`. 2) Update the ticket in Plane (Jira is dropped as of 2026-11-17). 3) README only on full phase completion. |
| Parking lot | EPIC-27..40 stories stay in section 7 and out of sprints. A story is pulled only when a buyer pays (signed DP agreement, order form or LOI with money) or asks in writing (questionnaire, email, RFP). Log buyer, document and date in the 2551 gate tracker, then pull the smallest set of stories that answers the ask, plus their deps. |
| EPIC-26 | Due 2026-12-01. Changed on 2026-11-17 only as decided: 2609 code moved after Dec 1; 2532, 2550, 2551 pulled in. |
| Commits | Author Sanshrit Singhai; no Co-Authored-By Claude (CLAUDE.md). |
| Terminology | **BYOK** = customer brings their own LLM API key (EPIC-19, 2573). **CMEK** = customer-managed encryption keys (2807). |
| Claims | No 'enterprise-grade' before a SOC 2 report. FinanceBench, FinDER, FaithEval are internal eval only: never training, few-shot banks, prompt examples or public/investor claims. The 20/20 extraction baseline (1004) is never cited. Before 2609 ships, never say private model mode was demoed. |
| Lanes | founder (AWS/DNS/secrets/contracts/money), senior-dev (`api/**`), devops (`infra/aws/**`, workflows), ai-engineer (evals, retrieval, prompts), PM (docs, playbooks), frontend (`src/**`). |
| Track split | Revenue (R) = EPIC-25 tracks except Wave C hardening, EPIC-24 A, EPIC-16/18/19/23, EPIC-26, EPIC-36/37/40. Enterprise (E) = EPIC-27..35, 38, 39 + Wave C hardening 2521-2526. |

---

## 2. Phase overview

| Phase | Dates | Gate | Work | New pts scheduled | New pts parked (trigger phase) | Open existing pts in phase | Capacity | Scheduled pts (E share) |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-11-17 to 2026-12-01 | EPIC-26: investor call held; 150 contacted, >= 12 booked, >= 6 held, >= 3 quotable pains, 1 LOI or verbal DP yes | EPIC-26 (2609 code moved out); 2532, 2550, 2551 pulled in; AWS Wave A test env | 0 | 0 | 65 | EPIC-26 lanes | 8 (0% E) |
| 1 | 2026-12-02 to 2027-02-28 | **>= $5K MRR + >= 6 paying logos by 2027-02-28.** Stretch: $8.3K MRR ($100K ARR run-rate). No $30K enterprise line in the math | Design partners, AWS Wave B with mid-January cutover, Firm workspace, Stage A, EPIC-16 rest, 2609; only 2812 + 3501/3510/3512/3513 from EPIC-27..40 | 14 | 56 | 156 | 170 | 170 (14% E) |
| 2 | 2027-03-01 to 2027-06-30 | Opens only if the Phase 1 gate is met. Roadmap exit items (SOC 2 Type I, pooled contract, HA, sign-off) apply only once a buyer triggers them | Remaining EPIC-25 items (S8) + parking-lot pulls on paid or written asks | 0 | 449 | 23 | 242 | 23 (22% E) |
| 3 | 2027-07-01 to 2027-12-31 | As roadmap; all parked until triggered | Parked | 0 | 146 | 0 | n/a | n/a |
| unscheduled | n/a | n/a | Deferred existing tickets (2522, 2525, 2527-2530, 1809, EPIC-19) | n/a | n/a | 41 | n/a | n/a |

EPIC-27..40: **665 pts in 172 stories**, of which **14 pts are scheduled** (2812, 3501, 3510, 3512, 3513) and **651 pts are parked**. Phase 2 capacity is 242 pts; 23 are committed in S8 and the rest (219 pts) is reserved for parking-lot pulls.

Gate consistency: the old revenue-plan W2 gate (30 calls) is replaced by EPIC-26's numbers. Interim gates W4 (3 paid DPs), W8 (>= $2.5K MRR or 4 logos) and W12 (>= $4K MRR) are scaled from the new $5K target; confirm them (decision D11).

---

## 3. Epic summary

| Epic / track | Name | Phase(s) | Track | Stories | Pts | P0 | Scheduled pts | Source |
|---|---|---|---|---|---|---|---|---|
| EPIC-26 | Pre-seed Readiness | 0, 1 (2609) | R | 10 | 19 | 7 | 5 | tracker |
| EPIC-25 AWS | AWS migration 2501-2530 + 2576, 2578 | 0, 1, 2, 3 | E (Wave C) / R | 32 | 100 | 0 | 54 | tracker |
| EPIC-25 GTM | GTM 2531-2535 | 0, 1 | R | 5 | 15 | 3 | 10 | tracker |
| EPIC-25 SEO | SEO 2536-2540 | 0, 1, 2 | R | 5 | 14 | n/a (no pri in tracker) | 9 | tracker |
| EPIC-25 AI proof | AI proof 2541-2545 | 0, 1 | R | 5 | 11 | 3 | 5 | tracker |
| EPIC-25 Sales ops | Sales ops 2546-2553 + 2577 | 0, 1 | R | 9 | 17 | 7 | 13 | tracker |
| EPIC-25 Firm | Firm workspace 2561-2575 | 1, 2 | R | 15 | 51 | n/a (no pri in tracker) | 50 | tracker |
| EPIC-24 Stage A | Isolation sweep 2401-2406 | 1 | R | 6 | 18 | 3 | 18 | docs/DOCBOT-2400 branch |
| EPIC-16 rest | Security remnants 1613, 1614, 1626-1628, 1703, 1708 | 1 | R | 7 | 11 | 2 | 11 | yc-10-day-plan |
| EPIC-19 | BYOK (customer LLM keys) 1901-1906 | deferred | R | 6 | 18 | 0 | 0 | yc-10-day-plan |
| EPIC-23 (ref) | Open bugs 2304, 2308, 2309, 2312 | 1, 2 | R | 4 | 9 | 0 | 9 | yc-10-day-plan |
| EPIC-18 (ref) | Billing 1810 (PR #103), 1809 | 1 | R | 2 | 8 | 0 | 3 | yc-10-day-plan / tracker |
| EPIC-27 | SOC 2 and Compliance Program | 1, 2, 3 (if pulled) | E | 18 | 40 | 3 | 0 | parking lot |
| EPIC-28 | Security Engineering and Pen Test | 1, 2, 3 (if pulled) | E | 13 | 49 | 3 | 3 | parking lot |
| EPIC-29 | Enterprise Identity and Access | 2, 3 (if pulled) | E | 10 | 37 | 0 | 0 | parking lot |
| EPIC-30 | Tenancy and Fine-grained Permissions | 1, 2, 3 (if pulled) | E | 9 | 44 | 1 | 0 | parking lot |
| EPIC-31 | Data Governance, Retention and DLP | 1, 2, 3 (if pulled) | E | 11 | 38 | 0 | 0 | parking lot |
| EPIC-32 | Deployment and Privacy Tiers | 2, 3 (if pulled) | E | 11 | 55 | 0 | 0 | parking lot |
| EPIC-33 | Reliability and SRE | 1, 2, 3 (if pulled) | E | 14 | 55 | 3 | 0 | parking lot |
| EPIC-34 | AI Grounding, Quality and Safety | 1, 2 (if pulled) | E | 10 | 42 | 1 | 0 | parking lot |
| EPIC-35 | Robustness and Benchmark Test Program | 0, 2, 3 (if pulled) | E | 26 | 109 | 14 | 11 | parking lot |
| EPIC-36 | Workpaper Review and Sign-off | 2, 3 (if pulled) | R | 11 | 39 | 2 | 0 | parking lot |
| EPIC-37 | Connectors: Accounting, Warehouse, Content | 1, 2, 3 (if pulled) | R | 12 | 65 | 0 | 0 | parking lot |
| EPIC-38 | Admin Console and Customer Observability | 2, 3 (if pulled) | E | 8 | 34 | 0 | 0 | parking lot |
| EPIC-39 | API, SDK and MCP Server | 2, 3 (if pulled) | E | 7 | 30 | 0 | 0 | parking lot |
| EPIC-40 | Enterprise GTM Enablement and Commercial Ops | 1, 2 (if pulled) | R | 12 | 28 | 2 | 0 | parking lot |
| **All** | | | | **278** | **956** | **54** | **201** | |

Referenced totals after the 2026-11-17 additions (script-checked, mirrored in the tracker): AWS 100 (97 + 2578), Sales ops 17 (12 + 2577), GTM 15, SEO 14, AI proof 11, Firm 51, EPIC-26 19 (of which 14 before Dec 1; 2609 moved).

---

## 4. Changes from the 2026-11-17 trim (founder decisions)

| # | Decision | Change in this file | Change in tracker |
|---|---|---|---|
| 1 | EPIC-27..40 become a parking lot | All 172 stories kept in section 7; removed from sprints except 2812 (sandbox egress) and 3512 (PII pin) with its dep 3501; Wave C 2523/2524/2526 stay as SOC 2 evidence (S5, S6) | n/a (EPIC-27..40 live only here) |
| 2 | 2609 code after Dec 1; 2608 stays before | 2609 scheduled S4 | EPIC-26 table, day-by-day, after-Dec-1 table |
| 3 | 2532 + 2550 before Dec 1 as P0; hook 'free tie-out of your last month' | Both in S0 | Pulled-in table, day-by-day, traction line |
| 4 | Feb goal $5K MRR + 6 logos by 02-28; $100K ARR stretch; no $30K enterprise line; EPIC-26 call numbers | Phase 1 gate, sprint gates | EPIC-25 header gates; after-Dec-1 exit gates |
| 5 | Wave A before Dec 1; cutover 2519 mid-January after 3 stable DPs; 2551 starts now | Wave B spread S1-S4, 2519 in S4, 2520 in S5; 2551 in S0 | AWS status rows, after-Dec-1 table |
| 6 | 3510, 3501, 3513, 3512 not before Dec 1; in S1 | All four in S1 (overlaps decision 1; see D12) | n/a |
| 7 | New 2577 (QBO/Xero CSV intake) and 2578 (MultiFernet dual key) | 2577 S1 P0; 2578 S1, dep of 2509 | Rows added in sales ops and AWS tables; totals updated |
| 8 | Jira dropped; Plane | 'After merge' uses Plane | after-Dec-1 rule line |

New ticket AC:

| ID | Title | Pts | Pri | AC | Deps | Lane |
|---|---|---|---|---|---|---|
| DOCBOT-2577 | QBO/Xero CSV export intake for DP onboarding (CSV only, no connector) | 5 | P0 | QBO and Xero P&L, balance sheet, trial balance and GL CSV exports upload through the existing CSV path; title rows, date-range header, 'Total' rows, indented sub-accounts and parenthesized negatives handled in `csv_preprocessor`; a tie-out question on each fixture returns cited rows; fixtures + tests in `tests/` | 2532 | senior-dev |
| DOCBOT-2578 | DB_ENCRYPTION_KEY MultiFernet dual-key rotation | 3 | P1 | `DB_ENCRYPTION_KEY` accepts an ordered key list (new key first) via MultiFernet; decrypt works with old or new key; a re-encrypt script rotates stored creds and per-org DEK wrapping (1529) idempotently; tests cover old-only, both, new-only; 2509 runbook uses it | none | senior-dev |
| DOCBOT-2579 | Merge dev into main before prod apply | 1 | P1 | `dev` (with `infra/aws/**` and `.github/workflows/deploy-aws-prod.yml`) merged into `main`, because the `aws-prod` GitHub environment only accepts branch `main`; the merge also triggers a Railway prod deploy, so it runs in a quiet window and a golden-path smoke (health, PDF chat with citation, CSV query, login, billing debit) passes on prod afterwards | none (dev PRs #106-#117 merged) | founder |

---

## 5. Sprint plan (2026-11-17 to 2027-06-30)

Capacity: solo founder + AI agents at 28 pts per 2-week sprint (20 over the holidays, 26 over New Year, short sprints pro-rated), founder lane <= 8 per sprint. Order inside a sprint is the execution order; dependencies are always earlier (script-checked).

| Sprint | Dates | Cap | Tickets (in order) | Pts | Founder pts | E share | Gate check |
|---|---|---|---|---|---|---|---|
| S0 | 11-17 to 12-01 (EPIC-26 window) | EPIC-26 lanes | 2551, 2532, 2550 | 8 | 0 | 0% | EPIC-26 by 12-01: 150 contacted, >= 12 calls booked, >= 6 held, >= 3 quotable pains, 1 LOI or verbal DP yes. Hook live: 'free tie-out of your last month' |
| S1 | 12-02 to 12-15 | 28 | 2577, 2549, 2812, 3501, 3512, 2578, 2579, 2576, 2513, 3510, 3513 | 29 (1 over cap: 2579 added 2026-11-17) | 5 | 50% | >= 1 DP onboarded through 2577 CSV intake; sandbox egress off; prod env applied, soak clock running |
| S2 | 12-16 to 12-29 (holidays, cap 20) | 20 | 2509, 2510, 2514, 1613, 1614, 2517, 2544 | 20 | 3 | 0% | W4 (about 12-20): 3 paid DPs; G1 (1613/1614) closed; key rotation proven on prod env |
| S3 | 12-30 to 01-12 (New Year, cap 26) | 26 | 2515, 2516, 2518, 1810, 2545, 2401, 1628, 1626 | 26 | 0 | 0% | 3 DPs stable >= 2 weeks (cutover precondition); rehearsals + rollback drill passed; W8 (about 01-06): >= $2.5K MRR or 4 paying logos |
| S4 | 01-13 to 01-26 | 28 | 2519, 2402, 2403, 2406, 2562, 2565, 2312, 2609 | 28 | 3 | 0% | Cutover 2519 mid-January (go/no-go needs 3 stable DPs); Stage A P0 merged |
| S5 | 01-27 to 02-09 | 28 | 2520, 2523, 2524, 2563, 2564, 2566, 2567, 1627, 1708, 2573 | 28 | 2 | 21% | Railway cancelled (cutover + 14 days); Firm workspace core usable by 1 DP |
| S6 | 02-10 to 02-23 | 28 | 2526, 2568, 2569, 2570, 2575, 1703, 2553, 2552, 2534, 2574, 2538 | 28 | 5 | 11% | W12 (02-11): >= $4K MRR; Firm tier sellable with per-workspace billing |
| S7 | 02-24 to 02-28 (gate week, cap 12) | 12 | 2571, 2405, 2404, 2535 | 12 | 0 | 0% | PHASE 1 EXIT (02-28): >= $5K MRR + >= 6 paying logos. Stretch: $8.3K MRR ($100K ARR run-rate). No enterprise contract in the math |
| S8 | 03-01 to 03-14 | 28 | 2521, 2572, 2539, 2540, 2304, 2308, 2309 | 23 | 2 | 22% | Phase 2 opens only if the Phase 1 gate is met; 5 pts left for the first parking-lot pull |
| S9 to S16 | 03-15 to 06-30 | 214 | Reserved for parking-lot pulls (paid or written ask only) | 0 | 0 | n/a | Re-plan at each pull; if no pull, spend on revenue (Firm, SEO, EPIC-36/37 if a buyer asks) |

- Phase 1: 170 of 170 pts scheduled; enterprise share 14% (2812, the four robustness items and Wave C SOC 2 evidence). The roadmap's 80/20 split is superseded by decision 1.
- Phase 2: 23 of 242 pts committed (S8); the rest is reserve.

---

## 6. Next 10 tickets (execute in this order)

Before Dec 1 only EPIC-26 work runs: its own tickets, the pull-ins (2531, 2533, 2536, 2537, 2543, 2546-2548, and now 2532, 2550, 2551) and AWS Wave A. Nothing below starts before Dec 1.

| # | ID | Title | Pts | Lane | Why now |
|---|---|---|---|---|---|
| 1 | DOCBOT-2577 | NEW: QBO/Xero CSV export intake for DP onboarding (CSV only, no connector) | 5 | senior-dev | Design partners onboard on their own QBO/Xero exports; the free tie-out hook depends on it |
| 2 | DOCBOT-2549 | Design-partner agreement template | 1 | founder | DP agreement signed before the first paid DP |
| 3 | DOCBOT-2812 | NEW: Sandbox egress off (SANDBOX_PIP_INTERNET=false + custom E2B template) | 3 | senior-dev | Closes the sandbox exfil path before DP client data arrives (P0) |
| 4 | DOCBOT-3501 | RB-01 Dataset fetch + cache script, licence manifest (DATA_LICENSES.md), git-ignored cache | 3 | ai-engineer | Fetch script + licence manifest; dep of 3512 |
| 5 | DOCBOT-3512 | RB-12 PII eval: Gretel finance subset + must-not-mask set (pins the 10-digit bug class) | 3 | ai-engineer | Pins the PII 10-digit masking bug before DP ledgers flow through |
| 6 | DOCBOT-2578 | NEW: DB_ENCRYPTION_KEY MultiFernet dual-key rotation (needed for 2509) | 3 | senior-dev | Dual-key rotation, needed by 2509 before cutover |
| 7 | DOCBOT-2576 | Founder-only: prod apply, prod secrets, aws-prod env | 3 | founder | Founder prod apply; starts the soak |
| 8 | DOCBOT-2513 | 7-day prod-env soak + cost check | 2 | devops | 7-day soak clock |
| 9 | DOCBOT-3510 | RB-10 SQL-through-NL attack set: validator unit tests + DB checksum + cross-tenant read | 3 | senior-dev | SQL-through-NL attack tests on the read-only rule |
| 10 | DOCBOT-3513 | RB-13 prompt_guard detector eval on deepset/prompt-injections + finance hard negatives (CI) | 2 | ai-engineer | prompt_guard recall/FP on real injection data |

Next-10 total: 28 pts (all of S1). Founder: 2549, 2576.

---

## 7. Parking lot: EPIC-27 to EPIC-40

Every story is kept with its AC so it can be pulled without re-planning. 'Phase' is the phase the story would land in when pulled. 'Sprint' is `parked` unless it is one of the Phase 1 gate items.

| Trigger (written or paid) | Pull first |
|---|---|
| Security questionnaire or SOC 2 ask | 2701-2705, 2716, 3108, then 4001 (Type I needs a post-cutover window; never before 2519) |
| DPA retention / deletion terms | 3101, 3102 |
| SSO / SCIM / MFA requirement | 2572 (existing), 2901, 2903 |
| Wrong-number or citation concern | 3401 (needs 2312) |
| Uptime / SLA requirement | 3301-3303, then 3306 + 3314 + 3305 + 3307 (pgvector removes the single-task Chroma limit) |
| Reviewer sign-off before client delivery | 3001, 3601, 3611, 3603, 3604, 3605 |
| QBO/Xero live connection (beyond 2577 CSV) | 3712, 3703, 3701, 3702 |
| Pen test report | 2802, 2813 |
| Private / single-tenant deployment | 3201, 3205, 3202, 3203 |

### 7.1 EPIC-27: SOC 2 and Compliance Program (18 stories, 40 pts, track E)

Audit window and Type I date only after cutover 2519; Railway never in scope. Split: 2712 + 2718 (NEW).

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-2701 | Select compliance automation platform; connect AWS, GitHub, Google Workspace, Vercel | 2 | P1 | 1 | Contract signed within the D2 budget; 4 integrations show green; first automated control test exported | 2503 | founder | parked |
| DOCBOT-2702 | System description and scope (AWS prod only; sub-processors listed) | 2 | P1 | 1 | Scope names AWS prod only and states Railway is out of scope; Vercel, Groq, Gemini, E2B, Stripe, Resend, Sentry listed with data types; audit window start = after 2519 | 2701 | PM | parked |
| DOCBOT-2703 | Adopt the 10-policy set (infosec, access, change, IR, BCP/DR, vendor, data class, retention, SDLC, AUP) | 3 | P1 | 1 | 10 policies edited to match real practice; each has owner + review date; founder e-signs in platform | 2701 | PM | parked |
| DOCBOT-2704 | Risk assessment and risk register, reviewed quarterly | 2 | P1 | 1 | >= 20 risks scored with owner and treatment, including roadmap top-10; quarterly review on calendar | 2703 | PM | parked |
| DOCBOT-2705 | Vendor register with each sub-processor's SOC report and DPA on file | 2 | P1 | 1 | Every sub-processor has SOC report (or equivalent) + DPA on file or a logged gap; links to 3108 terms | 3108 | PM | parked |
| DOCBOT-2706 | Change-management evidence for a solo founder (compensating controls) | 3 | P1 | 2 | Branch protection + required CI on main; prod deploy needs aws-prod approval; compensating-controls memo signed | 2512, 2701 | devops | parked |
| DOCBOT-2707 | Quarterly access review + onboarding/offboarding checklist | 2 | P1 | 2 | First review done on the 2526 template with evidence; checklist covers AWS, GitHub, Vercel, Stripe, Google | 2526 | founder | parked |
| DOCBOT-2708 | Security training and background checks for anyone with prod access | 1 | P2 | 2 | Training record in platform for each prod user; background check done or documented exception | 2701 | founder | parked |
| DOCBOT-2709 | Incident response plan + one tabletop exercise | 2 | P1 | 2 | IR plan with severities and comms; tabletop run on a 3303 runbook with notes and action items | 2703, 3303 | PM | parked |
| DOCBOT-2710 | BCP/DR plan linked to 3308 drills | 2 | P1 | 2 | Plan states RTO 4h / RPO 15m; cites 3308 drill evidence | 3308 | PM | parked |
| DOCBOT-2711 | Auditor selection, readiness assessment, gap list | 3 | P0 | 2 | Auditor engaged; readiness assessment done; every gap mapped to a ticket | 2702, 2703, 2704, 2705, 2706 | founder | parked |
| DOCBOT-2712 | SOC 2 Type I audit fieldwork and report | 3 | P0 | 2 | Type I report issued (Security TSC) with a point-in-time date after 2519; unqualified opinion; stored for NDA release | 2711, 2718, 2707, 2709, 2710, 2519 | founder | parked |
| DOCBOT-2713 | Trust center page (report under NDA, sub-processors, pen test summary; extends 2531) | 2 | P1 | 2 | Page live with NDA request flow; copy says only 'SOC 2 Type I report available' and only after 2712 | 2712, 2531, 2802 | frontend | parked |
| DOCBOT-2714 | Type II observation window: monthly evidence checks, exception log | 3 | P1 | 3 | Window start >= 2519 cutover and >= Type I date; monthly check logged; exceptions remediated or accepted | 2712 | PM | parked |
| DOCBOT-2715 | SOC 2 Type II audit and report | 3 | P1 | 3 | Type II report issued with zero unremediated exceptions | 2714 | founder | parked |
| DOCBOT-2716 | Cyber liability + tech E&O insurance in place | 1 | P1 | 1 | Policies bound; certificates on file for 4009 procurement pack and MSAs | none | founder | parked |
| DOCBOT-2717 | Compliance calendar and control-owner matrix | 1 | P2 | 2 | Every control has owner + frequency; reminders configured in platform | 2703 | PM | parked |
| DOCBOT-2718 | NEW: Type I evidence collection across in-scope controls | 3 | P0 | 2 | Evidence attached for 100% of in-scope controls; no control left 'not started' | 2711 | PM | parked |

### 7.2 EPIC-28: Security Engineering and Pen Test (13 stories, 49 pts, track E)

**2812 (NEW, scheduled S1):** `api/utils/sandbox_policy.py` grants internet to LLM-codegen runs that import a non-prebaked package (`SANDBOX_PIP_INTERNET` default on), so an injected payload can exfiltrate alongside `import statsmodels`. Split: 2802 + 2813 (NEW). CMEK (2807) is not BYOK.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-2801 | Threat model refresh (STRIDE) for AWS, workspaces, invites, API keys, sandbox | 3 | P1 | 2 | STRIDE doc covers each surface; every threat maps to a control or a ticket | 2519, 2563 | senior-dev | parked |
| DOCBOT-2802 | Third-party pen test (web, API, cloud config): engagement | 3 | P0 | 2 | Vendor contracted; scope web + API + AWS config; report received | 2519, 2521, 1628 | founder | parked |
| DOCBOT-2813 | NEW: Pen test remediation (split from 2802) | 5 | P0 | 2 | Zero open critical/high; vendor retest confirms; NDA summary ready for 4012 | 2802 | senior-dev | parked |
| DOCBOT-2803 | Vulnerability management policy with SLAs + tracker | 2 | P1 | 2 | Critical 7 days, high 30 days; tracker fed by CI scanners (1625) and 2805; one quarter of SLA evidence | 2703 | PM | parked |
| DOCBOT-2804 | DAST baseline (OWASP ZAP) against dev env in CI | 3 | P2 | 2 | Weekly ZAP baseline vs api-dev; committed baseline; new high fails the job | 2505 | devops | parked |
| DOCBOT-2805 | Container image scanning and signing (Trivy + cosign), verified at deploy | 3 | P1 | 2 | Trivy blocks critical in CI; prod deploy refuses an unsigned digest | 2512 | devops | parked |
| DOCBOT-2806 | Customer DB credentials moved to KMS-backed Secrets Manager references | 5 | P1 | 2 | New connections store only an ARN ref; existing creds migrated with rollback; DB dump has no ciphertext creds | 2523 | senior-dev | parked |
| DOCBOT-2807 | CMEK: customer KMS key wraps the org DEK; revoke = unreadable | 8 | P2 | 3 | Grant revoke makes reads fail with a clear error; re-grant restores; audit event. CMEK is not BYOK (EPIC-19) | 1529, 2523, 2806 | senior-dev | parked |
| DOCBOT-2808 | Egress allowlist proxy for API tasks | 5 | P2 | 3 | Only LLM, E2B, Stripe and customer DB hosts reachable; blocked call logged; smoke passes | 2521 | devops | parked |
| DOCBOT-2809 | Customer DB network options: static egress IP, SSH tunnel (PrivateLink later) | 5 | P1 | 2 | Static egress IP published; SSH-tunnel connection type with tests; PrivateLink design note filed | 2521 | senior-dev | parked |
| DOCBOT-2810 | Vulnerability disclosure policy + security.txt (no bounty before SOC 2) | 1 | P2 | 2 | /.well-known/security.txt served; VDP page live | none | PM | parked |
| DOCBOT-2811 | Annual pen test retest | 3 | P2 | 3 | Retest report with zero open critical/high | 2813 | founder | parked |
| DOCBOT-2812 | NEW: Sandbox egress off (SANDBOX_PIP_INTERNET=false + custom E2B template) | 3 | P0 | 1 | Template bakes statsmodels, xgboost, lightgbm, tabulate; E2B_SANDBOX_TEMPLATE + SANDBOX_PIP_INTERNET=false in dev and prod; regression test: injected requests.post to a canary fails while a statsmodels query passes | none | senior-dev | S1 |

### 7.3 EPIC-29: Enterprise Identity and Access (10 stories, 37 pts, track E)

References 2571, 2572, 2567, 1626, 1703.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-2901 | MFA (TOTP + recovery codes), enforceable per org | 5 | P1 | 2 | Enroll + hashed recovery codes; org can require MFA; password login blocked without 2nd factor when required | 1703 | senior-dev | parked |
| DOCBOT-2902 | Generic per-org OIDC SSO (Okta OIDC, Google Workspace) | 5 | P1 | 2 | Per-org OIDC config; JIT into org; mock-IdP integration test | 2572 | senior-dev | parked |
| DOCBOT-2903 | SCIM 2.0 users endpoint; deprovision revokes sessions and API keys | 5 | P1 | 2 | Create/update/deactivate pass Okta SCIM validator; deactivate revokes sessions + keys within 60 s | 2566, 2567 | senior-dev | parked |
| DOCBOT-2904 | SCIM groups mapped to roles and workspace grants | 3 | P2 | 2 | Group add grants role/workspace; group removal revokes | 2903, 3001 | senior-dev | parked |
| DOCBOT-2905 | Org session policy: idle timeout, max session, IP allowlist | 3 | P2 | 2 | Each policy enforced per org with tests | 2566 | senior-dev | parked |
| DOCBOT-2906 | Step-up re-auth (<= 15 min) before export, API key, role or SSO change | 3 | P2 | 2 | Stale auth returns step_up_required; UI re-prompts; tests per action | 2901 | senior-dev | parked |
| DOCBOT-2907 | Scoped API keys with expiry + service accounts | 3 | P1 | 2 | Keys carry scopes + expiry; expired key 401; service account survives member removal | 1530 | senior-dev | parked |
| DOCBOT-2908 | Login and SSO events in audit log + simple anomaly flag | 2 | P2 | 2 | Login/SSO events logged; flag on new country or >= 10 failures/hour | 2570 | senior-dev | parked |
| DOCBOT-2909 | Passkeys (WebAuthn) | 5 | P3 | 3 | Register + login; usable as MFA factor | 2901 | senior-dev | parked |
| DOCBOT-2910 | Okta Integration Network and Entra app gallery listings | 3 | P3 | 3 | Both listings approved | 2902, 2903 | founder | parked |

### 7.4 EPIC-30: Tenancy and Fine-grained Permissions (9 stories, 44 pts, track E)

3006 extends 2401 once 2568 lands.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3001 | Permission model v2: per-workspace preparer / reviewer / approver / client-viewer | 5 | P1 | 2 | Roles stored per workspace; org roles migrated; one helper guards routes; tests per role | 2568 | senior-dev | parked |
| DOCBOT-3002 | Resource-level grants on connections and document sets | 5 | P1 | 2 | Ungranted member gets 404 on query; route tests | 3001 | senior-dev | parked |
| DOCBOT-3003 | Restricted columns: masking before the LLM sees SQL/CSV results | 5 | P2 | 2 | Masked value absent from prompt log and UI; tests on SQL + CSV paths | 3002 | senior-dev | parked |
| DOCBOT-3004 | Org policy switches (sandbox, EDGAR, external LLM, exports, personas) | 3 | P1 | 2 | Disabled feature returns 403 with reason; toggle writes audit event | 3001 | senior-dev | parked |
| DOCBOT-3005 | Postgres RLS ENABLE + FORCE on tenant tables; app role NOBYPASSRLS | 5 | P1 | 2 | 3006 matrix passes with app-layer filter disabled in a test build | 2402, 3006 | senior-dev | parked |
| DOCBOT-3006 | Isolation test matrix in CI: org x workspace x role, every route + SSE | 5 | P0 | 1 | Every route and stream exercised for 2 orgs x 2 workspaces x roles; any cross access fails CI; extends 2401 | 2401, 2568 | senior-dev | parked |
| DOCBOT-3007 | Client guest access: read-only to one workspace | 3 | P2 | 2 | Guest sees one workspace read-only; cannot query connections | 3001, 2565 | senior-dev | parked |
| DOCBOT-3008 | Custom roles per org | 5 | P3 | 3 | Admin composes roles from permissions; checks use permissions not role names | 3001 | senior-dev | parked |
| DOCBOT-3009 | Multi-entity workspaces for consolidation tie-outs | 8 | P3 | 3 | Parent/child entities; consolidation tie-out lists eliminations | 3601 | senior-dev | parked |

### 7.5 EPIC-31: Data Governance, Retention and DLP (11 stories, 38 pts, track E)

3512 pins the PII 10-digit bug class (fixed in 2003); 3105 must keep it at 0 false masks.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3101 | Retention policy engine: per-org defaults matching the DPA | 5 | P1 | 1 | Defaults for sessions, uploads, vectors, traces, LLM logs match 2531 DPA; daily job deletes + writes evidence row; frozen-clock tests | 2531 | senior-dev | parked |
| DOCBOT-3102 | Verified deletion: job, receipt, audit event (extends 1708, 2575) | 3 | P1 | 2 | Receipt lists counts per store + timestamp + hash; audit event written | 1708, 2575, 3101 | senior-dev | parked |
| DOCBOT-3103 | Crypto-shred on org/workspace delete (destroy per-org DEK) | 3 | P2 | 2 | Restored backup cannot decrypt a deleted org's creds; test | 1529, 3102 | senior-dev | parked |
| DOCBOT-3104 | Full org/workspace export in an open format | 3 | P2 | 2 | Zip of docs, workpapers (JSON/XLSX), audit CSV; step-up gated | 1708, 2906 | senior-dev | parked |
| DOCBOT-3105 | DLP on ingest (SSN, bank, card) with block / mask / warn policy | 5 | P2 | 2 | Detectors on PDF + CSV ingest; org policy honored; 3512 must-not-mask set shows 0 false masks | 3512, 3004 | ai-engineer | parked |
| DOCBOT-3106 | Per-org redact-before-LLM + per-request provider/model/region record | 3 | P1 | 2 | Flag masks PII before prompt; every LLM call logs provider, model, region | 1501 | senior-dev | parked |
| DOCBOT-3107 | PII canary test across Sentry, LangSmith, CloudWatch, LLM cache | 2 | P1 | 2 | Nightly test ingests canary and asserts absence in all 4 sinks | 1618 | ai-engineer | parked |
| DOCBOT-3108 | Sub-processor terms: no-training + retention confirmed; list published | 2 | P1 | 1 | Written terms on file for Groq, Google, E2B (+Vercel, Stripe, Resend, Sentry); trust page list updated | 2531 | founder | parked |
| DOCBOT-3109 | DSAR process (access, delete) with 30-day clock | 2 | P2 | 2 | Runbook + intake; one test DSAR completed via 3102/3104 | 3102, 3104 | PM | parked |
| DOCBOT-3110 | Legal hold flag suspends deletion for a workspace | 2 | P3 | 3 | Deletion jobs skip held workspace; set/clear audited | 3101 | senior-dev | parked |
| DOCBOT-3111 | Data residency: region pinning; EU stack only on a signed EU contract | 8 | P3 | 3 | Contract-triggered; org pinned to region end to end | 3307 | devops | parked |

### 7.6 EPIC-32: Deployment and Privacy Tiers (11 stories, 55 pts, track E)

2609 (env-level private mode) is in S4; Tier 2 GA (3202) needs EPIC-19 1902/1903.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3201 | Tier matrix spec + automated 'what leaves the box' egress test per tier | 3 | P1 | 2 | Spec per tier (extends 2608); CI test asserts allowed outbound hosts per mode with mocks | 2608, 2609, 2812 | senior-dev | parked |
| DOCBOT-3202 | Private model mode GA: per-org endpoint + eval-gated model list | 5 | P1 | 2 | Org-level OpenAI-compatible endpoint overrides env; only gated models selectable; fallback off for that org | 2609, 1902, 1903, 3404 | senior-dev | parked |
| DOCBOT-3203 | Self-hosted Python sandbox for private mode (replaces E2B) | 8 | P1 | 2 | No-network container runner with CPU/mem/time limits behind the E2B interface; 3511 suite passes on it | 2812, 3511 | devops | parked |
| DOCBOT-3204 | Customer-hosted single-node bundle with license key | 8 | P2 | 2 | Compose bundle (local Postgres, local embeddings) installs on a clean VM in < 1 h; smoke passes | 3202, 3203 | devops | parked |
| DOCBOT-3205 | BYOC architecture decision (this repo infra/aws vs docbot-platform engine) | 2 | P1 | 2 | ADR signed; default one engine with tier flags (D1) | none | founder | parked |
| DOCBOT-3206 | BYOC deploy into customer AWS via cross-account role + ExternalId | 8 | P2 | 3 | Deploy and teardown in a test customer account; no DocBot creds left behind | 3205 | devops | parked |
| DOCBOT-3207 | BYOC releases: signed images, channels, customer-approved upgrades | 5 | P2 | 3 | Upgrade requires customer approval; rollback tested | 3206, 2805 | devops | parked |
| DOCBOT-3208 | Telemetry-off mode + opt-in metadata-only beacon | 3 | P2 | 2 | Flag disables Sentry/LangSmith/analytics; 3201 test proves it | 3201 | senior-dev | parked |
| DOCBOT-3209 | BYOC break-glass with customer approval and tenant-visible log | 3 | P2 | 3 | Access time-boxed and visible to tenant | 3807, 3206 | devops | parked |
| DOCBOT-3210 | Sizing and cost guide per tier | 2 | P2 | 2 | Instance sizes, monthly cost range, GPU needs per tier | 3204 | PM | parked |
| DOCBOT-3211 | Air-gapped bundle (contract-triggered) | 8 | P3 | 3 | Offline images + model tarballs + license; 3201 test shows zero outbound | 3204 | devops | parked |

### 7.7 EPIC-33: Reliability and SRE (14 stories, 55 pts, track E)

If uptime is pulled, pgvector (3306 + 3314) goes first: Chroma on one EFS volume pins prod to `desired_count <= 1`.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3301 | SLOs and error budgets (availability, TTFT p95, tie-out success) + dashboard | 3 | P1 | 1 | 3 SLOs defined; CloudWatch dashboard; burn-rate alarm fires in a forced test | 2511, 2519 | devops | parked |
| DOCBOT-3302 | Public status page + incident comms templates | 2 | P1 | 1 | Status page fed by health checks; 3 comms templates | 2519 | devops | parked |
| DOCBOT-3303 | Paging + runbooks for top 10 failure modes | 5 | P1 | 1 | 10 runbooks (LLM down, E2B down, RDS full, stuck sync, ...); test page reaches founder phone | 2511 | devops | parked |
| DOCBOT-3304 | Postmortem process and template | 1 | P2 | 1 | Template + rule: every SEV1/2 gets a postmortem within 5 days | none | PM | parked |
| DOCBOT-3305 | Background worker for Autopilot, ingest, connector sync | 8 | P1 | 2 | Jobs run in a worker service; API restart does not kill jobs; job state persisted | 2406 | senior-dev | parked |
| DOCBOT-3306 | pgvector vector store with org/workspace scoping (first Phase 2 build) | 8 | P0 | 2 | Vector interface with pgvector backend; queries filtered by org + workspace; Chroma behind a flag | 2403, 2563 | senior-dev | parked |
| DOCBOT-3314 | NEW: Chroma to pgvector migration + retrieval eval gate (split from 3306) | 5 | P0 | 2 | All live sessions moved; existing retrieval eval Recall@5 >= Chroma baseline - 0.02; Chroma/EFS removed from prod | 3306 | ai-engineer | parked |
| DOCBOT-3307 | Multi-AZ RDS + 2+ API tasks across AZs | 5 | P0 | 2 | desired_count >= 2; killing one task causes no SLO breach; RDS Multi-AZ failover tested | 3314, 3305 | devops | parked |
| DOCBOT-3308 | DR: cross-region backup copy + quarterly restore drill (extends 2510) | 3 | P1 | 2 | Cross-region copy on; drill meets RTO 4h / RPO 15m, recorded | 2510 | devops | parked |
| DOCBOT-3309 | Load test at 10x peak + capacity plan | 3 | P1 | 2 | Locust (3521) at 10x peak within SLO; capacity plan doc | 3521, 3307 | devops | parked |
| DOCBOT-3310 | LLM provider health routing + circuit breaker + degraded banner | 3 | P2 | 2 | Breaker opens after N failures; banner shows; recovers automatically | none | senior-dev | parked |
| DOCBOT-3311 | Alembic expand/contract migrations, zero-downtime deploys | 5 | P1 | 2 | Alembic baseline equals prod; one expand/contract migration shipped with zero failed requests | none | senior-dev | parked |
| DOCBOT-3312 | Quarterly game day | 2 | P3 | 3 | One AZ-loss game day with findings ticketed | 3307 | devops | parked |
| DOCBOT-3313 | Contractual SLA (99.9%) with service credits | 2 | P2 | 3 | Offered only after 90 measured days at target | 3301, 3307, 4002 | founder | parked |

### 7.8 EPIC-34: AI Grounding, Quality and Safety (10 stories, 42 pts, track E)

3401 needs 2312 (scheduled S4).

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3401 | Citation verifier v1: every number maps to a doc span or data row, else 'unverified' | 8 | P0 | 1 | Gold set: >= 99% of numbers matched or marked unverified; unmatched numbers rendered as unverified in UI | 2312, 2541 | ai-engineer | parked |
| DOCBOT-3402 | Deterministic arithmetic: sums, deltas, % recomputed in code; mismatch blocks answer | 5 | P1 | 2 | Recompute on every numeric claim; mismatch blocks with reason; tests | 3401 | ai-engineer | parked |
| DOCBOT-3403 | Abstain policy when retrieval confidence is low | 3 | P1 | 2 | 'Not enough evidence' path; false-answer rate measured on 3515 | 3515 | ai-engineer | parked |
| DOCBOT-3404 | Release gate: model + prompt registry; ship only if gold set passes with margin | 5 | P1 | 2 | CI blocks a model/prompt change that drops tie-out accuracy; extends 1503/1507 | 1503, 1507, 3518 | ai-engineer | parked |
| DOCBOT-3405 | Prompt-injection defense v2: corpus in CI + per-step tool allowlist | 5 | P1 | 2 | 3509 corpus runs in CI; Autopilot steps use allowlists; P0 cases 0% success | 3509 | ai-engineer | parked |
| DOCBOT-3406 | Red team: internal agent pass, then external pass; fixes | 3 | P1 | 2 | Both passes run with 3519 runbook; every finding fixed or ticketed | 3519 | ai-engineer | parked |
| DOCBOT-3407 | Prod quality sampling: LLM judge on sampled traces per org + drift alerts | 5 | P2 | 2 | Daily sample judged; alert on drift threshold | 1501 | ai-engineer | parked |
| DOCBOT-3408 | Feedback loop: thumbs-down to triage to gold-set candidate | 3 | P2 | 2 | Thumbs-down creates triage item; accepted items exported as gold candidates | none | ai-engineer | parked |
| DOCBOT-3409 | Provenance panel per answer (sources, model, prompt version, run id) | 3 | P2 | 2 | Panel on every answer; Zod-validated payload | 1510 | frontend | parked |
| DOCBOT-3410 | Customer-facing AI use and model card | 2 | P2 | 2 | Doc lists models, data use, limits, eval method; no NC-dataset numbers | 3202 | PM | parked |

### 7.9 EPIC-35: Robustness and Benchmark Test Program (26 stories, 109 pts, track E)

RB-n = 3500+n. Roadmap placeholders merged: old 3501 -> 3501; old 3502 -> 3503, 3505, 3506; old 3503 -> 3526 (NEW, public numbers only from XBRL synthetic + commercial-OK sets); old 3504 -> 3508, 3521, 3523; old 3505 -> 3518. FinanceBench, FinDER, FaithEval internal only. 3501, 3510, 3512, 3513 scheduled in S1 (decision 6).

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3501 | RB-01 Dataset fetch + cache script, licence manifest (DATA_LICENSES.md), git-ignored cache | 3 | P0 | 1 | Script fetches pinned versions; manifest flags FinanceBench/FinDER/FaithEval internal-only (never training, few-shot or public claims); nothing committed but code + manifest | none | ai-engineer | S1 |
| DOCBOT-3502 | RB-02 XBRL tie-out generator v1 (locate-once, SQLite books, 8 mismatch types, seeded) | 8 | P0 | 2 | Generator emits TieoutQuestion cases from companyfacts + 10-K; >= 40% controls; no company PDFs committed | 3501 | ai-engineer | parked |
| DOCBOT-3503 | RB-03 Wire generator into tieout_eval.py (--source xbrl, --limit), per-mismatch report | 3 | P0 | 2 | Existing 12 questions unchanged; per-type pass rates printed | 3502 | ai-engineer | parked |
| DOCBOT-3504 | RB-04 XBRL detector-only DiscrepancyCases in CI (200 cases, zero LLM) | 2 | P0 | 2 | 200 cases run in CI under 1 min; 0 FP on controls | 3502 | ai-engineer | parked |
| DOCBOT-3505 | RB-05 FinanceBench page-level retrieval + citation eval (internal only) | 5 | P0 | 2 | 150 Q, local embeddings, CI subset + nightly live; results never published (CC-BY-NC) | 3501 | ai-engineer | parked |
| DOCBOT-3506 | RB-06 BIRD Mini-Dev SQL harness (SQLite + PostgreSQL), EX by difficulty | 5 | P0 | 2 | financial DB gate >= 0.50 EX; data fetched at test time, not committed (CC BY-SA) | 3501 | ai-engineer | parked |
| DOCBOT-3507 | RB-07 TrustSQL abstention eval (feasible/infeasible, penalty score) | 3 | P0 | 2 | Correct refusal rate reported; target >= 0.80 | 3506 | ai-engineer | parked |
| DOCBOT-3508 | RB-08 CSV robustness: Pollock subset in CI + synthetic QBO/Xero export generator | 5 | P0 | 2 | Pollock no-crash 100%; header detection >= 0.90; QBO-shaped exports from XBRL facts | 3502 | ai-engineer | parked |
| DOCBOT-3509 | RB-09 Adversarial suite v1: doc/CSV/DB-cell injection + canary judge + tie-out unchanged check | 8 | P0 | 2 | Canary absent from all outputs; injected 'do not flag' cannot change score_run result | 3503 | ai-engineer | parked |
| DOCBOT-3510 | RB-10 SQL-through-NL attack set: validator unit tests + DB checksum + cross-tenant read | 3 | P0 | 1 | sql_validator rejects 100% of side-effect cases offline in CI; checksum unchanged end to end | none | senior-dev | S1 |
| DOCBOT-3511 | RB-11 Sandbox exfiltration suite (network, env secrets, pip typosquat, resource abuse) | 5 | P0 | 2 | With 2812 on, every egress case fails; dummy secrets never in stdout/artifacts; lifetime + cost ceiling enforced | 2812 | senior-dev | parked |
| DOCBOT-3512 | RB-12 PII eval: Gretel finance subset + must-not-mask set (pins the 10-digit bug class) | 3 | P0 | 1 | 0 false masks on CIK, accession, account numbers, raw XBRL values, '$4523000000'; phone/ssn/cc/email recall >= 0.90 | 3501 | ai-engineer | S1 |
| DOCBOT-3513 | RB-13 prompt_guard detector eval on deepset/prompt-injections + finance hard negatives (CI) | 2 | P0 | 1 | Recall >= 0.80; FP rate on benign financial text < 2%; runs offline in CI | 3501 | ai-engineer | S1 |
| DOCBOT-3514 | RB-14 Offline faithfulness scorer: HHEM-2.1-Open nightly, calibrated on RAGTruth | 3 | P1 | 2 | >= 0.90 of answers score consistent; zero LLM spend | 3505 | ai-engineer | parked |
| DOCBOT-3515 | RB-15 Abstention/conflict eval: FaithEval unanswerable + inconsistent (internal only) | 3 | P1 | 2 | Refusal >= 0.80; conflict mention >= 0.70; never cited publicly | 3505 | ai-engineer | parked |
| DOCBOT-3516 | RB-16 Hybrid numeric eval: TAT-QA + MultiHiertt with scale/sign scoring | 5 | P1 | 3 | TAT-QA EM >= 0.60 first gate; attribution in DATA_LICENSES.md | 3501 | ai-engineer | parked |
| DOCBOT-3517 | RB-17 Long-doc + multi-turn: DocFinQA long 10-Ks + ConvFinQA via rephrase memory | 5 | P1 | 3 | Subsets frozen by ID; results by turn depth | 3505 | ai-engineer | parked |
| DOCBOT-3518 | RB-18 Tiered CI wiring: T0 ci.yml, T1 nightly with $ cap + spend print, T2 weekly, frozen subsets | 5 | P0 | 2 | T0 adds < 5 min; T1 aborts at $3; suites not yet built plug in later without workflow edits | 3503, 3504, 3510, 3511, 3512, 3513 | devops | parked |
| DOCBOT-3519 | RB-19 garak + spikee pre-release red-team runbook vs staging; OWASP 2025 coverage matrix | 3 | P1 | 2 | Runbook executed once; matrix covers LLM01-LLM10 | 3509 | ai-engineer | parked |
| DOCBOT-3520 | RB-20 Autopilot agency suite (AgentDojo-style DocBot env + InjecAgent data-stealing) | 8 | P1 | 3 | Data-stealing success 0; tool calls within step budget | 3509 | ai-engineer | parked |
| DOCBOT-3521 | RB-21 Load/chaos: Locust SSE (25/100 concurrent), 300-page 10-K, 1M-row CSV, scanned 10-K set | 5 | P1 | 2 | 25 sessions: TTFT p95 < 4 s, errors < 1% (env stated); absorbs roadmap 3504 scanned-PDF perturbations | 3501 | devops | parked |
| DOCBOT-3522 | RB-22 XBRL generator v2: segment reallocation via dimensional facts, multi-page values | 5 | P1 | 3 | Segment cases flag A and B, not the total | 3502 | ai-engineer | parked |
| DOCBOT-3523 | RB-23 Spider 2.0-Lite SQLite (135) large-schema eval for table-selector caps | 3 | P1 | 3 | EX reported; schema-variant perturbations from roadmap 3504 covered | 3506 | ai-engineer | parked |
| DOCBOT-3524 | RB-24 Non-English tie-out pilot on 10 ESEF filings (internal only, after licence check) | 3 | P1 | 3 | Licence confirmed first, else dropped | 3502 | ai-engineer | parked |
| DOCBOT-3525 | RB-25 Eval results dashboard + weekly trend report (per component, per type, spend) | 3 | P1 | 3 | Weekly report auto-posted; spend per run shown | 3518 | ai-engineer | parked |
| DOCBOT-3526 | NEW: Baseline run + public method page (XBRL synthetic + commercial-OK sets only) | 3 | P1 | 2 | Page states date, model, N, set; no FinanceBench/FinDER/FaithEval numbers; 4011 claim entry added | 3503, 4011 | ai-engineer | parked |

### 7.10 EPIC-36: Workpaper Review and Sign-off (11 stories, 39 pts, track R)

3611 (NEW) is the UI for 3601-3604.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3601 | Workpaper object: saved tie-out with input hashes, query, snapshot time, outputs, lineage | 5 | P1 | 2 | Workspace-scoped; reopening shows identical outputs; tests | 2563, 2014 | senior-dev | parked |
| DOCBOT-3611 | NEW: Workpaper UI (list, detail, review notes, sign-off actions) | 5 | P1 | 2 | List + detail pages with Zod schemas; e2e test of open-to-approve flow | 3601 | frontend | parked |
| DOCBOT-3602 | Workpaper versions and diff between runs | 3 | P1 | 2 | Every re-run creates a version; diff highlights changed figures | 3601 | senior-dev | parked |
| DOCBOT-3603 | Line-level review notes with open/resolved status | 3 | P1 | 2 | Notes per line; approval blocked while any note is open | 3601 | senior-dev | parked |
| DOCBOT-3604 | Preparer/reviewer sign-off with dual control; approved workpaper locked | 5 | P0 | 2 | Preparer cannot approve own work; approved version immutable; audit event | 3001, 3603, 2570 | senior-dev | parked |
| DOCBOT-3605 | Client-facing export only after approval, stamped with approver + version | 2 | P0 | 2 | Export of unapproved workpaper refused; stamp on every page | 3604 | senior-dev | parked |
| DOCBOT-3606 | Roll-forward: re-run last period's workpaper on new period data | 5 | P1 | 2 | New version with period diff in one action | 3602 | senior-dev | parked |
| DOCBOT-3607 | Evidence binder export (citations, sign-off trail, audit excerpt) | 3 | P2 | 2 | Single PDF binder; extends 2014 | 3605 | senior-dev | parked |
| DOCBOT-3608 | Email notifications for review requests and approvals | 2 | P2 | 2 | Resend emails on request/approve; unsubscribe link | 3604 | senior-dev | parked |
| DOCBOT-3609 | Expiring read-only client share link with watermark | 3 | P3 | 3 | Link expires; watermark shows viewer + time | 3605 | senior-dev | parked |
| DOCBOT-3610 | Review queue view per workspace | 3 | P2 | 3 | Queue filters by status and assignee | 3603 | frontend | parked |

### 7.11 EPIC-37: Connectors: Accounting, Warehouse, Content (12 stories, 65 pts, track R)

2577 (existing range, S1) covers CSV intake; connectors stay parked. 3712 (NEW) starts app review when pulled.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3712 | NEW: Intuit + Xero developer app registration and review submission (long lead) | 2 | P1 | 1 | Both apps registered with read-only scopes; review requests submitted; dates logged | none | founder | parked |
| DOCBOT-3703 | Normalized accounting schema (accounts, journal lines, periods) | 5 | P1 | 2 | Same tie-out SQL runs on QBO, Xero and CSV sources; tests | 2532 | senior-dev | parked |
| DOCBOT-3701 | QuickBooks Online read-only connector (TB, GL, P&L, BS) | 8 | P1 | 2 | OAuth2 read-only; reports land in 3703 schema; sandbox-company integration test | 3703, 3712 | senior-dev | parked |
| DOCBOT-3702 | Xero read-only connector (reports API); certification | 8 | P1 | 2 | OAuth2 read-only; reports land in 3703 schema; certification checklist passed | 3703, 3712 | senior-dev | parked |
| DOCBOT-3704 | Snowflake connector (key-pair, read-only role, query tag) | 5 | P2 | 2 | Rejects a role with write grants; 3-layer read-only rule holds | none | senior-dev | parked |
| DOCBOT-3705 | BigQuery connector (service account, max-bytes-billed guard) | 5 | P2 | 2 | Query over byte cap refused before run | none | senior-dev | parked |
| DOCBOT-3706 | SharePoint / OneDrive document source (Graph, folder sync, access check) | 8 | P2 | 2 | Folder sync into workspace; user without file access cannot retrieve it | 2563 | senior-dev | parked |
| DOCBOT-3707 | Google Drive document source | 5 | P2 | 2 | Folder sync with access check | 2563 | senior-dev | parked |
| DOCBOT-3708 | Connector health page (last sync, errors, reauth prompts) | 3 | P2 | 2 | Page per connector; reauth CTA on expired token | 3701 | frontend | parked |
| DOCBOT-3709 | Warehouse cost guard (estimate before run, per-org limits) | 3 | P2 | 2 | Estimate shown; over-limit query refused | 3704, 3705 | senior-dev | parked |
| DOCBOT-3710 | NetSuite read-only (token auth, SuiteQL) | 8 | P3 | 3 | Read-only SuiteQL into 3703 schema | 3703 | senior-dev | parked |
| DOCBOT-3711 | Databricks SQL warehouse (OAuth M2M, Unity Catalog read-only) | 5 | P3 | 3 | Read-only enforced; 7-step pipeline reused | none | senior-dev | parked |

### 7.12 EPIC-38: Admin Console and Customer Observability (8 stories, 34 pts, track E)

Builds on `audit_service.py`, 2570, 1807, 1811, 1530.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3801 | Org admin console: members, workspaces, roles, SSO, policies | 8 | P1 | 2 | One console covers all five; Zod schemas; e2e | 3001, 3004, 2572 | frontend | parked |
| DOCBOT-3802 | Usage and credit dashboard per workspace and member | 3 | P2 | 2 | Matches ledger totals (1801) | 1811 | frontend | parked |
| DOCBOT-3803 | Audit log viewer with search and filters | 3 | P1 | 2 | Filter by workspace, actor, event, date; export CSV | 2570 | frontend | parked |
| DOCBOT-3804 | Audit hash chain + WORM archive to S3 Object Lock | 5 | P1 | 2 | Each row hashes previous; verifier script; daily archive in Object Lock | 2524 | senior-dev | parked |
| DOCBOT-3805 | SIEM export: audit stream to customer S3 or signed webhook | 5 | P2 | 3 | Events delivered within 5 min; signature verifiable | 3804 | senior-dev | parked |
| DOCBOT-3806 | Admin view of LLM data flow (provider, model, region per request) | 2 | P2 | 2 | Reads 3106 records | 3106 | frontend | parked |
| DOCBOT-3807 | Support access: JIT, customer-approved, time-boxed, logged | 5 | P1 | 2 | No access without approval; auto-expiry; audit event | 3804 | senior-dev | parked |
| DOCBOT-3808 | Admin alerts: new member, SSO change, export spike | 3 | P3 | 3 | Email alert per event type | 3801 | senior-dev | parked |

### 7.13 EPIC-39: API, SDK and MCP Server (7 stories, 30 pts, track E)

Builds on 1530 and 2907.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-3901 | Public API v1 contract: OpenAPI, versioning, deprecation, per-key rate limits | 5 | P1 | 2 | /v1 prefix; spec published; 429 on limit | 1530, 2907 | senior-dev | parked |
| DOCBOT-3902 | Tie-out run as async API job with polling | 5 | P1 | 2 | Job result equals UI result; scoped to key | 3901, 3305 | senior-dev | parked |
| DOCBOT-3903 | Signed webhooks: run.completed, discrepancy.found, workpaper.approved | 3 | P2 | 2 | HMAC signed; retries with backoff | 3902 | senior-dev | parked |
| DOCBOT-3904 | MCP server: read-only tools, OAuth, workspace scoping, audited | 8 | P1 | 2 | 4 tools work from an MCP client; every call audited | 3901, 3902 | ai-engineer | parked |
| DOCBOT-3905 | Developer docs site and quickstart | 3 | P2 | 2 | Quickstart runs a tie-out in < 10 min | 3901 | PM | parked |
| DOCBOT-3906 | Python SDK | 3 | P3 | 3 | Published package; examples | 3901 | senior-dev | parked |
| DOCBOT-3907 | TypeScript SDK | 3 | P3 | 3 | Published package; Zod types | 3901 | frontend | parked |

### 7.14 EPIC-40: Enterprise GTM Enablement and Commercial Ops (12 stories, 28 pts, track R)

4001 split into draft (4001) + v2 (4012, NEW). 4004 no longer depends on 1809.

| ID | Title | Pts | Pri | Phase | Acceptance criteria | Deps | Lane | Sprint |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-4001 | Security review kit draft: SIG Lite + CAIQ answers, architecture + data-flow diagrams | 3 | P1 | 1 | SIG Lite answerable in < 1 day; no SOC 2 claim; reuses 2531 and 2608 | 2531, 2608 | PM | parked |
| DOCBOT-4012 | NEW: Security review kit v2 with pen test summary + SOC 2 report (split from 4001) | 2 | P1 | 2 | Kit updated; SOC 2 wording unlocked only by 2712 | 2813, 2712 | PM | parked |
| DOCBOT-4002 | MSA, SLA exhibit and enterprise order form (lawyer review) | 3 | P0 | 1 | Lawyer-reviewed templates in data room; consistent with DPA (2531) | 2549 | founder | parked |
| DOCBOT-4003 | Packaging: Firm vs pooled vs BYOC feature matrix + enforcing flags | 3 | P1 | 1 | Matrix published internally; plan flags enforced server-side with tests | 2569 | senior-dev | parked |
| DOCBOT-4004 | Annual invoiced contracts: Stripe invoicing, PO numbers, net-30, commit tracking | 3 | P1 | 2 | Test invoice with PO and net-30 paid in Stripe test mode | 1810 | senior-dev | parked |
| DOCBOT-4005 | Paid pilot playbook: 30 days, written success criteria, exit report | 1 | P1 | 1 | Template used on the first pooled pilot | none | PM | parked |
| DOCBOT-4006 | ROI calculator from design-partner data | 2 | P2 | 2 | Inputs from 2553 case studies; numbers sourced | 2553 | PM | parked |
| DOCBOT-4007 | QuickBooks App Store and Xero App Store listings | 2 | P2 | 2 | Both listings live | 3701, 3702 | PM | parked |
| DOCBOT-4008 | Referral program for CAS and advisory networks | 2 | P2 | 2 | Terms + tracking link per referrer | none | PM | parked |
| DOCBOT-4009 | Procurement pack: W-9, insurance certificates, vendor forms, accessibility statement | 1 | P1 | 2 | Pack in data room | 2716 | founder | parked |
| DOCBOT-4010 | Enterprise demo environment with synthetic multi-entity data | 5 | P2 | 2 | Reset script restores demo in < 5 min | 2562 | senior-dev | parked |
| DOCBOT-4011 | Claims governance: every public claim mapped to evidence | 1 | P0 | 1 | Claims register; SOC 2 wording locked until 2712; NC-dataset numbers banned | none | PM | parked |

---

## 8. Referenced existing tickets (status from tracker)

| ID | Title | Pts | Pri | Status | Planned phase | Sprint | Lane | Track |
|---|---|---|---|---|---|---|---|---|
| DOCBOT-2601 | Positioning one-pager | 1 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2602 | Pitch deck v1 to v2 | 3 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2603 | Demo script + backup video | 2 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2604 | Prod golden-path hardening + freeze | 2 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | senior-dev | EPIC-26 Pre-seed |
| DOCBOT-2605 | Data room lite | 2 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2606 | Investor Q&A prep | 1 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2607 | Raise ask + use of funds | 1 | P0 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2608 | Deployment-options + data-flow page | 1 | P1 | In progress (EPIC-26, due by 12-01) | 0 | - | PM | EPIC-26 Pre-seed |
| DOCBOT-2609 | Private model mode (OpenAI-compatible endpoint) | 5 | P1 | Code moved after 12-01 (2026-11-17) | 1 | S4 | senior-dev | EPIC-26 Pre-seed |
| DOCBOT-2610 | Investor list + follow-up | 1 | P1 | In progress (EPIC-26, due by 12-01) | 0 | - | founder | EPIC-26 Pre-seed |
| DOCBOT-2501 | PR #86 pre-merge review | 2 | P1 | Open (Phase 0, Wave A) | 0 | - | devops | EPIC-25 AWS |
| DOCBOT-2502 | Fast-forward dev to main | 1 | P1 | Open (Phase 0, Wave A) | 0 | - | founder | EPIC-25 AWS |
| DOCBOT-2503 | Merge #86, account, credits, bootstrap | 2 | P1 | Open (Phase 0, Wave A) | 0 | - | founder | EPIC-25 AWS |
| DOCBOT-2504 | Dev apply, ACM, api-dev, dev secrets | 3 | P1 | Open (Phase 0, Wave A) | 0 | - | founder | EPIC-25 AWS |
| DOCBOT-2505 | First deploy + smoke | 3 | P1 | App readiness merged (PR #105); smoke PR #112 open | 0 | - | devops | EPIC-25 AWS |
| DOCBOT-2506 | Vercel Preview env for dev | 1 | P1 | Open (Phase 0, Wave A) | 0 | - | founder | EPIC-25 AWS |
| DOCBOT-2507 | Dev scale-to-zero + cost evidence | 2 | P1 | PR #115 open | 0 | - | devops | EPIC-25 AWS |
| DOCBOT-2508 | Shared module + envs/prod | 8 | P1 | PR #106 open (merge by 11-30, apply after) | 0 | - | devops | EPIC-25 AWS |
| DOCBOT-2576 | Founder-only: prod apply, prod secrets, aws-prod env | 3 | P1 | Open (Wave A, apply after 12-01) | 1 | S1 | founder | EPIC-25 AWS |
| DOCBOT-2511 | Alarms and notifications | 3 | P1 | PR #109 open | 0 | - | devops | EPIC-25 AWS |
| DOCBOT-2512 | Promote-by-digest dev to prod | 3 | P1 | PR #113 open | 0 | - | devops | EPIC-25 AWS |
| DOCBOT-2513 | 7-day prod-env soak + cost check | 2 | P1 | Open | 1 | S1 | devops | EPIC-25 AWS |
| DOCBOT-2578 | NEW: DB_ENCRYPTION_KEY MultiFernet dual-key rotation (needed for 2509) | 3 | P1 | New (2026-11-17) | 1 | S1 | senior-dev | EPIC-25 AWS |
| DOCBOT-2579 | NEW: Merge dev into main before prod apply (needed for 2576) | 1 | P1 | New (2026-11-17) | 1 | S1 | founder | EPIC-25 AWS |
| DOCBOT-2580 | NEW: Plane auto-sync rule and per-ticket state command | 1 | P2 | In progress (2026-11-17) | 0 | - | PM | EPIC-26 Pre-seed |
| DOCBOT-2581 | NEW: Waitlist instead of payments | 3 | P1 | In progress (2026-11-17) | 0 | - | senior-dev | EPIC-26 Pre-seed |
| DOCBOT-2582 | NEW: Remove "enterprise" claims from landing page | 1 | P1 | In progress (2026-11-17) | 0 | - | frontend | EPIC-26 Pre-seed |
| DOCBOT-2583 | NEW: Demo chat hangs when Groq rate-limits | 3 | P0 | In progress (2026-11-20) | 0 | - | senior-dev | EPIC-26 Pre-seed |
| DOCBOT-2584 | NEW: Run gpt-oss on Amazon Bedrock (AWS credits) behind LLM_PRIMARY | 3 | P1 | Done (2026-11-22, PR #134) | 0 | - | senior-dev | EPIC-26 Pre-seed |
| DOCBOT-2585 | NEW: All LLM calls on Bedrock (fallback + LangExtract) and swappable open-weight models | 3 | P1 | In progress (2026-11-22) | 0 | - | senior-dev | EPIC-26 Pre-seed |
| DOCBOT-2586 | NEW: Move ticket tracking from Plane to self-hosted OpenProject (pm.docbot.space) | 2 | P2 | In progress (2026-11-22) | 0 | - | PM | EPIC-26 Pre-seed |
| DOCBOT-2509 | Secrets rotation | 5 | P1 | Open (Wave B) | 1 | S2 | devops | EPIC-25 AWS |
| DOCBOT-2510 | Prod restore drill | 3 | P1 | Open (Wave B) | 1 | S2 | devops | EPIC-25 AWS |
| DOCBOT-2514 | Cutover runbook + go/no-go | 3 | P1 | Write-freeze code merged (PR #111); runbook open | 1 | S2 | devops | EPIC-25 AWS |
| DOCBOT-2515 | Postgres rehearsal Railway to prod RDS | 5 | P1 | Open (Wave B) | 1 | S3 | senior-dev | EPIC-25 AWS |
| DOCBOT-2516 | Chroma /data ADR + rehearsal | 5 | P1 | Open (Wave B) | 1 | S3 | senior-dev | EPIC-25 AWS |
| DOCBOT-2517 | DNS cutover plan, stable api hostname | 3 | P1 | Open (Wave B; start early) | 1 | S2 | founder | EPIC-25 AWS |
| DOCBOT-2518 | Rollback plan + drill | 3 | P1 | Open (Wave B) | 1 | S3 | devops | EPIC-25 AWS |
| DOCBOT-2519 | Founder-only: prod cutover | 3 | P1 | Open (Wave B; moved to mid-January, after 3 DPs stable) | 1 | S4 | founder | EPIC-25 AWS |
| DOCBOT-2520 | Railway sunset | 2 | P1 | Open (Wave C) | 1 | S5 | founder | EPIC-25 AWS |
| DOCBOT-2521 | Private subnets + NAT ADR | 5 | P2 | Open (Wave C) | 2 | S8 | devops | EPIC-25 AWS |
| DOCBOT-2523 | KMS CMKs for remaining services | 3 | P2 | Open (Wave C) | 1 | S5 | devops | EPIC-25 AWS |
| DOCBOT-2522 | WAF on the ALB | 3 | P3 | Open (Wave C) | unscheduled | - | devops | EPIC-25 AWS |
| DOCBOT-2524 | CloudTrail, GuardDuty, Config | 3 | P3 | Open (Wave C) | 1 | S5 | devops | EPIC-25 AWS |
| DOCBOT-2525 | IaC security scanning in CI | 2 | P3 | Open (Wave C) | unscheduled | - | devops | EPIC-25 AWS |
| DOCBOT-2526 | SOC 2 groundwork: control map + human access | 3 | P3 | Open (Wave C; pulled to Phase 1) | 1 | S6 | founder | EPIC-25 AWS |
| DOCBOT-2527 | Extract modules (PLAT-0405/0406) | 5 | P3 | Open (Wave C, parked) | unscheduled | - | devops | EPIC-25 AWS |
| DOCBOT-2528 | Tag engine copy base | 2 | P3 | Open (Wave C, parked) | unscheduled | - | devops | EPIC-25 AWS |
| DOCBOT-2529 | Re-point PLAT-0411 import | 3 | P3 | Open (Wave C, parked) | unscheduled | - | devops | EPIC-25 AWS |
| DOCBOT-2530 | Account + state migration plan | 3 | P3 | Open (Wave C, parked) | unscheduled | - | founder | EPIC-25 AWS |
| DOCBOT-2531 | Trust package (security page, DPA, questionnaire) | 3 | P0 | Open (pulled into EPIC-26, due 11-25) | 0 | - | PM | EPIC-25 GTM |
| DOCBOT-2532 | Proof-run playbook (QBO/Xero CSV + PDF) | 5 | P0 | Open; pulled before Dec 1 (2026-11-17) | 0 | S0 | PM | EPIC-25 GTM |
| DOCBOT-2533 | Design-partner offer + order form | 2 | P0 | Open (pulled into EPIC-26, due 11-24) | 0 | - | founder | EPIC-25 GTM |
| DOCBOT-2534 | Pitch/landing copy alignment | 2 | P1 | Open | 1 | S6 | senior-dev | EPIC-25 GTM |
| DOCBOT-2535 | Launch content + outreach drafts + funnel tracker | 3 | P1 | Open | 1 | S7 | PM | EPIC-25 GTM |
| DOCBOT-2536 | Landing SEO foundation | 2 | - | PR #99 open (held; pulled into EPIC-26) | 0 | - | senior-dev | EPIC-25 SEO |
| DOCBOT-2537 | Landing copy realignment + /landing dedupe | 3 | - | To do (pulled into EPIC-26) | 0 | - | senior-dev | EPIC-25 SEO |
| DOCBOT-2538 | FAQ + FAQPage JSON-LD + llms.txt | 2 | - | To do | 1 | S6 | senior-dev | EPIC-25 SEO |
| DOCBOT-2539 | Content pages (/tie-out, /fractional-cfo, /compare/*) | 5 | - | To do | 2 | S8 | senior-dev | EPIC-25 SEO |
| DOCBOT-2540 | Search Console/Bing + AI-citation tracking | 2 | - | To do | 2 | S8 | founder | EPIC-25 SEO |
| DOCBOT-2541 | Tie-out eval harness | 3 | P0 | Merged (PR #101) | done | - | ai-engineer | EPIC-25 AI proof |
| DOCBOT-2542 | Fix flaky segment tie-out question | 2 | P0 | Merged (PR #102) | done | - | ai-engineer | EPIC-25 AI proof |
| DOCBOT-2543 | Founder live tie-out eval run | 1 | P0 | Open (pulled into EPIC-26, due 11-18) | 0 | - | founder | EPIC-25 AI proof |
| DOCBOT-2544 | Scheduled live tie-out eval with spend cap | 2 | P1 | Open | 1 | S2 | ai-engineer | EPIC-25 AI proof |
| DOCBOT-2545 | Gold set v2 (public 10-K + export shapes) | 3 | P1 | Open | 1 | S3 | ai-engineer | EPIC-25 AI proof |
| DOCBOT-2546 | ICP prospect list | 2 | P0 | Open (pulled into EPIC-26) | 0 | - | founder | EPIC-25 Sales ops |
| DOCBOT-2547 | Outreach sequence drafts | 1 | P0 | Open (pulled into EPIC-26) | 0 | - | PM | EPIC-25 Sales ops |
| DOCBOT-2548 | Discovery call script | 1 | P0 | Open (pulled into EPIC-26) | 0 | - | PM | EPIC-25 Sales ops |
| DOCBOT-2549 | Design-partner agreement template | 1 | P0 | Open | 1 | S1 | founder | EPIC-25 Sales ops |
| DOCBOT-2550 | Tie-out demo script on prospect docs | 2 | P0 | Open; pulled before Dec 1 (2026-11-17) | 0 | S0 | PM | EPIC-25 Sales ops |
| DOCBOT-2551 | Weekly gate tracker | 1 | P0 | Starts now (2026-11-17) | 0 | S0 | PM | EPIC-25 Sales ops |
| DOCBOT-2552 | Enterprise target list + pooled one-pager | 2 | P1 | Open | 1 | S6 | founder | EPIC-25 Sales ops |
| DOCBOT-2553 | Design-partner case studies (3) | 2 | P1 | Open | 1 | S6 | PM | EPIC-25 Sales ops |
| DOCBOT-2577 | NEW: QBO/Xero CSV export intake for DP onboarding (CSV only, no connector) | 5 | P0 | New (2026-11-17) | 1 | S1 | senior-dev | EPIC-25 Sales ops |
| DOCBOT-2561 | Firm per-client workspace gap analysis | 1 | - | Draft done | done | - | PM | EPIC-25 Firm workspace |
| DOCBOT-2562 | Client workspace data model | 5 | - | To do | 1 | S4 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2563 | Workspace scoping in CallerScope | 5 | - | To do | 1 | S5 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2564 | Workspace switcher + CRUD | 3 | - | To do | 1 | S5 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2565 | Org invites (was 1709) | 5 | - | To do | 1 | S4 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2566 | Multi-org membership | 5 | - | To do | 1 | S5 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2567 | Seats + offboarding | 3 | - | To do | 1 | S5 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2568 | Per-workspace access grants | 3 | - | To do | 1 | S6 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2569 | Firm billing per client workspace | 5 | - | To do | 1 | S6 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2570 | Audit log per workspace | 3 | - | To do | 1 | S6 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2571 | Microsoft sign-in (Entra OIDC) | 3 | - | To do | 1 | S7 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2572 | Per-org SAML + domain enforcement | 5 | - | Later | 2 | S8 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2573 | BYOK 'contact us' (BYOK = customer LLM key, EPIC-19) | 1 | - | To do | 1 | S5 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2574 | Priority support | 1 | - | To do | 1 | S6 | PM | EPIC-25 Firm workspace |
| DOCBOT-2575 | Client workspace offboarding | 3 | - | To do | 1 | S6 | senior-dev | EPIC-25 Firm workspace |
| DOCBOT-2401 | Two-org isolation sweep in CI | 5 | P0 | Not on main (docs/DOCBOT-2400 branch) | 1 | S3 | senior-dev | EPIC-24 Stage A |
| DOCBOT-2402 | org_id NOT NULL for user-owned rows | 3 | P0 | Not on main | 1 | S4 | senior-dev | EPIC-24 Stage A |
| DOCBOT-2403 | Org-namespaced Chroma collections + purge | 3 | P0 | Not on main | 1 | S4 | senior-dev | EPIC-24 Stage A |
| DOCBOT-2404 | Per-org quotas | 3 | P1 | Not on main | 1 | S7 | senior-dev | EPIC-24 Stage A |
| DOCBOT-2405 | Engine pool keyed (org, conn); cost totals in Postgres | 3 | P1 | Not on main | 1 | S7 | senior-dev | EPIC-24 Stage A |
| DOCBOT-2406 | Single-replica assertion + scheduler leader lock | 1 | P1 | Not on main | 1 | S4 | senior-dev | EPIC-24 Stage A |
| DOCBOT-1613 | Prod smoke regression (G1) | 2 | P0 | Open | 1 | S2 | ai-engineer | EPIC-16 remainder |
| DOCBOT-1614 | Security review of new surfaces (G1) | 2 | P0 | Open | 1 | S2 | PM | EPIC-16 remainder |
| DOCBOT-1626 | SAML replay protection | 1 | P1 | Open | 1 | S3 | senior-dev | EPIC-16 remainder |
| DOCBOT-1627 | Shopify webhook fix | 1 | P1 | Open | 1 | S5 | senior-dev | EPIC-16 remainder |
| DOCBOT-1628 | Stop echoing exceptions to clients | 1 | P1 | Open | 1 | S3 | senior-dev | EPIC-16 remainder |
| DOCBOT-1703 | Password reset | 2 | P1 | Open | 1 | S6 | senior-dev | EPIC-16 remainder |
| DOCBOT-1708 | Account deletion + export | 2 | P1 | Open | 1 | S5 | senior-dev | EPIC-16 remainder |
| DOCBOT-1901 | Store org LLM keys | 3 | P1 | Open, deferred | unscheduled | - | senior-dev | EPIC-19 BYOK (LLM keys) |
| DOCBOT-1902 | Per-request key resolution | 5 | P1 | Open, deferred | unscheduled | - | senior-dev | EPIC-19 BYOK (LLM keys) |
| DOCBOT-1903 | OpenAI-compatible provider (allowlisted) | 3 | P2 | Open, deferred | unscheduled | - | senior-dev | EPIC-19 BYOK (LLM keys) |
| DOCBOT-1904 | BYOK settings UI | 2 | P1 | Open, deferred | unscheduled | - | senior-dev | EPIC-19 BYOK (LLM keys) |
| DOCBOT-1905 | Key redaction everywhere | 2 | P1 | Open, deferred | unscheduled | - | senior-dev | EPIC-19 BYOK (LLM keys) |
| DOCBOT-1906 | Formatting regression per BYOK model | 3 | P2 | Open, deferred | unscheduled | - | senior-dev | EPIC-19 BYOK (LLM keys) |
| DOCBOT-2312 | Correct page citations | 3 | P1 | Open | 1 | S4 | ai-engineer | EPIC-23 open bugs (referenced) |
| DOCBOT-2304 | Stop button + idle timeout | 3 | P1 | Open | 2 | S8 | frontend | EPIC-23 open bugs (referenced) |
| DOCBOT-2308 | DB server-side timeouts | 2 | P1 | Open | 2 | S8 | senior-dev | EPIC-23 open bugs (referenced) |
| DOCBOT-2309 | Schema-drift retry actually fires | 1 | P1 | Open | 2 | S8 | senior-dev | EPIC-23 open bugs (referenced) |
| DOCBOT-1810 | Pro subscription + dunning | 3 | P2 | PR #103 open (held for founder review) | 1 | S3 | senior-dev | EPIC-18 billing (referenced) |
| DOCBOT-1809 | Metered overage (Stripe Meters) | 5 | P2 | Open | unscheduled | - | senior-dev | EPIC-18 billing (referenced) |

---

## 9. Critical path and risks

| Chain | Path | Lands |
|---|---|---|
| Design partners | 2532 + 2550 (pre-Dec 1) -> 2577 -> 2549 -> 3 paid DPs -> 2553 case studies | W4 about 12-20; 2553 in S6 |
| Cutover | 2578 -> 2509; 2576 -> 2513; 2510, 2514, 1613, 1614, 2517 -> 2515 + 2516 -> 2518 -> 3 DPs stable >= 2 weeks -> 2519 -> 2520 (+14 days) -> 2523/2524 -> 2526 | 2519 S4 (mid-January), 2520 S5, 2526 S6 |
| Firm tier | 2401-2403 -> 2562 -> 2565 -> 2563 -> 2568 -> 2569 (needs 1810) | S6 |

| Risk | Trigger | Response |
|---|---|---|
| Phase 1 gate missed (< $5K MRR or < 6 logos on 02-28) | 2551 weekly | Phase 2 stays revenue-only; parking lot stays parked except written asks; S8 SEO/content items move up. |
| DPs not stable by early January | Fewer than 3 DPs stable 2 weeks by 01-12 | Hold 2519 one sprint (S5); Railway cost runs longer; Wave C and 2526 slip with it. |
| Written SOC 2 ask arrives in Phase 1 | Logged in 2551 | Pull 2701-2705 + 2716 only (long lead); audit itself waits for post-cutover window. |
| 2812 breaks analytics needing a non-template package | Template gap | Add the package to the template, not back to pip egress. |
| 2609 slips past S4 | Senior-dev lane busy | Deck/trust copy says 'next', never 'demoed'. |
| Groq llama-3.3-70b price change ('contact sales') | Invoice | Re-price per-tie-out cost before raising DP volume. |
| Founder overload (calls + AWS) | Founder lane > 8 | Defer 2540, 2552 first; never 2519 or 2576. |

---

## 10. Open founder decisions (default assumed)

| # | Decision | Default assumed |
|---|---|---|
| D1 | One engine or two for BYOC? | One engine with tier flags; 3205 ADR only when a BYOC ask is pulled. |
| D2 | SOC 2 budget and timing | Parked until a written ask; Wave C evidence (2523, 2524, 2526) keeps the clock ready. |
| D3 | Tier 2: ship without Python or require 3203? | Ship without Python (2609 behavior). |
| D4 | First hire: security/devops or sales? | Not in capacity. |
| D5 | Prospects outside the US? | None; 3111 parked. |
| D6 | Turn sandbox egress off in prod (2812)? | Yes, in S1, with statsmodels/xgboost/lightgbm/tabulate baked in. |
| D9 | License FinanceBench for public numbers? | No; public numbers only from XBRL synthetic + commercial-OK sets. |
| D10 | Firm billing per workspace (2569) timing | S6; DPs pay by invoice/Payment Link until then. |
| D11 (new) | Interim gates W8 >= $2.5K MRR or 4 logos, W12 >= $4K MRR (derived from the $5K target) | Use them in 2551 unless changed. |
| D12 (new) | Decisions 1 and 6 overlap: decision 1 keeps only 3512, decision 6 schedules 3510, 3501, 3513, 3512 in S1 | Followed decision 6 (11 pts in S1). If only 3512 was meant, park 3510 and 3513 (frees 5 pts); 3501 stays as 3512's dep. |
| D13 (new) | CLAUDE.md still has the Jira Sync Rule and `scripts/jira_update_status.py` | Not edited here; update CLAUDE.md when Plane is wired. |

