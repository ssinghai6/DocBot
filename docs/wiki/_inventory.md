# Wiki document inventory (2026-11-23)

Every document found for the DocBot wiki, with the import decision. Scope: repo `*.md` outside `node_modules/`, `.venv/`, `.git/` (worktree `ado-sync`, branch `docs/DOCBOT-2589-wiki-architecture`), plus the founder plans folder `~/Desktop/Personal_Projects/DocBot-plans` (`PLANS` below).

Decisions: **include** = current wiki page; **archive** = page under Archive; **exclude** = not imported. Dates: repo files use the last git commit; plans files use file mtime.

Reference facts for the stale check:
- Ticket tracker: Jira until 2026-11-17, Plane until 2026-11-22, OpenProject (pm.docbot.space) since 2026-11-22.
- Hosting: still Railway + Vercel. AWS cutover (DOCBOT-2519) is mid-January 2027 (master backlog S4, 01-13 to 01-26). Older docs say Dec 2-20 or ~2027-04-05.
- Revenue goal (revised 2026-11-17): >= $5K MRR + 6 paying logos by 2027-02-28; $100K ARR run-rate is the stretch.
- Demo is QuickBite (not TechCorp); auth switch is `APP_AUTH_MODE` (not `AUTH_REQUIRED`); prod LLM is gpt-oss (not Llama 3.3).

Counts: 24 include + 12 archive = 36 imported source docs (stories folder counted once), plus 13 pages from the two agents and the Home page. 21 manifest pages are sensitive (20 docs + the Business section). 31 files excluded (30 in PLANS + `archify-input.md`), plus 9 out-of-scope files seen.

## Repo

| Path | Decision | Wiki slug | Reason | Duplicate of | Last modified | Stale? |
|---|---|---|---|---|---|---|
| `README.md` | include | product-readme | Main product doc | - | 2026-10-28 | **Yes**: "686+ tests" badge, Railway-only diagrams, banned "No other tool" claim (line 48), no mention of credits/Stripe/AWS plan |
| `CLAUDE.md` | include | engineering-conventions | Engineering rules + board sync rule | - | 2026-11-23 | Partly: "Current state (2026-03-29)" block and "686+ tests" (line 245) are March-era. OpenProject rule is current |
| `finetune/README.md` | include | finetune-pipeline | Only doc for the fine-tuning tier | - | 2026-10-10 | Minor: strategy defers fine-tuning indefinitely; doc does not say so |
| `tests/eval/README.md` | include | eval-harness | Eval harness doc | - | 2026-11-11 | No |
| `scripts/README-openproject.md` | include | openproject-sync | Live board sync + server notes | - | 2026-11-23 | No |
| `scripts/README-plane.md` | archive | archive-plane-sync | Plane retired 2026-11-22 | superseded by README-openproject.md | 2026-11-17 | **Yes**: describes a retired board |
| `project-tasks/docbot-master-backlog.md` | include | master-backlog | Ticket source of truth | canonical (PLANS copy is older) | 2026-11-22 | Partly: "After merge" rule (line 16) and decision 8 (line 90) still say Plane |
| `project-tasks/docbot-enterprise-roadmap.md` | include | enterprise-roadmap | EPIC-27..40 roadmap | canonical (PLANS copy identical) | 2026-11-17 | Open conflict: EPIC-19 BYOK home vs strategy |
| `project-tasks/docbot-v2-project-tracking.md` | include | epic-history | Only full epic history | canonical (PLANS copy is older, 119 diff lines) | 2026-11-17 | **Yes**: header "Last Updated 2026-03-31", recommends Linear, EPIC-25 says "tracked in Plane", no EPIC-16..24 sections, AWS rows lack status |
| `project-tasks/docbot-multi-agent-architecture-decision.md` | include | adr-multi-agent-autopilot | Decision record | - | 2026-09-29 | No (dated record) |
| `project-tasks/docbot-robustness-test-data.md` | include | robustness-test-data | Test-data research plan | canonical (PLANS copy identical) | 2026-11-17 | Minor: RB-01..25 are now DOCBOT-3501..3525 |
| `project-tasks/manual-regression-checklist.md` | include | manual-regression-checklist | Only manual regression suite | canonical (PLANS copy identical) | 2026-03-29 | **Yes**: `AUTH_REQUIRED` guest mode, Amazon/Shopify, "Production (Railway)"; missing signup, credits, Stripe, TieOutCard, export, QuickBite |
| `project-tasks/yc-10-day-plan.md` | archive | archive-yc-w27-plan | Deadline passed; still only AC source for some EPIC-16/17/20/23 tickets | canonical (PLANS `_archive/yc-w27-10-day-plan.md` lacks section 8) | 2026-11-17 | **Yes**: Pro $49 / 100 credits, "AWS parked" |
| `project-tasks/playbooks/DOCBOT-2532-proof-run.md` | include | proof-run-playbook | Operator runbook | - | 2026-11-17 | No |
| `project-tasks/stories/DOCBOT-*.md` (96 files, 1613..3513) | include (one index page) | user-stories | Already the OpenProject work-package descriptions; one index page, not 96 pages | - | 2026-11-22 | Collision: `DOCBOT-2501.md` is the AWS story, while branch `feature/DOCBOT-2501-audit-hash-chain` reuses the id (roadmap says 3804) |
| `docs/wiki/architecture/01..08-*.md` | include (generated) | architecture-01..08 | ai-engineer agent output | - | in progress | n/a |
| `docs/wiki/architecture/archify-input.md` | exclude | (feeds architecture-00-diagrams) | Diagram spec, not a page | - | in progress | n/a |
| `docs/wiki/api/01..04-*.md` | include (generated) | api-01..04 | senior-developer agent output | - | in progress | n/a |

## Founder plans folder (`PLANS`)

### Imported

| Path | Decision | Wiki slug | Sensitive | Reason | Duplicate of | Last modified | Stale? |
|---|---|---|---|---|---|---|---|
| `03-strategy-pricing/docbot-strategy-2027.md` | include | strategy-2027 | yes | Strategy + pricing source of truth | newer than original on unpushed branch `docs/DOCBOT-2400-strategy-roadmap` | 2026-11-17 | **Yes**: banner says cutover Dec 2-20 (now mid-January); 3,000 free credits vs code default 500 |
| `01-investor-pitch/preseed-14day-plan-dec1.md` | include | preseed-plan-dec1 | yes | Current plan to the Dec 1 call | - | 2026-11-17 | Partly: P1 cutover Dec 2-20; 11-27 freeze vs 11-30 merge. Archive after 12-01 |
| `01-investor-pitch/pitch-deck-main.md` | include | pitch-deck-main | yes | Current deck spec; names investors | - | 2026-11-22 | No |
| `01-investor-pitch/pitch-deck-technical.md` | include | pitch-deck-technical | yes | Technical slides | - | 2026-11-22 | No (verified vs ef23130) |
| `01-investor-pitch/founder-plan/one-pager.md` | include | founder-plan-one-pager | yes | Investor pack | `temp/founder/one-pager.md` (untracked original) | 2026-11-22 | No |
| `01-investor-pitch/founder-plan/summary.md` | include | founder-plan-summary | yes | Investor pack | `temp/founder/summary.md` | 2026-11-22 | No |
| `01-investor-pitch/founder-plan/business-plan.md` | include | founder-plan-business-plan | yes | Investor pack | `temp/founder/business-plan.md` | 2026-11-22 | No |
| `01-investor-pitch/founder-plan/cfo.md` | include | founder-plan-cfo | yes | Unit economics | `temp/founder/cfo.md` | 2026-11-22 | No |
| `02-gtm-revenue/revenue-100k-plan.html` | include (HTML only, convert) | revenue-plan | yes | Revenue plan; no .md exists | - | 2026-11-17 | Check: title/target vs revised $5K MRR goal; W1/W2 gates overtaken by EPIC-26 |
| `02-gtm-revenue/market-gtm-plan.html` | include (HTML only, convert) | market-gtm-plan | yes | Market + GTM; no .md exists | - | 2026-11-11 | **Yes**: calendar from 11-16 slipped; promises per-client workspaces not built |
| `04-aws-migration/aws-migration-exec-plan.html` | include (HTML only, convert) | aws-migration-plan | no | Only AWS migration plan; no .md exists | - | 2026-11-14 | **Yes**: cutover Dec 2-20 (now mid-January); no statuses for merged #105/#111/#114 |
| `04-aws-migration/founder-actions-aws.md` | include | founder-aws-actions | yes | Founder-only AWS steps | - | 2026-11-14 | Partly: answers table empty; step 18 timed to the old cutover |
| `01-investor-pitch/demo-plan.md` | archive | archive-demo-plan | yes | Superseded by DOCBOT-2603 demo script | - | 2026-05-15 | **Yes**: TechCorp, `[DISCREPANCY]`, `AUTH_REQUIRED`, 686+ tests |
| `02-gtm-revenue/founder-ops.md` | archive | archive-founder-ops | yes | Plans README already marks it archived | `temp/founder/ops.md` | 2026-10-28 | **Yes**: Stripe "not merged", no AWS |
| `06-roadmap-backlog/decentralized-docs-idea.md` | archive | archive-decentralized-docs-idea | no | Idea doc; conflicting ids | - | 2026-11-16 | **Yes**: reuses EPIC-25 and DOCBOT-2501..2508 |
| `00-doc-status-audit.md` | archive | archive-doc-status-audit | yes | Snapshot audit; superseded by this inventory | - | 2026-11-17 | Snapshot (its contradictions list is still useful) |
| `_archive/investor-demo-sprint-mar.md` | archive | archive-investor-demo-sprint | yes | Historical | - | 2026-03-26 | Historical |
| `_archive/positioning-aws-gtm-100k-plan.md` | archive | archive-positioning-plan | yes | Executed | - | 2026-11-11 | Historical |
| `_archive/founder-idea.md` | archive | archive-founder-idea | yes | Superseded by strategy | - | 2026-10-28 | Historical |
| `_archive/strategy-roadmap-draft.md` | archive | archive-strategy-draft | yes | Draft of strategy doc (near-duplicate) | ~ strategy-2027 | 2026-10-30 | Historical; candidate to delete |
| `_archive/unit-economics-mar.md` | archive | archive-unit-economics | yes | Superseded by strategy section 3 | - | 2026-03-25 | Historical |
| `_archive/docbot-platform-plan.md` | archive | archive-platform-plan | no | Superseded by `docbot-platform/docs/PLAN.md` | - | 2026-10-22 | Historical (GCP, EPIC-16 at 24 pts) |

### Excluded

| Path | Reason | Duplicate of | Last modified |
|---|---|---|---|
| `06-roadmap-backlog/docbot-v2-project-tracking.md` | Older copy | repo `project-tasks/docbot-v2-project-tracking.md` | 2026-11-17 |
| `06-roadmap-backlog/docbot-master-backlog.md` | Older copy (pre-trim; 969 diff lines) | repo `project-tasks/docbot-master-backlog.md` | 2026-11-17 |
| `06-roadmap-backlog/docbot-enterprise-roadmap.md` | Identical copy | repo `project-tasks/docbot-enterprise-roadmap.md` | 2026-11-17 |
| `05-investor-call-testing/docbot-robustness-test-data.md` | Identical copy | repo `project-tasks/docbot-robustness-test-data.md` | 2026-11-17 |
| `05-investor-call-testing/manual-regression-checklist.md` | Identical copy | repo `project-tasks/manual-regression-checklist.md` | 2026-10-23 |
| `_archive/yc-w27-10-day-plan.md` | Older copy (no section 8) | repo `project-tasks/yc-10-day-plan.md` | 2026-10-24 |
| `README.md` | Folder index; replaced by wiki Home | - | 2026-11-22 |
| `index.html` | Folder index page | - | 2026-11-17 |
| `01-investor-pitch/founder-plan/idea.md` | Founder-skill input; covered by business plan | `temp/founder/idea.md` | 2026-11-22 |
| `01-investor-pitch/founder-plan/numbers.json` | Data file, not a page | `temp/founder/numbers.json` | 2026-11-22 |
| `01-investor-pitch/founder-plan/cfo-sources.md` | Merged into founder-plan-cfo as a Sources section | `temp/founder/cfo-sources.md` | 2026-11-22 |
| `00-doc-status-audit.html` | HTML rendering of .md | `00-doc-status-audit.md` | 2026-11-17 |
| `01-investor-pitch/demo-plan.html` | HTML rendering of .md | `demo-plan.md` | 2026-11-17 |
| `01-investor-pitch/preseed-14day-plan-dec1.html` | HTML rendering of .md | `preseed-14day-plan-dec1.md` | 2026-11-17 |
| `02-gtm-revenue/founder-ops.html` | HTML rendering of .md | `founder-ops.md` | 2026-11-17 |
| `03-strategy-pricing/docbot-strategy-2027.html` | HTML rendering of .md | `docbot-strategy-2027.md` | 2026-11-17 |
| `04-aws-migration/founder-actions-aws.html` | HTML rendering of .md | `founder-actions-aws.md` | 2026-11-17 |
| `05-investor-call-testing/docbot-robustness-test-data.html` | HTML rendering of .md | `docbot-robustness-test-data.md` | 2026-11-17 |
| `05-investor-call-testing/manual-regression-checklist.html` | HTML rendering of .md | `manual-regression-checklist.md` | 2026-11-17 |
| `06-roadmap-backlog/decentralized-docs-idea.html` | HTML rendering of .md | `decentralized-docs-idea.md` | 2026-11-17 |
| `06-roadmap-backlog/docbot-enterprise-roadmap.html` | HTML rendering of .md | `docbot-enterprise-roadmap.md` | 2026-11-17 |
| `06-roadmap-backlog/docbot-master-backlog.html` | HTML rendering of .md | `docbot-master-backlog.md` | 2026-11-17 |
| `06-roadmap-backlog/docbot-v2-project-tracking.html` | HTML rendering of .md | `docbot-v2-project-tracking.md` | 2026-11-17 |
| `_archive/docbot-platform-plan.html` | HTML rendering of .md | `_archive/docbot-platform-plan.md` | 2026-11-17 |
| `_archive/founder-idea.html` | HTML rendering of .md | `_archive/founder-idea.md` | 2026-11-17 |
| `_archive/investor-demo-sprint-mar.html` | HTML rendering of .md | `_archive/investor-demo-sprint-mar.md` | 2026-11-17 |
| `_archive/positioning-aws-gtm-100k-plan.html` | HTML rendering of .md | `_archive/positioning-aws-gtm-100k-plan.md` | 2026-11-17 |
| `_archive/strategy-roadmap-draft.html` | HTML rendering of .md | `_archive/strategy-roadmap-draft.md` | 2026-11-17 |
| `_archive/unit-economics-mar.html` | HTML rendering of .md | `_archive/unit-economics-mar.md` | 2026-11-17 |
| `_archive/yc-w27-10-day-plan.html` | HTML rendering of .md | `_archive/yc-w27-10-day-plan.md` | 2026-11-17 |

### Outside both sources (seen, excluded)

| Path | Reason |
|---|---|
| `~/orca/workspaces/DocBot/temp/founder/*` (8 files: idea, ops, business-plan, cfo, cfo-sources, numbers.json, one-pager, summary) | Untracked originals of the PLANS founder-plan copies; PLANS copies are imported. `ops.md` ~ PLANS `founder-ops.md` |
| `docbot-platform/docs/PLAN.md` (separate repo) | Live platform plan; out of scope for the DocBot wiki, linked from archive-platform-plan |

## Contradictions to resolve (carry-over from the 11-17 audit, re-checked today)

1. **AWS cutover date**: strategy-2027, pre-seed plan and AWS exec plan say Dec 2-20; master backlog and tracker EPIC-26 say mid-January. Pick mid-January and fix the three docs.
2. **Ticket board name**: master backlog and tracker EPIC-25 say Plane; CLAUDE.md and scripts say OpenProject.
3. **DOCBOT-2501 id collision**: AWS story vs hash-chain branch (roadmap id 3804).
4. **EPIC-19 BYOK home**: roadmap/backlog keep it in DocBot; strategy moves it to the platform.
5. **Free credits and packs**: docs say 3,000 free and $20/$50/$150; code defaults 500 and $10/$45/$160 (prod env may override).
6. **Test count**: README and CLAUDE.md say 686+; the 11-17 audit counted ~510 test functions.
