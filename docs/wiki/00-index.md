# DocBot wiki

DocBot is an AI analyst that works across a team's documents (PDFs, SEC filings) and live data (databases, CSVs) at once.
It answers in plain English with charts and citations to both sources, and flags every place where the documents and the data disagree (the tie-out).
Stack: Next.js on Vercel, FastAPI on Railway (AWS cutover planned for mid-January 2027), Groq/Gemini LLMs, E2B sandboxes, PostgreSQL.

## How this wiki is organised

| Section | What is in it |
|---|---|
| [Product](product) | What DocBot does and how to run it |
| [Architecture](architecture) | How the system is built: components, flows, data, AI, security, deployment, frontend |
| [Public API](public-api) | Today's API surface and the plan for a public, key-based API |
| [Planning and Backlog](planning) | Backlog, roadmap, user stories, epic history |
| [Operations and Runbooks](operations) | Engineering rules, ticket board sync, playbooks, AWS migration |
| [Testing and QA](testing) | Eval harness, manual regression, robustness test data |
| [Business (sensitive)](business) | Strategy, pricing, revenue, fundraising. Private |
| [Archive](archive) | Superseded or historical docs, kept for context |

## Pages

### Product
- [DocBot README](product-readme): features, setup, API summary, tech stack. Some numbers are out of date (see banner).

### Architecture
- [Architecture diagrams](architecture-00-diagrams): rendered system, request-flow and deployment diagrams.
- [System Overview](architecture-01-system-overview): components, boundaries and how they connect.
- [Request Flows](architecture-02-request-flows): chat, SQL, CSV, hybrid and Autopilot paths step by step.
- [Data Model](architecture-03-data-model): PostgreSQL tables, vector store, encrypted credential blobs.
- [Endpoint Reference](architecture-04-endpoint-reference): every internal FastAPI route with auth and role.
- [AI / LLM Architecture](architecture-05-ai-llm-architecture): models, routing, fallback, RAG, Autopilot, evals.
- [Security Architecture](architecture-06-security-architecture): auth, RBAC, SSRF, read-only SQL, PII masking, audit log.
- [Deployment and Operations](architecture-07-deployment-and-ops): Vercel, Railway, AWS plan, env vars, monitoring.
- [Frontend Architecture](architecture-08-frontend-architecture): Next.js app structure, hooks, components, schemas.
- [ADR: Multi-agent Autopilot (DOCBOT-1505)](adr-multi-agent-autopilot): why Autopilot stays a single bounded graph.
- [Fine-tuning pipeline (DOCBOT-1523)](finetune-pipeline): trace-to-dataset fine-tuning tier (built, not in use).

### Public API
- [Developer API: Current State](api-01-current-state): what exists today and what blocks exposing it.
- [Public API and Developer Portal Plan](api-02-public-api-plan): goals, auth, versioning, limits, billing, rollout.
- [DocBot API v1 Reference (Proposed)](api-03-public-api-reference): the planned public endpoints and schemas.
- [Public API Implementation Backlog](api-04-implementation-backlog): tickets to build it.

### Planning and Backlog
- [Master backlog](master-backlog): phases, sprints and tickets. Source of truth for the board.
- [Enterprise roadmap (EPIC-27 to EPIC-40)](enterprise-roadmap): longer-range enterprise epics (draft).
- [User stories](user-stories): index of every DOCBOT story file with links to its work package.
- [Epic history (v2 project tracking)](epic-history): full history of EPIC-01 to EPIC-26.

### Operations and Runbooks
- [Engineering conventions (CLAUDE.md)](engineering-conventions): architecture rules, code style, testing, branch and board rules.
- [OpenProject board sync](openproject-sync): how tickets get onto pm.docbot.space and the board server.
- [Proof-run playbook (DOCBOT-2532)](proof-run-playbook): running the free tie-out for a prospect in under 60 minutes.
- [AWS migration exec plan](aws-migration-plan): Railway to AWS waves, cutover and rollback.
- [Founder actions: AWS](founder-aws-actions): founder-only account steps for the migration. Sensitive.

### Testing and QA
- [Evaluation harness](eval-harness): discrepancy, retrieval and tie-out evals and which run in CI.
- [Manual regression checklist](manual-regression-checklist): click-through test of shipped features on prod (needs update).
- [Robustness and benchmark test data](robustness-test-data): public datasets and attack corpora for stress tests.

### Business (sensitive)
- [Strategy and pricing 2027](strategy-2027): ICP, pricing lock, gates, phases.
- [Pre-seed 14-day plan (to Dec 1)](preseed-plan-dec1): plan to the 2026-12-01 investor call.
- [Pitch deck: main](pitch-deck-main): slide-by-slide spec and design system.
- [Pitch deck: technical](pitch-deck-technical): technical slides T1 to T6 and appendix.
- [Founder plan: one-pager](founder-plan-one-pager): one page for a partner or investor.
- [Founder plan: summary](founder-plan-summary): executive summary and verdict.
- [Founder plan: business plan](founder-plan-business-plan): full business plan.
- [Founder plan: CFO model](founder-plan-cfo): unit economics and sources.
- [Revenue plan](revenue-plan): MRR target to 2027-02-28 with weekly gates.
- [Market and GTM plan](market-gtm-plan): market size, positioning and launch calendar.

### Archive
- [YC W27 10-day plan](archive-yc-w27-plan): Oct sprint; still holds acceptance criteria for some EPIC-16/17/20/23 tickets.
- [Demo plan (May 2026)](archive-demo-plan): old TechCorp demo script.
- [Decentralized docs idea](archive-decentralized-docs-idea): verifiable workpapers idea; ticket ids conflict.
- [Founder ops notes](archive-founder-ops): October ops notes.
- [Plane sync (retired)](archive-plane-sync): the Plane board sync, replaced by OpenProject.
- [Doc status audit (2026-11-17)](archive-doc-status-audit): audit of the plans folder and its contradictions.
- [Investor demo sprint (Mar 2026)](archive-investor-demo-sprint): March demo sprint board.
- [Positioning, AWS and GTM plan](archive-positioning-plan): plan that produced the exec-plan pages.
- [Founder idea (Oct 2026)](archive-founder-idea): first founder-pack idea.
- [Strategy roadmap draft](archive-strategy-draft): draft of the strategy doc.
- [Unit economics (Mar 2026)](archive-unit-economics): March cost model.
- [docbot-platform plan](archive-platform-plan): early multi-tenant platform plan.

## How the docs stay current

- **The repo is the source of truth.** Every page here is a copy of a file in the DocBot repo (or the founder plans folder for business docs). Edit the file, not the wiki page, then re-import. `docs/wiki/_manifest.json` maps each page to its source.
- **Tickets live on the board, not here.** Status of every DOCBOT ticket is on OpenProject at https://pm.docbot.space (project `docbot`). The board is fed from `project-tasks/docbot-master-backlog.md` and `project-tasks/stories/` by `scripts/openproject_sync.py`.
- **Sync rule.** CLAUDE.md ("OpenProject Sync Rule") moves tickets to In Progress, Testing and Done on every branch, PR and merge. README is updated on full phase completion. Tracker history: Jira until 2026-11-17, Plane until 2026-11-22, OpenProject since.
- **Stale pages carry a banner** naming what is out of date and where the current answer lives. `docs/wiki/_inventory.md` lists every document, its decision and its staleness.
