# DocBot Enterprise Roadmap (EPIC-27 to EPIC-40)

**Status:** Draft for founder review. **Author:** PM agent. **Date:** 2026-11-17.
**Input for:** senior PM agent, who turns each epic below into stories with points and acceptance criteria.
**Sources read:** `CLAUDE.md`, `project-tasks/docbot-v2-project-tracking.md` (section 2, EPIC-25, EPIC-26), `project-tasks/yc-10-day-plan.md` (EPIC-16 to EPIC-23), `project-tasks/exec-plan/{market-gtm-plan,revenue-100k-plan,aws-migration-exec-plan}.html`, `docbot-platform/docs/{PLAN,EPICS}.md` (found at `~/Desktop/Personal_Projects/docbot-platform`, not next to the worktree), and `git log` on this branch for merged ticket IDs.

---

## 0. Summary

| Item | Answer |
|---|---|
| What "enterprise-ready" means here | A firm or finance team can pass DocBot through security review without exceptions, trust each number it produces, control who sees which client, and choose where the data and the model run. Section 1 lists the checks. |
| Phasing rule | Revenue first. Each phase opens only when the gate before it is met on revenue or a customer signal, not just a date. |
| Capacity split (revenue / enterprise) | Phase 1: 80 / 20. Phase 2: 50 / 50. Phase 3: 40 / 60. |
| New epics | 14 (EPIC-27 to EPIC-40). Story IDs use one hundred-block per epic (EPIC-27 = DOCBOT-27xx, ..., EPIC-40 = DOCBOT-40xx). |
| Biggest single dependency | Chroma on EFS pins prod to one task (`desired_count <= 1`, DOCBOT-2508/2516). No HA, no SLA and no BYOC scale until the vector store moves to pgvector (DOCBOT-3306). |
| Biggest open decision | Whether BYOC ships from this repo or from a `docbot-platform` engine fork (question 1 in section 9). |
| Claim rule (unchanged) | Do not use "enterprise-grade" before a SOC 2 report exists. Say "eval loop", not "self-improving fine-tuning". This roadmap's title is internal and does not go in public copy. |

---

## 1. Definition: enterprise-ready for DocBot

Three buyer types, in the order we sell to them. A check is required for a tier when that buyer's security review or contract will fail without it.

| # | Requirement | Firm (CAS / advisory, $500-2K/mo) | Pooled Enterprise ($25-60K/yr) | BYOC ($75-150K/yr) | Epic |
|---|---|---|---|---|---|
| R1 | Security review passes: DPA, questionnaire, sub-processor list, pen test summary | Required (DPA + questionnaire) | Required (+ pen test) | Required | 27, 28, 40 |
| R2 | SOC 2 report | Nice to have | Type I required, Type II within 12 months | Type II required | 27 |
| R3 | Per-client isolation and per-client access grants | Required | Required | Required | 30 (on top of 2562-2568) |
| R4 | SSO per org, SSO enforcement, SCIM, MFA | Google/Microsoft sign-in + MFA | Per-org SAML/OIDC + SCIM | Same + customer IdP only | 29 |
| R5 | Retention, verified deletion, export | Required (DPA terms) | Required + legal hold | Required + crypto-shred | 31 |
| R6 | Customer-managed keys (CMEK) | No | Asked often | Required | 28 |
| R7 | Model and data location control (private model, residency) | No | Sometimes | Required | 32, 31 |
| R8 | Every number cited and checkable; human sign-off before client delivery | Required | Required | Required | 34, 36 |
| R9 | Published SLO, status page, DR targets, then contractual SLA | Status page | SLA 99.5-99.9% | SLA per contract | 33 |
| R10 | Admin console, audit search, SIEM export | Audit export (exists) | Required | Required | 38 |
| R11 | Accounting / warehouse / content connectors | QBO, Xero | + Snowflake/BigQuery, SharePoint | + NetSuite, Databricks | 37 |
| R12 | API, webhooks, MCP | Optional | Required for integration | Required | 39 |
| R13 | Contract stack: MSA, SLA exhibit, order form, insurance | Order form (2533) | Required | Required | 40, 27 |

---

## 2. Enterprise readiness scorecard

"Today" reflects `HEAD` of this branch (ticket IDs from `git log`) plus the open-ticket status in the tracker. Phase = when the target is met.

| Capability | Today | Target | Phase |
|---|---|---|---|
| SOC 2 | None. Control map planned (2526, P3). | Type I report by 2027-06-30; Type II report by end of 2027 | 2 (I), 3 (II) |
| Third-party pen test | None. CI runs bandit, pip-audit, npm audit, gitleaks (1625, 1633, 1634) | Annual external test, criticals and highs fixed, summary shareable under NDA | 2 |
| Hosting | Railway single container; AWS prod env being built (Wave A/B) | AWS prod, private subnets, WAF, CloudTrail/GuardDuty (2519-2524) | 1 |
| High availability | Single task, Chroma on volume/EFS, `desired_count <= 1` | 2+ tasks across AZs, Multi-AZ RDS, pgvector | 2 |
| Disaster recovery | RDS PITR planned, restore drill planned (2510) | RTO 4h / RPO 15m, cross-region backup copy, quarterly drill | 2 |
| SLO / SLA | None published; 7-day soak target 99.5% (2513) | Public status page (P1); 99.9% SLA with credits after 90 measured days | 1 (SLO), 3 (SLA) |
| Tenancy | Org per signup (1702), CallerScope (1601-1603, 1611), per-org DEK (1529) | Org > client workspace > resource ACLs; RLS FORCE as defense in depth | 1 (workspaces), 2 (ACLs) |
| Roles | 3 org-wide roles (viewer/analyst/admin) | Preparer / reviewer / approver / client-viewer per workspace; custom roles later | 2, 3 |
| SSO | Global SAML via env; Google, GitHub, email login | Per-org SAML (2572) + generic OIDC + Entra (2571) + enforcement | 1 (Entra), 2 |
| SCIM | None | SCIM 2.0 users + groups (Okta, Entra) | 2 |
| MFA | None for password accounts | TOTP enforceable per org; passkeys later | 2, 3 |
| Encryption | Fernet + per-org envelope DEK (1529); RDS/EFS on CMK in prod (2508) | CMEK: customer KMS key wraps org DEK, revoke = crypto-shred | 3 |
| Retention / deletion | Vector stores expire after 30 days idle; commerce raw JSON 90 days; account delete not built (1708) | Per-org retention policy, verified deletion with receipt, legal hold | 1 (policy), 2 |
| DLP | Regex PII masking on outputs (fixed for numbers in 2003) | Ingest-time detection with block / mask / warn policy per org | 2 |
| Data residency | US only, implicit | Region pinning; EU stack only on a signed EU contract | 3 |
| Private model | Not shipped. 2609 (env-level OpenAI-compatible endpoint) in flight | Per-org model endpoint, eval-gated model list, self-hosted sandbox | 0 (demo), 2 (GA) |
| BYOC (customer AWS) | Frozen plan (docbot-platform PLAT-0413 to 0420) | First BYOC customer live | 3 |
| Air-gapped | None | Offline bundle, only on a signed contract | 3+ |
| Citation integrity | Tie-out eval harness (2541/2542), page citation bug open (2312) | Every number in an answer maps to a doc span or data row, else marked unverified | 1 (v1), 2 |
| Eval gating | Eval CI gate (1503), prompt versioning (1507), nightly live eval planned (2544) | Model or prompt change blocked unless gold set passes with margin | 2 |
| Prompt-injection defense | `api/utils/prompt_guard.py` untrusted tags | Injection corpus in CI, tool allowlists per step, external red team | 2 |
| Human review of workpapers | None (export exists, 2014) | Versioned workpapers, preparer/reviewer sign-off, approval before client export | 2 |
| Audit log | Append-only with immutability trigger, CSV export, org-scoped | Per-workspace (2570), searchable viewer, hash chain, SIEM stream | 1, 2 |
| Connectors | Postgres, MySQL, SQLite, Azure SQL, CSV, EDGAR, Amazon, Shopify | QBO, Xero, Snowflake, BigQuery, SharePoint, Drive; then NetSuite, Databricks | 2, 3 |
| API / MCP | Developer API with org-scoped keys (1530) | Versioned public API, webhooks, MCP server, SDKs | 2, 3 |
| Contracts | DP agreement (2549) and order form (2533) in progress | MSA, SLA exhibit, DPA, cyber + tech E&O insurance | 1, 2 |

---

## 3. Phases and gates

| Phase | Dates | Goal | Enterprise share | Entry gate | Exit gate |
|---|---|---|---|---|---|
| **Phase 0** | 2026-11-17 to 2026-12-01 | Pre-seed call. **EPIC-26 as written, not changed here.** | 0% new | n/a | Investor call held (EPIC-26 section 5) |
| **Phase 1** | 2026-12-02 to 2027-02-28 | $100K ARR run-rate, AWS cutover, Firm workspace sellable | 20% | Phase 0 done | **Revenue:** >= $8.3K MRR in Stripe by 2027-02-11 (EPIC-25 W12), or >= $5K MRR with >= 6 paying Firm/DP logos by 2027-02-28. **Signal:** >= 2 prospects (Firm or pooled) have sent a security questionnaire or asked for SOC 2 in writing. |
| **Phase 2** | 2027-03-01 to 2027-06-30 | Enterprise foundations, SOC 2 Type I, first pooled Enterprise contract | 50% | Phase 1 exit gate met. If it is missed by more than 50% (under $4K MRR), Phase 2 stays revenue-only and only the long-lead compliance stories run (2701-2705, 2716). | SOC 2 Type I report issued; >= 1 pooled Enterprise contract signed (or signed LOI with a security review passed); HA live (3306, 3307); workpaper sign-off used by >= 3 firms |
| **Phase 3** | 2027-07-01 to 2027-12-31 | BYOC, SOC 2 Type II, scale | 60% | Phase 2 exit gate met **and** a signed BYOC LOI or a pooled customer who requires single tenancy at >= $75K/yr | SOC 2 Type II report; first BYOC customer live; 99.9% SLO met for 90 days; contractual SLA offered |

**Rules**
1. Phase 1 does not start any new enterprise story that is not on the long-lead list (SOC 2 setup, retention policy, citation verifier v1, SLO + status page, contracts). Everything else in Phase 1 is existing tickets.
2. An epic can move earlier only on a named customer request with money attached (signed DP, LOI or order form), logged in the weekly gate tracker (2551).
3. The SOC 2 audit window starts **after** the AWS cutover (2519). It never covers Railway.

### Phase 1 order of work (existing tickets first, new stories marked NEW)

| Order | Track | Tickets | Notes |
|---|---|---|---|
| 1 | AWS cutover (Wave B) | 2576, 2513, 2509, 2510, 2514-2519, then 2520 | Already planned in EPIC-25. Prerequisite for the SOC 2 scope. |
| 2 | Design partners live | 2550, 2532, 2553, 1810 (PR #103), 2544, 2545 | Revenue. |
| 3 | Isolation sweep | EPIC-24 Stage A (2401-2406) | Not on `main` yet. Must land before workspaces stack on top. |
| 4 | Firm workspace | 2565, 2562, 2563, 2564, 2569, then 2566-2568, 2570, 2571, 2575 | The thing Firm buyers pay for. |
| 5 | Open security debt | 1613, 1614, 1626, 1627, 1628, 1703, 1708 | Small. Closes questionnaire gaps. |
| 6 | Wave C hardening | 2521, 2523, 2522, 2524, 2525, 2526 | 2526 becomes the input to EPIC-27. |
| 7 | NEW long-lead | 2701-2705, 2716, 3101, 3108, 3301-3304, 3401, 4002, 4003, 4005, 4011 | About 20% of capacity. |

---

## 4. Epic list

Size key: S <= 8 pts, M 9-20, L 21-40, XL > 40 (rough, before story-level estimation).

| Epic | Name | Phase(s) | Size | Primary buyer or risk |
|---|---|---|---|---|
| EPIC-27 | SOC 2 and Compliance Program | 1 (setup), 2 (Type I), 3 (Type II) | XL | Every buyer above Firm; panel trust objection |
| EPIC-28 | Security Engineering and Pen Test | 2, 3 | L | Security reviewers; breach risk |
| EPIC-29 | Enterprise Identity and Access | 2, 3 | L | IT admins at pooled/BYOC buyers |
| EPIC-30 | Tenancy and Fine-grained Permissions | 2, 3 | L | Firms with staff per client; client data leakage |
| EPIC-31 | Data Governance, Retention and DLP | 1 (policy), 2, 3 | L | DPA terms, GDPR/CCPA, client confidentiality |
| EPIC-32 | Deployment and Privacy Tiers | 2, 3 | XL | Regulated buyers; founder privacy priority |
| EPIC-33 | Reliability and SRE | 1 (SLO), 2, 3 | L | SLA buyers; month-end outage risk |
| EPIC-34 | AI Grounding, Quality and Safety | 1 (v1), 2 | L | Liability for a wrong number in a client deliverable |
| EPIC-35 | Robustness and Benchmark Test Program | 2, 3 | M | Investor and buyer proof; regression risk |
| EPIC-36 | Workpaper Review and Sign-off | 2 | L | Firm reviewers and partners; professional standards |
| EPIC-37 | Connectors: Accounting, Warehouse, Content | 2, 3 | XL | ICP clients on QBO/Xero; enterprise data stacks |
| EPIC-38 | Admin Console and Customer Observability | 2, 3 | M | Org admins, security teams |
| EPIC-39 | API, SDK and MCP Server | 2, 3 | M | Integrators; AI-native firms |
| EPIC-40 | Enterprise GTM Enablement and Commercial Ops | 1, 2, 3 | M | Closing pooled and BYOC deals |

---

## 5. Epics

### EPIC-27: SOC 2 and Compliance Program (XL)

| Field | Detail |
|---|---|
| Goal | Get a SOC 2 Type I report by 2027-06-30 and a Type II report by the end of 2027, scoped to the AWS system. |
| Why | The simulated panel: 18 of 20 buyers passed on trust and 19 of 20 named a DPA or SOC 2 (market-gtm-plan). Pooled Enterprise cannot close a second logo without a report. |
| Absorbs / depends on | Builds on 2526 (control map, IAM Identity Center, access review template) and 2531 (trust package). Depends on 2519 (cutover; scope is AWS only), 2524 (CloudTrail, GuardDuty, Config), 2523 (CMKs), 2512 (deploy approvals), 1625 (CI security jobs). Does not duplicate 2526: 2526 produces the control map, EPIC-27 runs the program. |
| Exit criteria | Type I report issued (Security criterion at minimum); trust center links to it under NDA; Type II observation window running with no open exceptions; public copy says only "SOC 2 Type I report available". |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-2701 | Select a compliance automation platform and connect AWS, GitHub, Google Workspace, Vercel | 1 |
| DOCBOT-2702 | System description and scope: AWS prod only; Vercel, Groq, Gemini, E2B, Stripe, Resend, Sentry listed as sub-processors | 1 |
| DOCBOT-2703 | Adopt the policy set (infosec, access, change management, incident response, BCP/DR, vendor, data classification, retention, SDLC, acceptable use) | 1 |
| DOCBOT-2704 | Risk assessment and risk register, reviewed quarterly | 1 |
| DOCBOT-2705 | Vendor register with each sub-processor's SOC report and DPA on file | 1 |
| DOCBOT-2706 | Change management evidence for a solo founder: branch protection, required CI, deploy approval, compensating controls written down | 2 |
| DOCBOT-2707 | Quarterly access reviews and an onboarding/offboarding checklist (staff and contractors) | 2 |
| DOCBOT-2708 | Security training and background checks for anyone with prod access | 2 |
| DOCBOT-2709 | Incident response plan plus one tabletop exercise (uses the 3303 runbooks) | 2 |
| DOCBOT-2710 | BCP/DR plan linked to the 3308 restore drills | 2 |
| DOCBOT-2711 | Auditor selection, readiness assessment, gap remediation list | 2 |
| DOCBOT-2712 | SOC 2 Type I audit and report | 2 |
| DOCBOT-2713 | Trust center page: report request under NDA, sub-processors, pen test summary (extends 2531) | 2 |
| DOCBOT-2714 | Type II observation window: monthly evidence checks, exception log | 3 |
| DOCBOT-2715 | SOC 2 Type II audit and report | 3 |
| DOCBOT-2716 | Cyber liability and tech E&O insurance in place (needed for MSAs and for the "wrong number" risk) | 1 |
| DOCBOT-2717 | Compliance calendar and control-owner matrix | 2 |

### EPIC-28: Security Engineering and Pen Test (L)

| Field | Detail |
|---|---|
| Goal | Find and fix what an attacker or a reviewer would find first; add customer-managed keys and controlled egress. |
| Why | A pen test summary is the second document every security reviewer asks for. CMEK is a standard BYOC and regulated-buyer ask. |
| Absorbs / depends on | Depends on open EPIC-16 items 1613, 1614, 1626, 1627, 1628 (finish them first, do not re-ticket) and AWS Wave C 2521 (private subnets, static egress IP), 2522 (WAF), 2523 (CMKs for platform services), 2525 (IaC scanning). Builds on 1529 (per-org DEK). Platform analogs: PLAT-0308 (KMS per org), PLAT-0408 (image signing). |
| Exit criteria | External pen test report with zero open critical/high; vulnerability SLAs met for one quarter; CMEK available to pooled and BYOC tenants; egress limited to an allowlist. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-2801 | Threat model refresh (STRIDE) for the AWS architecture, workspaces, invites and API keys | 2 |
| DOCBOT-2802 | Third-party pen test (web, API, cloud config) and remediation | 2 |
| DOCBOT-2803 | Vulnerability management policy with SLAs (critical 7 days, high 30 days) and a tracker | 2 |
| DOCBOT-2804 | DAST baseline (OWASP ZAP) against the dev env in CI | 2 |
| DOCBOT-2805 | Container image scanning and signing (Trivy + cosign), verified at deploy | 2 |
| DOCBOT-2806 | Customer DB credentials moved from the app DB blob to KMS-backed Secrets Manager references | 2 |
| DOCBOT-2807 | CMEK: the customer's KMS key wraps the org DEK; revoking it makes the data unreadable | 3 |
| DOCBOT-2808 | Egress allowlist proxy for API tasks (LLM, E2B, Stripe, customer DB hosts only) | 3 |
| DOCBOT-2809 | Customer DB network options: published static egress IP, SSH tunnel, then AWS PrivateLink | 2 |
| DOCBOT-2810 | Vulnerability disclosure policy and security.txt; bug bounty only after SOC 2 | 2 |
| DOCBOT-2811 | Annual pen test retest | 3 |

### EPIC-29: Enterprise Identity and Access (L)

| Field | Detail |
|---|---|
| Goal | Customer IT controls who gets in, how, and when they lose access. |
| Why | Pooled and BYOC buyers require SSO enforcement and SCIM deprovisioning. Firms with staff turnover need MFA and offboarding. |
| Absorbs / depends on | References, does not duplicate: 2571 (Entra OIDC), 2572 (per-org SAML, verified domain, require SSO), 2567 (seats, offboarding), 1626 (SAML replay), 1703 (password reset), 1617 (identity linking, merged), 1530 (developer API keys, merged). Depends on 2565/2566 (invites, multi-org membership). |
| Exit criteria | An Okta and an Entra tenant can each provision, update and deprovision users through SCIM; an org can enforce SSO and MFA; sensitive actions need recent re-auth. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-2901 | MFA (TOTP + recovery codes) for password accounts, enforceable per org | 2 |
| DOCBOT-2902 | Generic per-org OIDC SSO (Okta OIDC, Google Workspace) next to per-org SAML (2572) | 2 |
| DOCBOT-2903 | SCIM 2.0 users endpoint with deprovision that revokes sessions and API keys | 2 |
| DOCBOT-2904 | SCIM groups mapped to roles and workspace grants | 2 |
| DOCBOT-2905 | Org session policy: idle timeout, max session length, optional IP allowlist | 2 |
| DOCBOT-2906 | Step-up re-auth (<= 15 min) before export, API key creation, role or SSO changes | 2 |
| DOCBOT-2907 | Scoped API keys with expiry, plus service accounts not tied to a person | 2 |
| DOCBOT-2908 | Login and SSO events in the audit log with a simple anomaly flag (new country, many failures) | 2 |
| DOCBOT-2909 | Passkeys (WebAuthn) | 3 |
| DOCBOT-2910 | Okta Integration Network and Entra app gallery listings | 3 |

### EPIC-30: Tenancy and Fine-grained Permissions (L)

| Field | Detail |
|---|---|
| Goal | Inside a firm, each person sees only the clients, connections and workpapers they are granted, enforced in the app and again in the database. |
| Why | A firm's biggest fear is one client's books showing up in another client's answer. Workpaper sign-off (EPIC-36) needs preparer and reviewer roles. |
| Absorbs / depends on | Depends on EPIC-24 Stage A 2401-2406 (two-org isolation sweep, `org_id` NOT NULL, org-namespaced Chroma, per-org quotas) and Firm workspace 2562, 2563, 2568, 2575. Extends 1625 (route-guard sweep). Platform analogs: PLAT-0203 (RLS FORCE), PLAT-0303 (module gating). |
| Exit criteria | Org x workspace x role isolation matrix runs in CI over every route; RLS FORCE on tenant tables; client guests can view one workspace read-only; org admins can switch off risky features. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3001 | Permission model v2: per-workspace roles preparer, reviewer, approver, client-viewer, mapped from today's org roles | 2 |
| DOCBOT-3002 | Resource-level grants on connections and document sets (who can query which ledger) | 2 |
| DOCBOT-3003 | Restricted columns: per-connection column masking applied to SQL and CSV results before the LLM sees them | 2 |
| DOCBOT-3004 | Org policy switches: Python sandbox, EDGAR, external LLM, exports, persona list | 2 |
| DOCBOT-3005 | Postgres RLS ENABLE + FORCE on tenant tables, app role NOBYPASSRLS, as a second wall behind CallerScope | 2 |
| DOCBOT-3006 | Isolation test matrix in CI: org x workspace x role for every route and SSE stream | 1 |
| DOCBOT-3007 | Client guest access: invite a client user read-only to one workspace | 2 |
| DOCBOT-3008 | Custom roles per org | 3 |
| DOCBOT-3009 | Multi-entity workspaces: parent and child entities for consolidation tie-outs | 3 |

### EPIC-31: Data Governance, Retention and DLP (L)

| Field | Detail |
|---|---|
| Goal | Every byte of client data has a retention rule, a provable deletion path and a record of which provider saw it. |
| Why | The DPA (2531) makes retention and deletion promises that the code must keep. GDPR/CCPA requests. Firms answer to their own clients. |
| Absorbs / depends on | References 1708 (account deletion + export), 2575 (workspace offboarding), 2531 (DPA), 1618 (Sentry scrubbing, merged), 2003 (PII masker fix, merged), existing `VECTOR_STORE_RETENTION_DAYS` and `COMMERCE_RAW_JSON_RETENTION_DAYS`. Uses the 1529 per-org DEK for crypto-shred. Platform analog: PLAT-0419 (offboarding with crypto-shred). |
| Exit criteria | Retention policy per org enforced by a daily job with evidence; deletion produces a receipt; DLP policy runs on ingest; a canary PII string never appears in logs, traces or Sentry. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3101 | Retention policy engine: per-org defaults for sessions, uploads, vectors, traces, LLM call logs, matching the DPA | 1 |
| DOCBOT-3102 | Verified deletion: job, deletion receipt, audit event (extends 1708 and 2575) | 2 |
| DOCBOT-3103 | Crypto-shred on org or workspace delete by destroying the per-org DEK | 2 |
| DOCBOT-3104 | Full org/workspace export (documents, workpapers, audit log) in an open format | 2 |
| DOCBOT-3105 | DLP on ingest: detect SSN, bank, card and personal data in PDFs and CSVs; org policy block, mask or warn | 2 |
| DOCBOT-3106 | Per-org "redact before LLM" option and a per-request record of provider, model and region | 2 |
| DOCBOT-3107 | PII canary test across Sentry, LangSmith, CloudWatch and the LLM cache | 2 |
| DOCBOT-3108 | Sub-processor terms: confirm no-training and retention terms with Groq, Google, E2B; publish the list | 1 |
| DOCBOT-3109 | DSAR process (access, delete) with a 30-day clock | 2 |
| DOCBOT-3110 | Legal hold flag that suspends deletion for a workspace | 3 |
| DOCBOT-3111 | Data residency: region pinning; an EU stack only when an EU contract is signed | 3 |

### EPIC-32: Deployment and Privacy Tiers (XL)

| Field | Detail |
|---|---|
| Goal | Ship the four privacy tiers as products: Hosted, Private model mode, Single-tenant in the customer's AWS (BYOC), Air-gapped. |
| Why | Founder priority, and the deciding factor for regulated buyers and the BYOC price band ($75-150K/yr). |
| Absorbs / depends on | Starts from 2609 (env-level private model mode) and 2608 (data-flow page). References EPIC-19 (1901-1906: org LLM keys, per-request key resolution, OpenAI-compatible provider, per-model formatting suite). BYOC is docbot-platform M4 (PLAT-0413 to 0420) plus PLAT-0305/0306 (model routing) and PLAT-0711 (break-glass). Docker Compose from EPIC-06 is the base for the customer-hosted bundle. Do not re-ticket PLAT stories; 3205 decides where they run. |
| Exit criteria | Tier 2 GA with a per-org model endpoint and working Python; a customer-hosted bundle installs in under one hour; one BYOC customer live; air-gapped bundle exists only if a contract requires it. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3201 | Tier matrix spec and an automated "what leaves the box" egress test per tier (extends 2608) | 1 |
| DOCBOT-3202 | Private model mode GA: per-org model endpoint (not only env), eval-gated list of supported open-weight models | 2 |
| DOCBOT-3203 | Self-hosted Python sandbox for private mode (container runner replacing E2B) | 2 |
| DOCBOT-3204 | Customer-hosted single-node bundle (compose, local Postgres, local embeddings) with a license key for POCs | 2 |
| DOCBOT-3205 | BYOC architecture decision: this repo's `infra/aws` modules vs docbot-platform engine extraction | 2 |
| DOCBOT-3206 | BYOC deploy into a customer AWS account via cross-account role with ExternalId (executes PLAT-0413/0414 or the 3205 choice) | 3 |
| DOCBOT-3207 | BYOC releases: signed images, release channels, customer-approved upgrades (PLAT-0416 analog) | 3 |
| DOCBOT-3208 | Telemetry-off mode and an opt-in, metadata-only health beacon | 2 |
| DOCBOT-3209 | BYOC support access: break-glass with customer approval and a tenant-visible log (PLAT-0711 analog) | 3 |
| DOCBOT-3210 | Sizing and cost guide per tier | 2 |
| DOCBOT-3211 | Air-gapped bundle: offline image and model tarballs, offline license, zero outbound calls verified (contract-triggered) | 3+ |

### EPIC-33: Reliability and SRE (L)

| Field | Detail |
|---|---|
| Goal | Publish SLOs, survive an AZ failure, restore within stated RTO/RPO, then sell an SLA. |
| Why | Month-end close is the worst time to be down. Pooled Enterprise contracts carry an SLA exhibit. |
| Absorbs / depends on | Depends on AWS Wave B/C: 2509 (rotation), 2510 (restore drill), 2511 (alarms), 2513 (soak), 2518 (rollback). References open EPIC-23 items 2304 (stop + idle timeout), 2308 (DB server timeouts), 2309 (schema-drift retry), and merged 2307 (LLM fallback). Platform analogs: PLAT-0207/0208 (vector interface, pgvector), PLAT-0209 (worker entrypoint), PLAT-0201 (Alembic). |
| Exit criteria | Status page live; 2+ tasks across AZs with pgvector; quarterly restore drill meets RTO 4h / RPO 15m; 99.9% availability for 90 days; SLA exhibit offered. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3301 | SLOs and error budgets: availability, time to first token p95, tie-out run success rate; dashboard | 1 |
| DOCBOT-3302 | Public status page and incident comms templates | 1 |
| DOCBOT-3303 | Paging and runbooks for the top 10 failure modes (LLM provider down, E2B down, RDS full, stuck sync) | 1 |
| DOCBOT-3304 | Postmortem process and template | 1 |
| DOCBOT-3305 | Background worker for Autopilot, ingest and connector sync, off the API process | 2 |
| DOCBOT-3306 | Vector store to pgvector with org/workspace scoping and a retrieval eval gate (removes the single-task limit) | 2 |
| DOCBOT-3307 | Multi-AZ RDS and 2+ API tasks across AZs | 2 |
| DOCBOT-3308 | DR: cross-region backup copy, RTO/RPO targets, quarterly restore drill (extends 2510) | 2 |
| DOCBOT-3309 | Load test at 10x current peak (concurrent Autopilot + SSE) and a capacity plan | 2 |
| DOCBOT-3310 | LLM provider health routing with circuit breaker and a degraded-mode banner | 2 |
| DOCBOT-3311 | Schema migrations via Alembic with expand/contract, zero-downtime deploys | 2 |
| DOCBOT-3312 | Quarterly game day | 3 |
| DOCBOT-3313 | Contractual SLA (99.9%) with service credits, after 90 measured days | 3 |

### EPIC-34: AI Grounding, Quality and Safety (L)

| Field | Detail |
|---|---|
| Goal | No number reaches a client unless it is traced to a source or clearly marked; model and prompt changes cannot ship a regression; documents cannot hijack the agent. |
| Why | The product's promise is checkable numbers. One wrong figure in a lender report is a liability event and a lost logo. Board packs and 10-Ks are untrusted input. |
| Absorbs / depends on | Builds on 2541/2542 (tie-out eval, merged), 2543-2545 (live number, nightly, gold set v2), 1503 (eval CI gate), 1504 (structured output validation), 1507 (prompt versioning), `prompt_guard.py`. Depends on 2312 (correct page citations, open). PLAT-0609 (red team) is the platform analog. Robustness datasets live in EPIC-35, not here. |
| Exit criteria | Citation coverage of numbers in answers >= 99% on the gold set with the rest marked unverified; release gate blocks any model/prompt change that drops tie-out accuracy; injection corpus passes in CI; one external red team with findings fixed. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3401 | Citation verifier v1: each number in an answer maps to a cited doc span or data row, else shown as "unverified" | 1 |
| DOCBOT-3402 | Deterministic arithmetic: sums, deltas and percentages recomputed in code; mismatch blocks the answer | 2 |
| DOCBOT-3403 | Abstain policy: "not enough evidence" when retrieval confidence is low; false-answer rate measured | 2 |
| DOCBOT-3404 | Release gate: model + prompt registry; a change ships only if the gold set passes with a margin | 2 |
| DOCBOT-3405 | Prompt-injection defense v2: document-borne injection corpus in CI, tool allowlist per Autopilot step | 2 |
| DOCBOT-3406 | Red team: internal agent pass, then an external pass; findings fixed | 2 |
| DOCBOT-3407 | Production quality sampling: LLM judge on sampled prod traces per org with drift alerts | 2 |
| DOCBOT-3408 | Feedback loop: thumbs-down to triage to gold-set candidate | 2 |
| DOCBOT-3409 | Provenance panel per answer: sources, model, prompt version, run id | 2 |
| DOCBOT-3410 | Customer-facing AI use and model card document | 2 |

### EPIC-35: Robustness and Benchmark Test Program (M)

| Field | Detail |
|---|---|
| Goal | Measure DocBot on outside benchmarks and on messy real-world inputs, and keep those numbers from regressing. |
| Why | The only allowed public number today is the internal tie-out eval (2543). Buyers and investors will ask how it holds up on data we did not write. |
| Absorbs / depends on | Dataset list comes from a separate research agent (pending). Reuses the `tests/eval` scorer (2541) and gold set v2 (2545). Never cites the 20/20 LLM extraction baseline (1004). |
| Exit criteria | Defined after the dataset list arrives. Minimum: baseline published with method, nightly run with thresholds. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3501 | Take in the research agent's dataset list; license review; pick the benchmark set | 2 |
| DOCBOT-3502 | Harness adapters (doc QA, table QA, text-to-SQL, tie-out) on the existing scorer | 2 |
| DOCBOT-3503 | Baseline run and a published method page | 2 |
| DOCBOT-3504 | Perturbation suite: scanned PDFs, messy QBO/Xero exports, schema variants | 2 |
| DOCBOT-3505 | CI subset plus nightly full run with thresholds | 3 |

### EPIC-36: Workpaper Review and Sign-off (L)

| Field | Detail |
|---|---|
| Goal | A tie-out becomes a versioned workpaper that a preparer prepares, a reviewer signs off, and only then goes to the client. |
| Why | Firms work under review standards; a partner will not send an AI result without sign-off. This also makes DocBot sticky month over month (roll-forward). |
| Absorbs / depends on | Builds on 2014 (tie-out XLSX/PDF export, merged), `lineage_service.py`, 2563 (workspace scoping), 2570 (audit per workspace), 3001 (preparer/reviewer roles). |
| Exit criteria | >= 3 firms complete a full prepare, review, approve, export cycle on a real close; every export carries version and approver. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3601 | Workpaper object: a saved tie-out with input hashes, query text, data snapshot time, outputs, lineage | 2 |
| DOCBOT-3602 | Versions and diff between runs | 2 |
| DOCBOT-3603 | Line-level review notes with open/resolved status | 2 |
| DOCBOT-3604 | Preparer and reviewer sign-off with dual control; approved workpaper is locked | 2 |
| DOCBOT-3605 | Client-facing export allowed only after approval, stamped with approver and version | 2 |
| DOCBOT-3606 | Roll-forward: re-run last period's workpaper on the new period's data | 2 |
| DOCBOT-3607 | Evidence binder export: citations, sign-off trail, audit excerpt (extends 2014) | 2 |
| DOCBOT-3608 | Email notifications for review requests and approvals | 2 |
| DOCBOT-3609 | Expiring read-only client share link with watermark | 3 |
| DOCBOT-3610 | Review queue view per workspace | 3 |

### EPIC-37: Connectors: Accounting, Warehouse, Content (XL)

| Field | Detail |
|---|---|
| Goal | Read the client's books and documents where they already live, read-only. |
| Why | ICP clients run QuickBooks Online and Xero, not SQL (panel: P004, P010). Enterprise finance teams keep data in Snowflake, BigQuery, NetSuite, and documents in SharePoint or Drive. |
| Absorbs / depends on | Reuses `api/connectors/` (base, registry, rate limiter; 701), sync worker (704), persistence (706), connector RBAC (1602, merged). Depends on 2532 (CSV proof-run path, which stays the fallback). Warehouses use the 7-step SQL pipeline and 3-layer read-only rule. Commerce connectors are not ICP and get no new work. |
| Exit criteria | QBO and Xero live with app-store approval; one warehouse connector used by a pooled customer; SharePoint ingest used by a firm. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3701 | QuickBooks Online read-only connector (OAuth2; trial balance, GL, P&L, balance sheet); submit Intuit app review early | 2 |
| DOCBOT-3702 | Xero read-only connector (OAuth2; reports API); Xero app certification | 2 |
| DOCBOT-3703 | Normalized accounting schema (accounts, journal lines, periods) so tie-out SQL works the same across sources | 2 |
| DOCBOT-3704 | Snowflake connector (key-pair auth, read-only role, query tag) | 2 |
| DOCBOT-3705 | BigQuery connector (service account or workload identity, max-bytes-billed guard) | 2 |
| DOCBOT-3706 | SharePoint / OneDrive document source (Graph API, folder sync, access check at ingest) | 2 |
| DOCBOT-3707 | Google Drive document source | 2 |
| DOCBOT-3708 | Connector health page: last sync, errors, reauth prompts | 2 |
| DOCBOT-3709 | Warehouse cost guard: estimate before run, per-org limits | 2 |
| DOCBOT-3710 | NetSuite read-only (token auth, SuiteQL) | 3 |
| DOCBOT-3711 | Databricks SQL warehouse (OAuth M2M, Unity Catalog read-only) | 3 |

### EPIC-38: Admin Console and Customer Observability (M)

| Field | Detail |
|---|---|
| Goal | One place where an org admin manages people, workspaces, SSO and policy, and where a security team can see and export what happened. |
| Why | Security reviewers ask "can we see who did what, and can we pipe it to our SIEM". Firm admins need seat and usage control. |
| Absorbs / depends on | Builds on `audit_service.py`, 2570 (audit per workspace), 1807 (revenue metrics in admin), 1811 (billing UI), 1530 (API keys). Platform analogs: PLAT-0309 (audit hash chain, WORM), PLAT-0711 (break-glass). |
| Exit criteria | Admin console covers members, workspaces, roles, SSO, policies, usage; audit events stream to a customer S3 bucket or webhook; support access needs customer approval. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3801 | Org admin console: members, workspaces, roles, SSO, policies | 2 |
| DOCBOT-3802 | Usage and credit dashboard per workspace and member | 2 |
| DOCBOT-3803 | Audit log viewer with search and filters | 2 |
| DOCBOT-3804 | Audit hash chain and WORM archive to S3 Object Lock | 2 |
| DOCBOT-3805 | SIEM export: audit stream to customer S3 or signed webhook | 3 |
| DOCBOT-3806 | Admin view of LLM data flow (provider, model, region per request; uses 3106) | 2 |
| DOCBOT-3807 | Support access: just-in-time, customer-approved, time-boxed, logged | 2 |
| DOCBOT-3808 | Admin alerts: new member, SSO change, export spike | 3 |

### EPIC-39: API, SDK and MCP Server (M)

| Field | Detail |
|---|---|
| Goal | Let customers and their AI tools run tie-outs and read workpapers programmatically, with the same scoping and audit as the UI. |
| Why | Pooled customers integrate into close tooling. AI-native firms use Claude/ChatGPT with MCP; an MCP server makes DocBot the trusted numbers tool in their stack (also a distribution channel). |
| Absorbs / depends on | Builds on 1530 (developer API, org-scoped keys) and 2907 (scoped keys). PLAT-0602 (MCP server over the bounded pipelines) is the platform analog. |
| Exit criteria | Versioned public API with OpenAPI spec and deprecation policy; signed webhooks; MCP server with OAuth and read-only tools used by >= 1 customer. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-3901 | Public API v1 contract: OpenAPI spec, versioning and deprecation policy, per-key rate limits | 2 |
| DOCBOT-3902 | Tie-out run as an async API job with polling | 2 |
| DOCBOT-3903 | Signed webhooks: run.completed, discrepancy.found, workpaper.approved | 2 |
| DOCBOT-3904 | MCP server: read-only tools (run tie-out, doc search, SQL query, get workpaper), OAuth, workspace scoping, audited | 2 |
| DOCBOT-3905 | Developer docs site and quickstart | 2 |
| DOCBOT-3906 | Python SDK | 3 |
| DOCBOT-3907 | TypeScript SDK | 3 |

### EPIC-40: Enterprise GTM Enablement and Commercial Ops (M)

| Field | Detail |
|---|---|
| Goal | Make an enterprise deal closable by a solo founder: paperwork ready, pilot shape fixed, procurement answered. |
| Why | The $100K plan books one pooled Enterprise deal ($30K/yr, 30% of target) by W12. That deal stalls on paperwork more often than on product. |
| Absorbs / depends on | References 2531 (trust package), 2549 (DP agreement), 2552 (enterprise target list + one-pager), 2553 (case studies), 2533 (order form), 1809 (metered overage), 1810 (subscriptions). Copy rules from EPIC-26 claim audit apply to every artifact. |
| Exit criteria | MSA, SLA exhibit, DPA and order form signed off by a lawyer; security review kit answers a SIG Lite in under one day; >= 1 pooled contract closed on these documents. |

| ID | Candidate story | Phase |
|---|---|---|
| DOCBOT-4001 | Security review kit: SIG Lite and CAIQ answers, architecture and data-flow diagrams | 1 (draft), 2 (with pen test + SOC 2) |
| DOCBOT-4002 | MSA, SLA exhibit and enterprise order form (lawyer review) | 1 |
| DOCBOT-4003 | Packaging: Firm vs pooled vs BYOC feature matrix and the flags that enforce it | 1 |
| DOCBOT-4004 | Annual invoiced contracts: Stripe invoicing with PO numbers, net-30, usage commit tracking | 2 |
| DOCBOT-4005 | Paid pilot playbook: 30 days, written success criteria, exit report | 1 |
| DOCBOT-4006 | ROI calculator from design-partner data (hours per close, discrepancies caught) | 2 |
| DOCBOT-4007 | QuickBooks App Store and Xero App Store listings (after 3701/3702) | 2 |
| DOCBOT-4008 | Referral program for CAS and advisory networks | 2 |
| DOCBOT-4009 | Procurement pack: W-9, insurance certificates, vendor forms, accessibility statement | 2 |
| DOCBOT-4010 | Enterprise demo environment with synthetic multi-entity data | 2 |
| DOCBOT-4011 | Claims governance: every public claim mapped to evidence; SOC 2 wording unlocked only by a report | 1 (ongoing) |

---

## 6. Dependency map (existing open tickets these epics wait on)

| Existing ticket(s) | Status | Unblocks |
|---|---|---|
| 2519 AWS cutover (and 2509, 2510, 2513-2518) | Open, Wave B | EPIC-27 audit scope, EPIC-28 pen test target, EPIC-33 |
| 2521-2525 AWS Wave C hardening | Open, P2/P3 | EPIC-27 evidence, EPIC-28 |
| 2526 SOC 2 groundwork | Open, P3 (pull to Phase 1) | EPIC-27 |
| EPIC-24 Stage A 2401-2406 | Not on `main` | EPIC-30, EPIC-33 (3306), docbot-platform PLAT-0102 |
| 2562-2570, 2575 Firm workspace | Open | EPIC-30, EPIC-36, EPIC-38 |
| 2571, 2572 Entra, per-org SAML | Open | EPIC-29 |
| 1613, 1614, 1626, 1627, 1628 (EPIC-16 remainder) | Open | EPIC-28 pen test readiness |
| 1703, 1708 password reset, account deletion | Open | EPIC-29, EPIC-31 |
| 1901-1906 EPIC-19 BYOK (LLM keys) | Open, deferred | EPIC-32 Tier 2 GA |
| 2609 private model mode | In flight (Phase 0 stretch) | EPIC-32 |
| 2312 page citations | Open | EPIC-34 (3401) |
| 2543-2545 live eval, nightly, gold set v2 | Open | EPIC-34, EPIC-35 |
| 2531, 2549, 2552, 2553 trust, DP agreement, enterprise list, case studies | Open | EPIC-40 |
| PLAT-0413 to 0420 (docbot-platform M4) | Frozen | EPIC-32 BYOC (after 3205 decision) |

**Naming collision to fix in the tracker:** EPIC-19 "BYOK" means bring your own **LLM API key**. Enterprise buyers mean customer-managed **encryption keys**. This roadmap calls the second one **CMEK** (DOCBOT-2807). Use those two terms in all copy.

---

## 7. Top 10 risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Enterprise work starves revenue (solo founder) | High | High | Capacity split per phase; Phase 2 opens only on the revenue gate; weekly check in 2551 |
| 2 | SOC 2 as a solo founder: no segregation of duties, key-person risk | High | High | Compliance platform, written compensating controls (2706), fractional security advisor; first hire can own controls |
| 3 | Two engines drift: DocBot repo vs a docbot-platform fork (PLAT-0102 copies the engine) | Medium | High | Decide in 3205 before any BYOC build; default to one engine with tier flags |
| 4 | Chroma pins prod to one task, so no HA and no SLA | High (certain today) | High | 3306 pgvector migration early in Phase 2 with a retrieval eval gate |
| 5 | Sub-processors (Groq, Gemini, E2B) fail a buyer's DPA or residency review | Medium | High | 3108 terms on file; Tier 2 private mode with self-hosted sandbox (3202, 3203) as the answer |
| 6 | A wrong number reaches a client deliverable | Medium | Very high | 3401 citation verifier, 3402 deterministic math, 3604 sign-off, E&O insurance (2716) |
| 7 | Prompt injection in a client document steers SQL or the answer | Medium | High | 3405 corpus in CI, read-only 3-layer SQL rule, tool allowlists, 3406 red team |
| 8 | Enterprise deal concentration: one $30K logo is 30% of the ARR target | High | Medium | Existing swap rule (no LOI by W8 → 5 more Firm partners); 4002/4005 shorten the cycle |
| 9 | Connector platform reviews (Intuit, Xero) take weeks and APIs change | Medium | Medium | Start app review in Phase 1; CSV export path (2532) stays the fallback |
| 10 | Claim drift: copy says "enterprise-grade" or cites SOC 2 before a report exists | Medium | High (trust) | 4011 claims governance; EPIC-26 claim audit applied to every page and deck |

---

## 8. What NOT to build (parking lot)

| Item | Why parked | Revisit when |
|---|---|---|
| New commerce connector features (Amazon, Shopify) | Sellers are not ICP | Never as product focus; keep only as data sources |
| Per-tenant fine-tuning (PLAT EPIC-07) and automatic retraining in prod | Cost, consent and eval burden; claim risk | >= 10 paying tenants asking for it, with consent flows |
| OpenCode analyst workbench (PLAT EPIC-06) | Large agent-runtime surface before core trust is proven | After SOC 2 Type II and red team |
| ISO 27001, HIPAA, FedRAMP, PCI | Not asked for by ICP or pooled buyers | A signed contract requires it |
| Multi-region active-active | Cost and complexity far beyond the SLA need | Contractual SLA above 99.95% |
| EU / other-region stacks | No EU buyer yet | Signed EU contract (3111) |
| BYOC on Azure or GCP, Kubernetes/Helm charts | AWS BYOC first | Two qualified non-AWS BYOC deals |
| White-label theme engine (PLAT-0507) | Firms need logos on exports at most | A paying firm asks for client-branded portals |
| Excel add-in, Slack bot, mobile app | Distribution ideas without demand evidence | Repeated asks from paying firms |
| Real-time co-editing of workpapers | Review notes + sign-off cover the need | Firms with 10+ reviewers on one workpaper |
| More DB dialects (Oracle, on-prem SQL Server) | Ledgers live in QBO/Xero/NetSuite/warehouses | A pooled deal requires it |
| Bug bounty program | Noise before SOC 2 and a pen test | After Type I |

---

## 9. Open questions for the founder

1. **One engine or two?** Should BYOC deploy this repo (extending `infra/aws` modules) instead of unfreezing docbot-platform's engine extraction (PLAT-0102)? Recommendation: one engine with tier flags; reuse PLAT designs, not a fork.
2. **SOC 2 budget and timing:** year one is roughly $25-45K (compliance platform, auditor, pen test, insurance; estimate, not quoted). Start the platform in Phase 1 from revenue, or wait for the pre-seed to close?
3. **Python in private model mode:** for Tier 2, is "no Python sandbox" acceptable at launch, or must a self-hosted sandbox (3203) ship with Tier 2 GA?
4. **First hire:** security/devops (owns SOC 2 controls, on-call, BYOC) or sales (owns the enterprise pipeline)? This changes Phase 2 capacity and the segregation-of-duties story.
5. **Residency signal:** are any current prospects outside the US (EU, UK, Canada)? That decides whether 3111 stays in Phase 3 or is parked.
