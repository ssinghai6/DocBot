# DocBot "Chat with Your Database" — Master Plan
> Synthesized from AI Engineer + Software Architect + Product Manager research
> Date: 2026-03-16

---

## THE VERDICT (Read This First)

Every existing tool — Vanna.ai, LangChain SQL Agent, Text2SQL.ai, Metabase AI — solves one side of the problem:
- **They do SQL.** They cannot read your documents.
- **DocBot already does documents.** With expert personas.

The **one thing no tool on the market does today** is answer a question that spans both simultaneously:

> *"Our Q3 board deck projected $2.4M ARR. What does our actual transactions database show?"*

That cross-source synthesis — where the Finance Expert persona reads the PDF projection AND queries the live DB — is the product. Not Text2SQL. Not another RAG chatbot. **The reconciliation of documents against live data, in a single expert-framed answer with dual citations.**

**Positioning:** "DocBot is the only assistant that can reconcile your live data with your strategic documents."

*Phase 4 extension:* "For commerce teams: the only tool that unifies Amazon, Shopify, and your internal documents in a single natural language query."

---

## PRE-BUILD VALIDATION (Do This Before Writing Code)

The PM's most important finding: **Run 5 discovery interviews before a single line is written.**

The risk: target users (Finance Managers, Ops Leads) may have their data in QuickBooks and Google Sheets — not PostgreSQL. If that's true, the connector changes; the hybrid concept does not.

**5 questions to validate in each interview:**
1. Where does your business data actually live? (PostgreSQL? MySQL? Excel? QuickBooks?)
2. Do you have direct SQL access or only through a BI tool?
3. What documents do you cross-reference against that data? (PDFs? Decks? Reports?)
4. How long does that reconciliation take you today?
5. "If I could answer [specific hybrid question] in 8 seconds, would you use that daily?"

If 4 of 5 say their data is in Google Sheets → pivot to Google Sheets connector in v1. Hybrid concept is identical; data source changes.

---

## WHAT MAKES THIS GENUINELY UNIQUE

| Feature | DocBot | Vanna.ai | LangChain SQL Agent | Any Text2SQL tool |
|---------|--------|----------|---------------------|-------------------|
| Chat with PDFs | ✅ | ❌ | ❌ | ❌ |
| Chat with SQL DB | ✅ (new) | ✅ | ✅ | ✅ |
| **Hybrid: Docs + DB in one answer** | ✅ **(only one)** | ❌ | ❌ | ❌ |
| Expert personas (Doctor/Finance/Lawyer) | ✅ | ❌ | ❌ | ❌ |
| Query explanation in plain English | ✅ (new) | SQL only | SQL only | SQL only |
| Schema-semantic RAG (LLM-described tables) | ✅ (new) | ❌ | ❌ | ❌ |
| Self-improving (few-shot from history) | ✅ (new) | ✅ (requires training) | ❌ | ❌ |
| Discrepancy detection (doc vs. DB) | ✅ **(only one)** | ❌ | ❌ | ❌ |

---

## PHASED BUILD PLAN

### Phase 0: Validation (1 week, zero code)
- [ ] 5 user discovery interviews
- [ ] Confirm target users have SQL access (not just BI tools)
- [ ] Validate that the "hybrid" use case is a real daily pain

---

### Phase 1: MVP DB Chat (2–3 weeks after validation)

**Goal:** Basic, safe SQL generation against a SQLite file upload. Hybrid mode ships in simplified form.

**Key decision from PM:** Start with **SQLite file upload**, NOT connection strings.
- Sidesteps the #1 adoption blocker: users won't give production DB credentials to a third-party tool
- Sidesteps the Vercel timeout risk (local file, no network latency)
- Uses the same drag-and-drop UI pattern as PDF upload — minimal new UX

**Deliverables:**
1. `api/db_service.py` — new module with all DB logic
2. 5 new API routes in `api/index.py` (additive, no existing code touched)
3. SQLite DDL for 5 new tables (added to existing `init_db()`)
4. Frontend: "Database" tab with connection panel + DB chat interface
5. Simplified Hybrid mode (doc session + SQLite = hybrid answer)

**New API Routes:**
```
POST   /api/db/connect              → connect DB, introspect schema
POST   /api/db/chat                 → NL → SQL → execute → answer
GET    /api/db/schema/{id}          → return schema overview
DELETE /api/db/disconnect/{id}      → cleanup
POST   /api/hybrid/chat             → docs + DB = one answer
GET    /api/db/sessions             → list active DB sessions
```

**New SQLite Tables:**
```sql
db_connections    -- encrypted credentials registry
db_sessions       -- one session per connection, tracks history
schema_cache      -- TTL-based schema storage (key: db_connection_id)
query_history     -- every successful query stored for few-shot learning
query_embeddings  -- NL question vectors for similarity retrieval
```

**The Query Pipeline (7 bounded steps, max 3 LLM calls):**
```
NL Question
  ↓ [1] Schema Retrieval     (SQLite cache → miss → introspect DB)
  ↓ [2] Table Selector       (LLM call #1: which tables are relevant?)
  ↓ [3] Few-Shot Retrieval   (cosine similarity over query_embeddings)
  ↓ [4] SQL Generator        (LLM call #2: generate SQL with schema + examples)
  ↓ [5] SQL Validator        (DETERMINISTIC: sqlglot AST, no LLM)
  ↓ [6] Executor             (SQLAlchemy, 15s timeout, 500 row cap)
  ↓ [7] Answer Generator     (LLM call #3: explain results in persona voice)
```

**Dependencies to add:**
```
sqlalchemy>=2.0.0
cryptography>=41.0.0    # Fernet for credential encryption
sqlglot>=23.0.0         # AST-based SQL validation (NOT regex)
psycopg2-binary>=2.9.0  # PostgreSQL (add later; SQLite needs nothing)
pymysql>=1.1.0          # MySQL (add later)
```

---

### Phase 2: Schema Intelligence + Safety (1–2 weeks after Phase 1)

**Goal:** Handle real-world databases (large schemas, joins, bad column names).

**Deliverables:**
1. **Schema-Aware RAG** — at connect time, Groq generates a one-sentence description of each table from column names + sample values. These are embedded and stored. Future queries retrieve the most semantically relevant tables — not just keyword matches on names.
   - Solves cryptic column names (`cust_ord_hdr_rec` → "Customer order header records")
   - Handles 100+ table databases without context overflow
2. **Full SQL Validator** — `sqlglot` AST walking, table whitelist, dialect-aware `LIMIT` injection
3. **Read-only enforcement** — PostgreSQL: `SET TRANSACTION READ ONLY`; MySQL: `SET SESSION TRANSACTION READ ONLY`; Universal fallback: wrap in rolled-back transaction
4. **Data Analyst Persona** — 8th persona added to `EXPERT_PERSONAS`:
   - Domain: SQL, BI analysis, KPI tracking, cross-source reconciliation
   - Always shows SQL + plain-English query explanation
   - Flags data quality issues (NULLs, outliers, unexpected values)
5. **Schema Browser UI** — collapsible table explorer in the sidebar

---

### Phase 3: Full Hybrid + Discrepancy Detection (2–3 weeks after Phase 2)

**Goal:** The killer feature. Questions spanning PDFs and live database in one synthesized answer.

**Deliverables:**
1. **Query Intent Classifier** — single cheap LLM call: "Is this DB only, docs only, or hybrid?"
2. **Context Fusion Engine** — parallel async execution of RAG retrieval + SQL execution, then unified synthesis
3. **Discrepancy Detection** — when doc projections differ from DB actuals, the answer explicitly flags it:
   > *"Your board deck projected $2.4M ARR (Source: Q3_deck.pdf, Page 7). Your transactions database shows $2.1M actual — 87.5% attainment with 3 weeks remaining."*
4. **Dual Citation System** — `[Source: file.pdf, Page X]` for docs + `[DB: table_name]` for database results
5. **Streaming responses** — Groq streaming for answer generation to stay within Vercel's 30s limit

**Hybrid Fusion Prompt Pattern:**
```
{persona_def}

Document context:
{retrieved_doc_chunks_with_citations}

Database query result (SQL: {generated_sql}):
{formatted_result_table}

Answer using BOTH sources: {user_question}
When documents and database data agree, state it.
When they disagree, highlight it prominently — this is usually the most valuable insight.
Cite documents as [Source: file, Page X] and database as [DB: table_name].
```

---

### Phase 3b: Smart Agent Auto-Routing (EPIC-08, 18 points)

**Goal:** Make expert personas feel like genuinely different agents — not just system prompt swaps. Auto-route questions to the right expert, enforce structured output per agent, render each agent's output distinctively.

**Problem:** Current persona system uses identical LLM pipelines for all 8 personas. User must manually pre-select before asking. Responses look and feel the same regardless of persona selected.

**Deliverables:**
1. **Structured Output Contracts** (DOCBOT-801) — Each persona gets `required_sections`, `detection_keywords`, `tool_preference`, `output_conventions`. `persona_def` extended with OUTPUT FORMAT CONTRACT block enforcing section order.
2. **Client-Side Question Routing** (DOCBOT-802) — `routeQuestion()` weighted keyword scorer (zero LLM calls, <1ms). Auto-mode state. Tags each response with `agentPersona`.
3. **Sidebar Auto/Override UX** (DOCBOT-803) — Replace 2×4 persona grid with AUTO/Manual toggle. Grid becomes collapsible Manual Override.
4. **Agent Badge on Messages** (DOCBOT-804) — Colored pill showing which agent answered, using per-persona `accent_color`.
5. **Per-Agent Response Rendering** (DOCBOT-805) — Finance: amber metric tables. Lawyer: red risk highlights. Doctor: green disclaimer callout. Data Analyst: collapsible SQL block.

**Key design constraint:** No extra LLM calls for routing. No new backend services. `persona: string` API contract unchanged — routing is purely client-side.

**Tickets:** DOCBOT-801 (SCRUM-400) through DOCBOT-805 (SCRUM-404)

---

### Phase 4: Commerce Connector Layer (post Phase 3 + discovery confirmation)

**Goal:** Vertical specialization for mid-market Amazon sellers and agencies managing 20–50 brands.
Build only after general DB chat is live and at least 3 of 5 discovery interviews confirm the seller segment.

**ICP for this phase:** Mid-market Amazon sellers ($1M–$20M ARR) and agencies. NOT solo sellers.

**Key architectural constraint:** Amazon Reports API is async (1–45 min latency). Never query SP-API on-demand. Always sync → PostgreSQL → existing SQL pipeline queries Postgres. The 7-step pipeline sees no change — it gains new normalized tables.

**Deliverables:**
1. Pluggable `MarketplaceConnector` ABC interface + `ConnectorRegistry` (adding any future marketplace = implement ABC, register)
2. Credential vault extension — OAuth token lifecycle (LWA refresh tokens, 60-min access token rotation) on top of existing Fernet encryption
3. Unified commerce schema in PostgreSQL: `orders`, `products`, `order_line_items`, `inventory_snapshots`, `marketplace_connections` — with JSONB `raw_attributes` to preserve 100% of marketplace-specific data
4. Row-level security (RLS) policies for multi-tenant isolation (no separate schemas per tenant)
5. Amazon SP-API connector: Orders + Finances APIs in Phase A; Advertising API (separate auth domain) in Phase B
6. Background sync worker (APScheduler now → ARQ + Redis migration path when >10 connected stores)
7. Shopify connector (second implementation of the same ABC interface — ~200 lines)
8. PostgreSQL materialized views for caching (`mv_daily_sales`, `mv_product_revenue`) refreshed via `REFRESH MATERIALIZED VIEW CONCURRENTLY` — no Redis needed yet

**Commerce Vertical ICP Evolution:**
```
Stage 1 (PLG):   Solo seller → 1 store, self-serve, $99–299/month
Stage 2 (Team):  Small brand team → multi-user, shared dashboards, $299–599/month
Stage 3 (Agency): Managing 20+ brands → white-label reporting, $1,000–3,000/month
Stage 4 (Enterprise): Aggregators with 50+ brands → custom contract, SSO
```

**Connector interface (adding Shopify after Amazon = implement interface, not rewrite):**
```python
class MarketplaceConnector(ABC):
    @abstractmethod
    async def validate_credentials(self, credentials: dict) -> bool: ...
    @abstractmethod
    async def refresh_token(self, credentials: dict) -> dict: ...
    @abstractmethod
    async def fetch_orders_incremental(self, credentials, cursor, limit) -> tuple[list, SyncCursor | None]: ...
    @abstractmethod
    async def fetch_products_incremental(self, credentials, cursor, limit) -> tuple[list, SyncCursor | None]: ...
    @abstractmethod
    async def fetch_inventory(self, credentials, skus=None) -> list: ...
```

**New file structure:**
```
api/
  connectors/
    __init__.py            -- register all connectors
    base.py                -- MarketplaceConnector ABC, dataclasses, exceptions
    registry.py            -- ConnectorRegistry
    credential_service.py  -- encrypt/decrypt, OAuth token lifecycle
    rate_limiter.py        -- TokenBucket per-credential (Amazon: 0.0167 req/s Orders)
    status_maps.py         -- normalize_order_status across marketplaces
    sync_jobs.py           -- APScheduler setup, sync_marketplace_incremental
    amazon_sp.py           -- AmazonSPConnector (python-amazon-sp-api for Reports; httpx for Orders)
    shopify.py             -- ShopifyConnector (Phase B)
  migrations/
    004_marketplace_credentials.sql
    005_unified_commerce_schema.sql
    006_rls_policies.sql
    007_materialized_views.sql
```

---

## ARCHITECTURE DECISIONS

### What NOT to use

| Rejected Approach | Reason |
|-------------------|--------|
| `create_sql_agent()` (LangChain) | Unbounded LLM loops, 3-8 calls typical, unpredictable latency |
| `vanna` package | Pulls 150MB of deps (Chromadb, fastembed, web server) |
| `llama-index` | Parallel dep tree conflicts with LangChain; pick one |
| Regex for SQL validation | Trivially bypassed by `SE/**/LECT` or comment injection |
| In-memory VECTOR_STORES for schema | Lost on Vercel cold start — must use SQLite |
| Separate vector DB (Chroma, Pinecone) | Overkill; SQLite + JSON embedding storage handles personal scale |

### What to use

| Component | Technology | Why |
|-----------|------------|-----|
| SQL execution | `langchain_community.utilities.SQLDatabase` | Already in stack, handles SQLAlchemy boilerplate |
| SQL validation | `sqlglot` AST parser | Dialect-aware, bypasses regex limitations |
| Credential storage | `cryptography.fernet.Fernet` | Symmetric encryption, key in Vercel env vars |
| Schema cache | SQLite (existing `/tmp/docbot_sessions.db`) | No new infrastructure needed |
| Few-shot embeddings | Same `all-MiniLM-L6-v2` model + existing `get_embeddings()` | Reuse cached singleton |
| LLM — SQL + synthesis | `llama-3.3-70b-versatile` via Groq | Already in stack; achieves ~82% on business queries with structured prompts |
| LLM — Python code gen | `qwen/qwen3-32b` via Groq | Free coding model for E2B analysis code generation |
| LLM — Financial extraction | `gemini-2.5-flash` via Google LangExtract | Full-document chunked extraction with char_interval source grounding; free tier (10 RPM), max_workers=2 |

### Security Model (3 independent layers)

```
Layer 1: LLM prompt    "RULES: Read-only queries ONLY"
Layer 2: sqlglot AST   Reject any non-SELECT root statement
Layer 3: DB transaction SET TRANSACTION READ ONLY (PostgreSQL/MySQL)
```

All three must be independently bypassed for a write operation to succeed. This is not achievable through prompt injection.

**SSRF Prevention:** The `DBConnectionRequest` validator blocks all RFC 1918 private addresses, loopback, link-local, and IPv6 ULA. Prevents DocBot from being used as a proxy to internal VPC services.

**Credential protection:** Connection strings with passwords are NEVER:
- Logged anywhere
- Passed to the LLM context
- Stored in plain text

Only the Fernet-encrypted blob is stored in SQLite. The encryption key lives in Vercel env vars only.

---

## CRITICAL VERCEL CONSTRAINTS

The current architecture's known limits:

| Constraint | Limit | Impact on DB Feature |
|------------|-------|---------------------|
| Function execution time | 30s | Hybrid chat worst-case is exactly 30s — must use streaming |
| Writable filesystem | `/tmp` only | Schema cache must be SQLite in `/tmp` |
| Bundle size | 250MB | `psycopg2-binary` + `sqlglot` adds ~13MB — safe, but monitor |
| Process memory | Per-invocation | `VECTOR_STORES` dict already lost on cold start; DB follows same pattern |
| Cold start frequency | High on free tier | Schema introspection must be cached aggressively |

**Mitigation for 30s limit:** Use Groq's streaming API for the final answer generation step. Return `StreamingResponse` from FastAPI. The `StreamingResponse` import already exists in `api/index.py`.

**Long-term:** If DB feature gains traction, move FastAPI to Railway/Render/Fly.io for persistent processes and real connection pooling. Next.js stays on Vercel.

---

## TARGET USERS (The Three Personas)

### Maya — Finance Manager, SaaS Startup (30–150 employees)
- **Pain:** Manually reconciles board deck projections vs. actual database numbers. 6-8 hours/month. Fails to answer follow-up questions in board meetings.
- **Use:** Uploads Q3 board deck + connects company PostgreSQL → "Are we hitting the revenue target we committed to?"
- **Primary Persona:** Finance Expert

### Raj — Operations Manager, Healthcare Clinic (10–80 staff)
- **Pain:** Cannot get custom reports from EMR without paying the IT vendor. Makes decisions on gut feel.
- **Use:** Uploads clinical protocol PDFs + connects MySQL → "Show all patients where follow-up was missed per the 7-day protocol window"
- **Primary Persona:** Doctor

### Sarah — Strategy Analyst, Consulting Firm (15–75 people)
- **Pain:** Manually searches Confluence + CSVs + old decks to find relevant past work for new proposals. 3-4 hours per pitch.
- **Use:** Uploads 5 case study PDFs + connects project database → "Find past healthcare projects above target margin and explain what made them successful"
- **Primary Persona:** Consultant

### Alex — Amazon Seller / Brand Owner ($2M–$15M GMV) *(Phase 4 ICP)*
- **Pain:** Revenue is up but profit is down and he doesn't know why. His data is spread across Seller Central (orders), the Ads console (ACOS), and a supplier spreadsheet (COGS). No tool unifies all three in one answer. Manually reconciling takes 3+ hours per month and still misses cross-source insights.
- **Use:** Connects SP-API credentials → uploads supplier contracts PDF → asks "Which SKUs are profitable after fees and ad spend, and am I on track with the margin targets I projected in Q4?"
- **Primary Persona:** Data Analyst (8th persona, auto-selected on marketplace connect)
- **Segment entry point:** Phase 4
- **Agency escalation:** An agency managing 20+ brands like Alex pays $1,000–3,000/month and is the enterprise wedge into the commerce vertical

---

## THE 60-SECOND DEMO SCRIPT

> "I'm a Finance Manager. My board wants to know if we're on track for Q3.
>
> I drag in our board deck. [pause] I drop in our transactions database. [pause]
>
> One question: 'Are we on track to hit the Q3 revenue target from our board presentation?'
>
> [answer loads]
>
> It read page 7 of the deck, queried our actual transactions, calculated the gap, and told me — in plain English — that we're at 87.5% attainment with 3 weeks left.
>
> Every number has a source. The deck page for the target. The SQL query for the actuals.
>
> That answer used to take me 2 hours. I got it in 8 seconds."

---

## SUCCESS METRICS (60-90 days post-launch)

| Metric | Target | Signal If Miss |
|--------|--------|----------------|
| DB Session Activation Rate | 30% of new users try connecting a DB | Positioning or onboarding is failing |
| Hybrid Query Rate | 40% of DB sessions ask a hybrid question | Hybrid value isn't landing; need better discovery |
| Query Trust Rate | 70% of shown SQL queries are executed (not abandoned) | Accuracy or transparency problem |
| 7-Day Return Rate for DB users | 45% return within 7 days | Solved a one-time problem, not a workflow |
| Organic Referral Rate | 25% of new DB users arrive via word-of-mouth | Demo story isn't shareable enough |

---

## RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Users won't share production DB credentials | HIGH | CRITICAL | **Ship SQLite file upload first.** No credentials needed. Proves value before asking for trust. |
| SQL hallucination (valid query, wrong answer) | HIGH | HIGH | Always show SQL + plain English explanation. User verifies before trusting. |
| Vercel 30s timeout on hybrid queries | HIGH | MEDIUM | Use streaming responses for answer generation step. |
| Vanna.ai ships document integration | MEDIUM | HIGH | The persona depth creates the moat. Generic RAG + SQL ≠ Finance Expert reasoning. Move fast. |
| Target users' data is in Google Sheets, not SQL | MEDIUM | CRITICAL | **Validate in discovery interviews.** If true, build Google Sheets connector instead. Hybrid concept unchanged. |
| PII exposure in query results | MEDIUM | HIGH | Add optional column-level masking via env var config. Non-technical users shown PII warning on connect. |
| Helium 10 / Jungle Scout adds AI chat on existing data | MEDIUM | HIGH | They have the data moat for Amazon-native data. DocBot's counter: multi-source (Amazon + Shopify + internal docs) synthesis is architecturally harder than a chat wrapper; move to Phase 4 before they pivot. |
| Amazon SP-API auth complexity blocks Phase 4 | MEDIUM | MEDIUM | Budget 12 dev days for auth + sync infrastructure before any AI features. Do not underestimate LWA OAuth + IAM setup per seller. |

---

## IMMEDIATE NEXT STEPS (Ordered)

1. **This week:** Run 5 discovery interviews. Validate SQL access + hybrid use case.
2. **Week 2:** Based on interviews, confirm SQLite-first approach or pivot to Google Sheets.
3. **Week 3–4:** Build Phase 1 (SQLite upload + basic DB chat + simplified hybrid).
4. **Week 5:** Ship to 3-5 early users from interviews. Watch Query Trust Rate.
5. **Week 6:** Add `psycopg2`/`pymysql` for connection string support after trust UX is validated.
6. **Week 7–8:** Phase 2 (schema RAG + validator + Data Analyst persona).
7. **Week 9–11:** Phase 3 (full hybrid + discrepancy detection + streaming).
8. **Post Phase 3:** ProductHunt launch with the 60-second hybrid demo GIF.
9. **Phase 4 gate:** If ≥3 of 5 discovery interviews identify sellers/commerce teams as the segment, begin Phase 4 (Commerce Connector Layer). Add Q6 to every discovery interview: *"If DocBot could connect directly to your Amazon Seller Central or Shopify store and let you ask questions across your orders, fees, and ad data alongside any documents you upload — would that change how you think about this tool?"*
10. **Phase 4 sequence:** Connector interface + credential vault → unified schema + RLS → Amazon SP-API (Orders + Finances) → background sync worker → Shopify connector.
