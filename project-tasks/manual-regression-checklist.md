# DocBot Manual Regression Checklist

> **Target:** 85 tests across all shipped features
> **Environment:** Production (Railway backend + Vercel frontend)
> **Date:** ___________
> **Tester:** ___________
> **Overall Result:** ___ / 85 passing

---

## 1. Landing Page & Navigation (5 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 1 | `/` loads investor landing page with hero, features, CTA | | |
| 2 | "Get Started" CTA navigates to `/chat` | | |
| 3 | Landing page renders correctly on mobile (375px) | | |
| 4 | Landing page renders correctly on tablet (768px) | | |
| 5 | Demo video auto-plays muted on landing page | | |

---

## 2. PDF Upload & RAG Chat (10 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 6 | Upload a PDF -- file accepted, progress shown, session created | | |
| 7 | Ask a factual question about the PDF -- correct answer with [Source] citation | | |
| 8 | Ask a follow-up question -- conversational memory preserves context | | |
| 9 | Smart auto-routing selects appropriate persona (e.g., financial question routes to Finance Expert) | | |
| 10 | Manually switch persona via selector -- response style changes | | |
| 11 | Generalist persona: balanced response with citations | | |
| 12 | Finance Expert persona: financial terminology, computation steps | | |
| 13 | Lawyer persona: legal analysis style response | | |
| 14 | Data Analyst persona: data-focused response with tables | | |
| 15 | Ask about something NOT in the PDF -- model says "not found in document" | | |

---

## 3. Database Connectivity & SQL (10 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 16 | Connect to PostgreSQL database -- connection succeeds, schema loaded | | |
| 17 | Ask natural language question -- SQL generated, executed, answer returned | | |
| 18 | Verify SQL is read-only (try "delete all records" -- should be rejected) | | |
| 19 | Schema cache loads tables and views | | |
| 20 | Query with JOIN across multiple tables -- correct result | | |
| 21 | Ask follow-up question about DB -- conversational context maintained | | |
| 22 | Connect to MySQL database -- connection succeeds | | |
| 23 | Connect to SQLite file upload -- connection succeeds | | |
| 24 | Schema refresh endpoint works (`POST /api/db/refresh-schema/{id}`) | | |
| 25 | Bad connection string -- clear error message, no credential leak | | |

---

## 4. CSV Upload & Pandas Queries (8 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 26 | Upload CSV -- file accepted, DataProfile computed (dtypes, sample rows) | | |
| 27 | Ask simple question about CSV data -- pandas code generated, result returned | | |
| 28 | Ask for chart/visualization -- matplotlib chart rendered inline | | |
| 29 | Ask complex analytical question -- adaptive limits applied (longer timeout) | | |
| 30 | Multi-section CSV detected and handled correctly | | |
| 31 | Ask follow-up question about CSV -- conversational context maintained | | |
| 32 | Sandbox error triggers retry with feedback -- corrected result returned | | |
| 33 | Upload large CSV (1000+ rows) -- processed without timeout | | |

---

## 5. Hybrid Mode (8 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 34 | Upload PDF + connect DB -- hybrid mode auto-activates | | |
| 35 | Ask question spanning both sources -- dual citations (doc + SQL) | | |
| 36 | Discrepancy detection: conflicting data between PDF and DB flagged | | |
| 37 | Upload PDF + CSV -- hybrid mode works with both sources | | |
| 38 | Intent classifier correctly routes to doc-only vs db-only vs hybrid | | |
| 39 | Ask computation question with user-provided parameters -- calculation performed | | |
| 40 | Hybrid synthesis includes both RAG context and SQL results | | |
| 41 | Follow-up question in hybrid mode -- context preserved across sources | | |

---

## 6. Analytical Autopilot (6 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 42 | Analytical query auto-triggers Autopilot (e.g., "compare revenue trends") | | |
| 43 | Toast notification appears when Autopilot activates | | |
| 44 | Multi-step investigation: planner decomposes, executor runs steps | | |
| 45 | Deep retrieval for doc_search steps -- sub-question decomposition visible | | |
| 46 | Final synthesis uses markdown tables for numerical data | | |
| 47 | Autopilot works with CSV source (not just PDF) | | |

---

## 7. SEC EDGAR Integration (8 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 48 | EDGAR panel visible in sidebar | | |
| 49 | Search "AAPL" -- Apple Inc appears in results with CIK | | |
| 50 | Search by company name "Microsoft" -- MSFT found | | |
| 51 | Select company -- filing list loads (10-K, 10-Q options) | | |
| 52 | Change filing type dropdown -- filings update | | |
| 53 | Single ingest: click a filing -- download, chunk, embed, session created | | |
| 54 | Chat with ingested filing -- answers with citations | | |
| 55 | Batch ingest: ingest last 3 filings -- all succeed with status indicators | | |

---

## 8. Commerce Connectors (8 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 56 | Marketplace panel visible in sidebar | | |
| 57 | Type selector shows both "Amazon" and "Shopify" options | | |
| 58 | Amazon: fill credentials, click "Test & Connect" -- connector registers | | |
| 59 | Shopify: fill shop_domain + access_token, connect -- connector registers | | |
| 60 | Connected connector appears in list with type label | | |
| 61 | Sync panel: set date range, click Sync -- orders/financials fetched | | |
| 62 | Sync result shows order count and financial count | | |
| 63 | Disconnect connector -- removed from list | | |

---

## 9. Authentication & Authorization (8 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 64 | Guest mode (AUTH_REQUIRED=false): app accessible without login | | |
| 65 | Email/password signup -- account created, session started | | |
| 66 | Email/password login -- session created, redirected to chat | | |
| 67 | GitHub OAuth login -- redirects, authenticates, returns to app | | |
| 68 | Google OAuth login -- redirects, authenticates, returns to app | | |
| 69 | Session persists across page refreshes (cookie-based) | | |
| 70 | Logout -- session destroyed, redirected to login | | |
| 71 | RBAC: viewer role cannot access admin panel | | |

---

## 10. Admin Panel & Audit (5 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 72 | Admin panel loads for admin-role user | | |
| 73 | Metrics endpoint returns usage stats (`GET /api/admin/metrics`) | | |
| 74 | Audit log shows recent events with timestamps | | |
| 75 | Audit log CSV export downloads correctly | | |
| 76 | PII masking: names/emails/SSNs redacted in responses | | |

---

## 11. API Health & Infrastructure (5 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 77 | `GET /api/health` returns 200 on Railway | | |
| 78 | Frontend loads on Vercel production URL | | |
| 79 | CORS: frontend can call backend without errors | | |
| 80 | SSE streaming: chat responses stream token-by-token | | |
| 81 | LLM fallback: if Groq fails, Gemini fallback activates (check logs) | | |

---

## 12. Responsive & Cross-Browser (4 tests)

| # | Test | Pass? | Notes |
|---|------|-------|-------|
| 82 | Chat app renders correctly on desktop (1280px+) | | |
| 83 | Chat app renders correctly on tablet (768px) -- sidebar collapses | | |
| 84 | Chat app renders correctly on mobile (375px) | | |
| 85 | Sidebar sections spacing normalized, no overflow/clipping | | |

---

## Summary

| Section | Tests | Passed | Failed |
|---------|-------|--------|--------|
| 1. Landing Page & Navigation | 5 | | |
| 2. PDF Upload & RAG Chat | 10 | | |
| 3. Database Connectivity & SQL | 10 | | |
| 4. CSV Upload & Pandas Queries | 8 | | |
| 5. Hybrid Mode | 8 | | |
| 6. Analytical Autopilot | 6 | | |
| 7. SEC EDGAR Integration | 8 | | |
| 8. Commerce Connectors | 8 | | |
| 9. Authentication & Authorization | 8 | | |
| 10. Admin Panel & Audit | 5 | | |
| 11. API Health & Infrastructure | 5 | | |
| 12. Responsive & Cross-Browser | 4 | | |
| **TOTAL** | **85** | | |

---

## Sign-Off

- [ ] All P0 bugs fixed
- [ ] All P1 bugs fixed or documented with workaround
- [ ] Investor demo rehearsal completed
- [ ] **APPROVED FOR DEMO**

Signed: ___________________________ Date: ___________
