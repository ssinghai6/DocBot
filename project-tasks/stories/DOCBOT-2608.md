# DOCBOT-2608 Deployment-options + data-flow page
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P1 | 1 | PM | 11-24 | In progress (EPIC-26, due by 12-01) |

## User story
As a fractional CFO deciding whether to upload a client's board pack and ledger export, I want a diagram showing exactly what leaves DocBot (and to whom) and what stays local, so that I can tell my client where their data went.

## Why now
Simulated panel: 18 of 20 buyers passed on trust and 19 of 20 named "show me where the data goes" as what would flip them. The diagram feeds the trust page (2531, due 11-25) and the deck privacy slide (2602).

## Scope
In:
- Data-flow diagram of today's hosted product: browser to Vercel to FastAPI on Railway; outbound LLM calls to Groq (primary) and Gemini (fallback, plus LangExtract); code execution in E2B; local embeddings (all-MiniLM-L6-v2 via ONNX) and local reranker; Fernet-encrypted connection credentials in Postgres; read-only DB access
- Three tiers table: Hosted (now), Private model mode (next, DOCBOT-2609), Single-tenant in your AWS (roadmap, docbot-platform)
- Sub-processor list with data type sent to each
Out:
- Building 2609
- Any claim that data never leaves the box today

## Acceptance criteria
- [ ] Given the diagram, when a reviewer traces each outbound arrow, then each matches a call site in code: `api/utils/llm_provider.py` (Groq, Gemini), `api/document_extractor.py` (Gemini), `api/sandbox_service.py` (E2B), and HF only when `EMBEDDINGS_PROVIDER=hf` or `RERANKER_PROVIDER=hf`
- [ ] Given the "stays local" box, when checked against CLAUDE.md, then it lists only local ONNX embeddings, local reranker and Fernet-encrypted credentials
- [ ] Given the tiers table, when read, then tier 2 says "next" and does not say "demoed" or "available"
- [ ] Given the page, when reused, then the same diagram file is referenced by the 2531 trust page and the 2602 privacy slide (one source, no forks)
- [ ] Given the page, when grepped, then no "offline", "air-gapped" or "enterprise-grade" claim appears for tier 1

## Deliverable and approach
`project-tasks/preseed/2608-deployment-options.md` plus `project-tasks/preseed/2608-data-flow.svg`. The SVG is copied into `public/` only when 2531 builds the security page. Inputs: CLAUDE.md architecture and env table, `api/utils/llm_provider.py`, `api/utils/embeddings.py`, sub-processor list from parked story 2702 (Vercel, Groq, Gemini, E2B, Stripe, Resend, Sentry; Railway until cutover 2519).

## Test / verification
Senior-dev agent reviews arrows against code (grep for `groq`, `genai`, `e2b`, `huggingface` in `api/`); PM checks tier wording against 2601.

## Dependencies
Blocked by: none  Blocks: 2531, 2602, 2605

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Does PII masking (`api/utils/pii_masking.py`) run before LLM calls or only on responses? The diagram should say which; verify in code before publishing.
