# DOCBOT-2609 Private model mode (OpenAI-compatible LLM endpoint)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 1 | S4 | P1 | 5 | senior-dev | **after 12-01** (moved 2026-11-17; mid-January slot in the master backlog). Until it ships, the deck says private model mode is next, not demoed | Code moved after 12-01 (2026-11-17) |

## User story
As a CPA firm partner whose engagement letters forbid sending client ledgers to third-party AI APIs, I want DocBot to send every model call to an OpenAI-compatible server I run (Ollama or vLLM), with no fallback to Groq or Gemini, so that I can use the tie-out on client data without client data leaving my network.

## Why now
Privacy is the most common objection in the ICP, and the trust page (2608) already describes a "Private model" tier. Until this ships, copy must say "next", never "demoed". Code was moved after 12-01; S4 is the slot.

## Scope
In:
- Env `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` (optional) in `api/utils/llm_provider.py`; when `LLM_BASE_URL` is set, every chat call goes there.
- Groq to Gemini fallback disabled in this mode; errors surface instead.
- Other outbound paths closed in this mode: LangExtract/Gemini extraction, HF embeddings and reranker, E2B sandbox, LangSmith.
- Offline demo on the demo dataset with the network off.
Out:
- Per-org endpoints and eval-gated model list (3202).
- A self-hosted Python sandbox to replace E2B (3203); Python and CSV analysis report "unavailable in private mode".
- Automated per-tier egress test matrix (3201).

## Acceptance criteria
- [ ] Given `LLM_BASE_URL` set and a mocked OpenAI-compatible server (`httpx.MockTransport`), when `chat_completion`, `chat_completion_stream`, `achat_completion`, `achat_completion_stream`, `call_llm` and `get_llm` are called, then all requests hit `{LLM_BASE_URL}/chat/completions` with `LLM_MODEL`, and the `groq.Groq` and Gemini clients are never constructed (test patches them to raise).
- [ ] Given private mode and a 500 or timeout from the endpoint, when a chat call fails, then the error propagates (or a clear SSE `error` event is sent) and no Groq or Gemini call is attempted.
- [ ] Given private mode, when a user runs Python analysis or a CSV query, then `run_python` and `run_csv_query_on_e2b` return "Python analysis is unavailable in private mode" without importing or calling E2B.
- [ ] Given private mode with `EMBEDDINGS_PROVIDER=hf` or `RERANKER_PROVIDER=hf` set, when the app boots, then both are forced to local and a single warning is logged; LangExtract extraction (`api/document_extractor.py`) is skipped.
- [ ] Given a laptop with the network off, Ollama running locally, and ONNX models pre-cached, when the demo dataset (`POST /api/demo/init`, `dataset="tieout"`) is run end to end, then a hybrid tie-out answer with citations is produced; a short screen recording is saved for the data room.

## Deliverable and approach
- `api/utils/llm_provider.py`: `private_mode_enabled()` reads `LLM_BASE_URL` at call time. Branch at the top of `chat_completion` (line 749), `chat_completion_stream` (line 819), `call_llm` (line 565) and `get_llm` (line 472, used by `deep_research_service.py:143`). Callers pass `model=GROQ_CODE_MODEL` or `GROQ_MODEL` explicitly (`sandbox_service.py:706, 952, 1114`); in private mode the requested model is replaced by `LLM_MODEL`.
- HTTP via `httpx` (already pinned, `requirements.txt:55`): plain POST and SSE parsing, no new dependency. A LangChain wrapper `api/utils/_openai_compat_wrapper.py` modeled on `api/utils/_gemini_wrapper.py` for `get_llm`.
- Keep `_log_llm_call` with `provider="private"`; unknown-model cost is null, which is fine.
- `api/sandbox_service.py`: guard in `run_python` (line 790) and `run_csv_query_on_e2b` (line 1473).
- `api/document_extractor.py`: skip `extract_document_fields` (line 311) in private mode.
- `api/utils/embeddings_provider.py` and `api/utils/reranker.py`: force `local` in private mode.
- Env vars documented in `CLAUDE.md` env table: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. These are operator env only, never settable via API (per-org endpoints with SSRF checks are 3202).
- Branch `feature/DOCBOT-2609-private-model-mode`.

## Test / verification
- `tests/unit/test_private_model_mode.py`: mocked transport, streaming parse, no-fallback, sandbox guard, extractor skip, provider forcing; plus a guard that patches `socket.create_connection` to fail for any host except the mock endpoint. No network.
- `pytest tests/unit/test_private_model_mode.py tests/unit/test_llm_provider.py -v`; full CI `-m "not external and not postgres"`.
- Manual offline demo as in AC 5. Optional: `python -m tests.eval.tieout_eval --mode live --runs 3` with `LLM_BASE_URL` set, reported as an internal private-model number, never mixed with the Groq number. Cost $0 (local model).

## Dependencies
Blocked by: 2608 (data-flow page defines the tiers).  Blocks: 3201, 3202, 3203; deck/trust copy change from "next" to "demoed".

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Points: 5 is tight. There are six chat entry points plus a LangChain wrapper, extractor, sandbox, embeddings and reranker gates. 8 is more realistic.
- The wedge depends on CSV exports (2577), and CSV queries run only in E2B, so in private mode the QBO/Xero CSV tie-out does not work until 3203. Either state this on the trust page, or load accounting exports for private mode another way (decision needed; `CLAUDE.md` forbids CSV on the SQL pipeline).
- Other outbound paths to decide on: Sentry (`api/utils/sentry_setup.py`, error payloads can include data) should be off or scrubbed in private mode; EDGAR calls go to SEC with public identifiers only and can stay.
- The offline demo needs the chromadb ONNX embedding model and the fastembed reranker model cached before the network is cut; add a warm-cache step to the runbook.
- Local model tie-out accuracy is unmeasured; a small open model may miss the 90% bar. Do not claim parity with the hosted number.
