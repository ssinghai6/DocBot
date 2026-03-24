"""
Create DOCBOT-901 through DOCBOT-904 in Jira (SCRUM project).
EPIC-09: LangGraph Deep Research Mode
"""

import json
import requests
from requests.auth import HTTPBasicAuth

JIRA_URL  = "https://dbdocbot.atlassian.net"
EMAIL     = "singhai.sanshrit@gmail.com"
API_TOKEN = "ATATT3xFfGF0S9OF3iHAA0J1UW64U5DP-b4r2RG2CH8hi87OebCXdYZGTgvLYZUP-123K5XxdwPYBd0yq-dZJFwcOB06XI4jlyyrAOM5I79pMawXazXQADKRdSKNvzjdOhrWpdA8hfUf_s-rlPWE_zb04-pwQ3aJZ_PpIm6bCovbfBtamxEznPo=92D43F04"

AUTH    = HTTPBasicAuth(EMAIL, API_TOKEN)
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

# SCRUM project key
PROJECT_KEY = "SCRUM"

TICKETS = [
    {
        "docbot_id": "DOCBOT-901",
        "summary": "LangGraph Deep Research State Machine Core",
        "description": (
            "Create api/deep_research_service.py containing the full LangGraph state machine "
            "for Deep Research mode. This replaces the DEEP_RESEARCH_ADDON string-append approach "
            "with a proper multi-step reasoning graph.\n\n"
            "TASKS:\n"
            "1. Define DeepResearchState TypedDict with fields: question, session_id, persona_def, "
            "vector_store, sub_questions, retrieved_chunks, gaps, iterations, final_answer, citations.\n"
            "2. Implement query_planner node (LLM call #1): use ChatGroq llama-3.3-70b-versatile "
            "with temperature=0 to decompose the question into 3-5 focused sub-questions. "
            "Return JSON array. Include _parse_json_list() helper that handles markdown fences, "
            "malformed JSON, and falls back to [original_question] — never raises.\n"
            "3. Implement parallel_retriever node (0 LLM calls): for each sub-question call "
            "expand_query() from api/utils/query_expansion.py, run concurrent similarity searches "
            "via ThreadPoolExecutor, merge with deduplicate_docs(). On re-entry (iterations>0) "
            "only search gap sub-questions. Increment iterations counter.\n"
            "4. Implement evidence_evaluator node (0 LLM calls): deterministic — a sub-question "
            "is 'covered' if it has >= 2 unique chunks. Uncovered ones go into state.gaps.\n"
            "5. Implement gap_router conditional edge function: returns 'parallel_retriever' if "
            "gaps exist AND iterations < 2, else returns 'synthesizer'.\n"
            "6. Implement synthesizer node (LLM call #2, streaming): flatten all retrieved_chunks "
            "across sub-questions via deduplicate_docs(), compose structured answer with ## headers "
            "addressing each sub-question, cite every claim. Stream tokens into asyncio.Queue.\n"
            "7. Implement build_deep_research_graph() using StateGraph with factory-closure pattern "
            "to inject token_queue and groq_api_key into nodes.\n"
            "8. Implement _emit_progress() helper that puts progress events into token_queue.\n"
            "9. Implement run_deep_research() public coroutine: creates asyncio.Queue, builds graph, "
            "runs graph as background task, consumes queue and yields SSE-formatted strings.\n\n"
            "CONSTRAINTS:\n"
            "- Max 3 LLM calls total (2 typical: query_planner + synthesizer)\n"
            "- gap_router re-entry reruns parallel_retriever (0 LLM calls), capped at iterations < 2\n"
            "- Never pass connection strings or API keys into LLM context\n"
            "- asyncio.Queue is not stored in graph state (closures only)\n"
            "- _parse_json_list must never raise\n"
            "- Use specific exception types, no bare except\n\n"
            "SSE progress event shapes:\n"
            "  {\"type\": \"progress\", \"step\": \"planning\", \"message\": \"Breaking down your question...\"}\n"
            "  {\"type\": \"progress\", \"step\": \"retrieving\", \"message\": \"Searching: <sub_q[:60]>...\"}\n"
            "  {\"type\": \"progress\", \"step\": \"evaluating\", \"message\": \"Evaluating evidence coverage...\"}\n"
            "  {\"type\": \"progress\", \"step\": \"gap_fill\", \"message\": \"Finding more context for N topics...\"}\n"
            "  {\"type\": \"progress\", \"step\": \"synthesizing\", \"message\": \"Composing your answer...\"}\n"
            "  {\"type\": \"token\", \"content\": \"...\"}\n"
            "  {\"type\": \"citations\", \"citations\": [...]}\n\n"
            "Phase: 3 | Priority: Must Have | Points: 8 | Epic: EPIC-09"
        ),
        "story_points": 8,
    },
    {
        "docbot_id": "DOCBOT-902",
        "summary": "Streaming Queue Bridge + /api/chat Deep Research Wiring",
        "description": (
            "Wire run_deep_research() into the /api/chat SSE route in api/index.py. "
            "When ChatRequest.deep_research=True, branch to the LangGraph path instead of "
            "the existing single-shot qa_chain.astream() path.\n\n"
            "TASKS:\n"
            "1. Extract the existing event_stream() body from /api/chat into a private helper "
            "_existing_chat_stream(request, groq_api_key) that yields SSE strings. "
            "This is a pure refactor — behaviour for deep_research=False must be unchanged.\n"
            "2. Add branch in event_stream(): if request.deep_research, import run_deep_research "
            "from api.deep_research_service and call it with question=request.message, "
            "session_id, persona_def from EXPERT_PERSONAS, vector_store from VECTOR_STORES. "
            "Yield all SSE lines from the generator.\n"
            "3. Ensure DB persist (_persist()) still fires after deep research stream ends — "
            "collect final_answer from the last token events and citations from the citations event.\n"
            "4. Ensure error handling: any exception from run_deep_research() must emit "
            "{\"type\": \"error\", \"detail\": safe_error_message(e)} and not crash the worker.\n"
            "5. Verify that the new SSE event type 'progress' does not break existing frontend "
            "SSE parsing (it must be ignored gracefully if the frontend does not yet handle it).\n\n"
            "ACCEPTANCE CRITERIA:\n"
            "- curl /api/chat with deep_research=true receives progress events then token events then citations\n"
            "- curl /api/chat with deep_research=false receives identical output to pre-change behavior\n"
            "- No new Pydantic model changes needed (ChatRequest.deep_research: bool already exists)\n\n"
            "Phase: 3 | Priority: Must Have | Points: 5 | Epic: EPIC-09 | Depends: DOCBOT-901"
        ),
        "story_points": 5,
    },
    {
        "docbot_id": "DOCBOT-903",
        "summary": "Frontend Deep Research Progress Strip UI",
        "description": (
            "Add a progress indicator strip in src/app/page.tsx that renders during Deep Research "
            "mode, showing which step the graph is currently executing.\n\n"
            "TASKS:\n"
            "1. Add DeepResearchProgressState type: {currentStep: ProgressStep | null, "
            "currentMessage: string, stepHistory: ProgressStep[]}. "
            "ProgressStep = 'planning'|'retrieving'|'evaluating'|'gap_fill'|'synthesizing'|'done'.\n"
            "2. Add deepResearchProgress state via useState, initialised to null.\n"
            "3. In the SSE reader loop, add case for parsed.type === 'progress': update "
            "deepResearchProgress state with step and message.\n"
            "4. On first parsed.type === 'token': set deepResearchProgress.currentStep = 'done' "
            "to collapse the progress strip.\n"
            "5. Implement DeepResearchProgress component (inline or separate): renders only when "
            "deepResearchProgress is not null AND currentStep !== 'done'. "
            "Shows icon (Brain/Search/CheckSquare/RefreshCw/FileText from lucide-react) + message text "
            "+ step dots (5 dots: filled indigo for completed steps, outlined for pending).\n"
            "6. Strip styling: bg-indigo-50 dark:bg-indigo-950/30, rounded-lg, p-3, text-sm. "
            "Active icon has animate-pulse class. Strip fades out with transition-opacity when "
            "currentStep becomes 'done'.\n"
            "7. On mobile (< 640px via sm: prefix): hide message text, show only icon + dots.\n"
            "8. Reset deepResearchProgress to null on each new message send (in handleSend before "
            "the fetch call).\n"
            "9. Render the strip above the streaming response bubble, inside the assistant message "
            "container, only for messages where deep_research was active.\n\n"
            "ACCEPTANCE CRITERIA:\n"
            "- Progress strip appears immediately after send when deep_research toggle is on\n"
            "- Steps light up in sequence: planning → retrieving → evaluating → (gap_fill if triggered) → synthesizing\n"
            "- Strip collapses smoothly when first answer token arrives\n"
            "- No TypeScript any; use Zod or explicit types for SSE event parsing\n"
            "- Works at 375px and 1280px viewports\n\n"
            "Phase: 3 | Priority: Must Have | Points: 3 | Epic: EPIC-09 | Depends: DOCBOT-902"
        ),
        "story_points": 3,
    },
    {
        "docbot_id": "DOCBOT-904",
        "summary": "Unit and Integration Tests for Deep Research Service",
        "description": (
            "Write the full test suite for api/deep_research_service.py.\n\n"
            "UNIT TESTS (tests/unit/test_deep_research_service.py) — no network, no API keys:\n"
            "1. test_parse_json_list_valid — well-formed JSON array returns correct list\n"
            "2. test_parse_json_list_with_fences — markdown ```json fences stripped before parse\n"
            "3. test_parse_json_list_fallback — malformed JSON returns [original_question], no raise\n"
            "4. test_evidence_evaluator_identifies_gaps — sub-question with 0 docs added to gaps\n"
            "5. test_evidence_evaluator_no_gaps — sub-questions each with 2+ docs → gaps=[]\n"
            "6. test_gap_router_loops_when_gaps_and_under_limit — returns 'parallel_retriever'\n"
            "7. test_gap_router_proceeds_at_max_iterations — returns 'synthesizer' at iterations=2\n"
            "8. test_gap_router_proceeds_when_no_gaps — returns 'synthesizer' with gaps=[]\n\n"
            "INTEGRATION TESTS (tests/integration/test_deep_research_pipeline.py) — uses in-memory "
            "vector store, mocked ChatGroq, no real API calls:\n"
            "1. test_full_graph_emits_events_in_order — runs graph end-to-end with mocked LLM, "
            "verifies progress events in order: planning, retrieving, evaluating, synthesizing, "
            "then citations event, then None sentinel.\n"
            "2. test_gap_fill_loop_triggers — seed vector store so 2/3 sub-questions get <2 chunks; "
            "verify gap_fill progress event emitted and iterations incremented.\n\n"
            "FIXTURES (add to tests/conftest.py if not present):\n"
            "- mock_groq_llm: AsyncMock for ChatGroq.astream() that yields token chunks\n"
            "- tiny_vector_store: InMemoryVectorStore with 5 known Documents, no HuggingFace call\n\n"
            "RULES:\n"
            "- All unit tests: no network calls (mock everything with unittest.mock)\n"
            "- Integration tests with mocked LLM: not marked external, run in CI\n"
            "- Any test requiring real Groq or HuggingFace: mark @pytest.mark.external\n"
            "- Run command: pytest tests/unit/test_deep_research_service.py -v\n\n"
            "Phase: 3 | Priority: Must Have | Points: 3 | Epic: EPIC-09 | Depends: DOCBOT-901"
        ),
        "story_points": 3,
    },
]


def get_project_id() -> str:
    r = requests.get(
        f"{JIRA_URL}/rest/api/3/project/{PROJECT_KEY}",
        auth=AUTH, headers={"Accept": "application/json"}
    )
    r.raise_for_status()
    return r.json()["id"]


def create_issue(ticket: dict, project_id: str) -> "str | None":
    payload = {
        "fields": {
            "project": {"id": project_id},
            "summary": f"[{ticket['docbot_id']}] {ticket['summary']}",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": ticket["description"]}]
                    }
                ]
            },
            "issuetype": {"name": "Story"},
        }
    }

    r = requests.post(
        f"{JIRA_URL}/rest/api/3/issue",
        auth=AUTH, headers=HEADERS,
        data=json.dumps(payload)
    )

    if r.status_code == 201:
        key = r.json()["key"]
        print(f"  Created {key} — {ticket['docbot_id']}: {ticket['summary']}")
        return key
    else:
        print(f"  Failed {ticket['docbot_id']}: {r.status_code} {r.text[:300]}")
        return None


def main():
    print("Creating DOCBOT-901 through DOCBOT-904 in Jira...\n")

    project_id = get_project_id()
    print(f"Project ID: {project_id}\n")

    created = []
    for ticket in TICKETS:
        key = create_issue(ticket, project_id)
        if key:
            created.append((key, ticket["docbot_id"]))

    print(f"\n{'='*50}")
    print(f"Created {len(created)}/{len(TICKETS)} tickets")
    for jira_key, docbot_key in created:
        print(f"  {jira_key} -> {docbot_key}")
    print("\nDone. Add the SCRUM keys above to jira_update_status.py for future Done transitions.")


if __name__ == "__main__":
    main()
