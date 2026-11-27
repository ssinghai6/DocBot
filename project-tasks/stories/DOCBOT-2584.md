# DOCBOT-2584 NEW: Run gpt-oss on Amazon Bedrock (AWS credits) behind LLM_PRIMARY
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P1 | 3 | senior-dev | 2026-11-26 | In progress (2026-11-21) |

## User story
As the founder, I want DocBot's LLM calls to run on Amazon Bedrock and be paid from the $5K AWS credits, so that the demo stops hitting Groq's 8K tokens-per-minute limit and the privacy story becomes "your data stays in AWS".

## Why now
Groq on_demand 429s caused the demo hang (DOCBOT-2583). The investor call is 2026-12-01, the prod freeze starts 11-27, and the AWS credits are otherwise unused.

## Scope
In:
- `LLM_PRIMARY=bedrock` sends `chat_completion`, `chat_completion_stream` and `call_llm` to Bedrock's OpenAI-compatible Chat Completions route (`openai.gpt-oss-20b-1:0`, `openai.gpt-oss-120b-1:0`), reusing the Groq SDK with `base_url`.
- Bedrock model rates in the cost table, so `PLATFORM_DAILY_USD_CAP`, `DEMO_DAILY_USD_CAP` and the per-session ceiling still count Bedrock spend.
- Strip inline `<reasoning>` spans from Bedrock gpt-oss output.
- Unset or `groq` keeps current behaviour (instant rollback).

Out:
- LangExtract PDF extraction still uses Gemini (DOCBOT-2609).
- Groq as a second fallback after Bedrock (fallback stays Gemini).
- AWS console setup (founder only: model access, IAM user, Bedrock API key, Budget + budget action).

## Acceptance criteria
- [ ] Given `LLM_PRIMARY=bedrock` and `BEDROCK_API_KEY`, when chat runs, then requests go to `https://bedrock-runtime.<BEDROCK_REGION>.amazonaws.com/openai/v1/chat/completions` with the Bedrock model id.
- [ ] Given a Bedrock call, when it is logged, then provider is `bedrock` and `estimated_cost_usd` is non-null.
- [ ] Given Bedrock output containing `<reasoning>...</reasoning>`, when streamed or returned, then the user never sees it.
- [ ] Given `LLM_PRIMARY` unset, when chat runs, then Groq is used exactly as before.
- [ ] Golden path (`scripts/golden_path.py`) passes on prod with `LLM_PRIMARY=bedrock`.

## Deliverable and approach
`api/utils/llm_provider.py` (`_primary_target`, `_strip_reasoning_stream`); tests in `tests/unit/test_bedrock_primary.py`.

## Test / verification
`pytest tests/unit/test_bedrock_primary.py`; full suite; after the founder sets Railway env vars, run golden path on prod.

## Dependencies
Blocked by: DOCBOT-2583 (PR #132, same file).
Blocks: DOCBOT-2609.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Bedrock rates in the cost table come from aggregators; confirm on aws.amazon.com/bedrock/pricing.
- Not yet run against live Bedrock (no key in this session). The `<reasoning>` inline format is guarded for but not observed.
