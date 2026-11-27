"use client"

// DOCBOT-3905: 4-step quickstart (key -> sandbox -> hybrid question -> lineage).
// Each step runs in the browser against the same-origin /v1 rewrite with the
// in-memory test key, and shows the equivalent curl for the user's own code.

import { useState } from "react"
import { apiCall } from "@/lib/developerApi"
import type { ParsedApiError } from "@/lib/schemas/developer"
import {
  V1HybridChatRequestSchema,
  V1HybridChatResponseSchema,
  type V1HybridChatResponse,
} from "@/lib/schemas/v1"
import {
  V1SandboxSessionRequestSchema,
  V1SandboxSessionResponseSchema,
  type V1SandboxSessionResponse,
} from "@/lib/schemas/v1Sandbox"
import { CodeBlock, ErrorNote, btnPrimaryCx, cardCx } from "./ui"

const DEFAULT_QUESTION = "Are there any discrepancies between the filing and the database?"

function Step({ n, title, done, children }: { n: number; title: string; done: boolean; children: React.ReactNode }) {
  return (
    <li className="relative pl-10">
      <span
        aria-hidden="true"
        className={`absolute left-0 top-0 flex h-7 w-7 items-center justify-center rounded-full border text-xs font-semibold ${
          done ? "bg-[var(--ap-accent-strong)] border-transparent text-[var(--ap-on-accent)]" : "border-[var(--ap-border)]"
        }`}
      >
        {done ? "✓" : n}
      </span>
      <h3 className="text-base font-semibold">
        {title}
        {done && <span className="sr-only"> (done)</span>}
      </h3>
      <div className="mt-1 text-sm text-[var(--ap-muted)]">{children}</div>
    </li>
  )
}

export default function Quickstart({ testKey }: { testKey: string | null }) {
  const host = typeof window === "undefined" ? "https://docbot.space" : window.location.origin
  const [sandbox, setSandbox] = useState<V1SandboxSessionResponse | null>(null)
  const [question, setQuestion] = useState(DEFAULT_QUESTION)
  const [answer, setAnswer] = useState<V1HybridChatResponse | null>(null)
  const [busy, setBusy] = useState<"sandbox" | "ask" | null>(null)
  const [error, setError] = useState<ParsedApiError | null>(null)

  async function createSandbox() {
    if (!testKey) return
    setBusy("sandbox")
    setError(null)
    const res = await apiCall(
      "/v1/sandbox/sessions",
      { method: "POST", bearer: testKey, body: JSON.stringify(V1SandboxSessionRequestSchema.parse({ dataset: "quickbite" })) },
      V1SandboxSessionResponseSchema,
    )
    setBusy(null)
    if (!res.ok) return setError(res.error)
    setSandbox(res.data)
    setAnswer(null)
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault()
    if (!testKey || !sandbox) return
    if (!question.trim()) return setError({ code: null, message: "Enter a question.", requestId: null })
    const body = V1HybridChatRequestSchema.parse({
      question: question.trim(),
      session_id: sandbox.session_id,
      connection_id: sandbox.connection_id,
    })
    setBusy("ask")
    setError(null)
    const res = await apiCall(
      "/v1/hybrid/chat",
      { method: "POST", bearer: testKey, body: JSON.stringify(body) },
      V1HybridChatResponseSchema,
    )
    setBusy(null)
    if (!res.ok) return setError(res.error)
    setAnswer(res.data)
  }

  const sessionId = sandbox?.session_id ?? "<session_id>"
  const connectionId = sandbox?.connection_id ?? "<connection_id>"
  const runId = answer?.run_id ?? "<run_id>"

  return (
    <section aria-labelledby="qs-heading" className={cardCx}>
      <h2 id="qs-heading" className="text-lg font-semibold">Quickstart: your first hybrid answer</h2>
      <p className="text-sm text-[var(--ap-muted)]">
        About 5 minutes. Run each step here, or copy the curl into your terminal.
      </p>

      <ol className="mt-5 space-y-6">
        <Step n={1} title="Get a test key" done={Boolean(testKey)}>
          {testKey ? "Key ready. Export it for curl:" : "Use “Get test key” above, then export it for curl:"}
          <CodeBlock label="export key command" code={`export DOCBOT_API_KEY="<paste your dk_test_ key>"\nexport HOST="${host}"`} />
        </Step>

        <Step n={2} title="Create a sandbox session" done={Boolean(sandbox)}>
          A demo 10-K plus a matching SQL ledger (QuickBite Inc.) with deliberate discrepancies.
          <CodeBlock
            label="create sandbox curl"
            code={`curl -sS -X POST "$HOST/v1/sandbox/sessions" \\\n  -H "Authorization: Bearer $DOCBOT_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{"dataset": "quickbite"}'`}
          />
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <button type="button" className={btnPrimaryCx} onClick={createSandbox} disabled={!testKey || busy !== null}>
              {busy === "sandbox" ? "Creating…" : "Run step 2"}
            </button>
            {sandbox && (
              <span className="text-xs">
                session_id <code className="font-mono text-[var(--ap-text)]">{sandbox.session_id}</code>
              </span>
            )}
          </div>
        </Step>

        <Step n={3} title="Ask a hybrid question" done={Boolean(answer)}>
          DocBot reads the filing and queries the ledger in one call, then flags mismatches.
          <CodeBlock
            label="hybrid chat curl"
            code={`curl -sS -X POST "$HOST/v1/hybrid/chat" \\\n  -H "Authorization: Bearer $DOCBOT_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify({ question, session_id: sessionId, connection_id: connectionId })}'`}
          />
          <form onSubmit={ask} className="mt-3 flex flex-col gap-2 sm:flex-row">
            <label htmlFor="qs-question" className="sr-only">
              Question
            </label>
            <input
              id="qs-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="flex-1 min-w-0 h-9 rounded-md border border-[var(--ap-border)] bg-[var(--ap-bg)] px-3 text-sm text-[var(--ap-text)]"
            />
            <button type="submit" className={btnPrimaryCx} disabled={!sandbox || busy !== null}>
              {busy === "ask" ? "Asking… (up to a minute)" : "Run step 3"}
            </button>
          </form>
          {answer && (
            <div className="mt-3 rounded-lg border border-[var(--ap-border)] bg-[var(--ap-bg)] p-3 text-[var(--ap-text)]">
              <p className="whitespace-pre-wrap text-sm">{answer.answer}</p>
              <p className="mt-2 text-xs text-[var(--ap-muted)]">
                intent {answer.intent ?? "n/a"} · docs {answer.has_docs ? "yes" : "no"} · SQL {answer.has_sql ? "yes" : "no"} ·{" "}
                {answer.citations?.length ?? 0} citations · run_id <code className="font-mono">{answer.run_id ?? "n/a"}</code>
              </p>
            </div>
          )}
        </Step>

        <Step n={4} title="Trace the answer (lineage)" done={false}>
          Every number links back to its PDF page or SQL row.
          <CodeBlock
            label="lineage curl"
            code={`curl -sS "$HOST/v1/lineage/${runId}" \\\n  -H "Authorization: Bearer $DOCBOT_API_KEY"`}
          />
        </Step>
      </ol>
      <ErrorNote error={error} />
    </section>
  )
}
