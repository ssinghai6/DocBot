"use client"

// DOCBOT-1528: "Data & Model Handling" disclosure panel.
//
// Static content panel — every claim below is sourced from shipped code /
// CLAUDE.md, not marketing copy. See the inline citation comment on each
// section for what backs it, so a future edit can re-verify instead of
// drifting into an unverifiable claim:
//   - Document storage: api/utils/vector_store.py (Chroma persistent store,
//     DOCBOT-1001) + local ONNX embeddings (DOCBOT-1511, CLAUDE.md Tech Stack).
//     Retention window is deliberately unspecified — DOCBOT-1525 (vector
//     store retention lifecycle) is in flight on a separate branch and not
//     yet merged to main as of this panel's writing.
//   - LLM providers: api/utils/llm_provider.py (GROQ_MODEL primary,
//     GEMINI_MODEL fallback; _is_retriable_error() lists the fallback
//     trigger conditions — rate limits, 5xx, timeouts, connection errors).
//   - PII masking: CLAUDE.md "Investor Readiness Sprint" entry — "PII
//     masking applied to all SSE, sandbox, audit response paths."
//   - Credential handling: CLAUDE.md "Critical Architecture Rules" —
//     Fernet-encrypted, never logged, never passed to LLM context.

import React from "react"
import { X, Database, Cpu, EyeOff, KeyRound, ShieldCheck } from "lucide-react"

interface DataHandlingPanelProps {
  onClose: () => void
}

export default function DataHandlingPanel({ onClose }: DataHandlingPanelProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      <div className="relative z-10 w-full max-w-lg max-h-[85vh] flex flex-col bg-[var(--color-bg-surface)] border border-[var(--color-border-default)] rounded-2xl shadow-2xl overflow-hidden">
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[var(--color-border-subtle)] shrink-0">
          <div className="w-8 h-8 rounded-lg bg-[var(--color-cyan-500)]/20 flex items-center justify-center">
            <ShieldCheck className="w-4 h-4 text-[var(--color-cyan-500)]" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Data &amp; Model Handling</h2>
            <p className="text-xs text-[var(--color-text-tertiary)]">What happens to your data, in plain language</p>
          </div>
          <button
            onClick={onClose}
            className="ml-auto p-1.5 rounded-lg hover:bg-[var(--color-bg-overlay)] text-[var(--color-text-tertiary)] hover:text-gray-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5 text-sm">
          <section className="flex gap-3">
            <Database className="w-4 h-4 text-[var(--color-amber-500)] shrink-0 mt-0.5" />
            <div>
              <h3 className="text-xs font-semibold text-gray-200 mb-1">Document storage</h3>
              <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
                Uploaded documents are chunked and embedded into a vector store (Chroma,
                persisted on the server) using a local embedding model that runs on our
                infrastructure — your document text is not sent to a third-party embeddings
                API. Documents are retained for a limited period and then deleted; they are
                not used to train any model.
              </p>
            </div>
          </section>

          <section className="flex gap-3">
            <Cpu className="w-4 h-4 text-[var(--color-cyan-500)] shrink-0 mt-0.5" />
            <div>
              <h3 className="text-xs font-semibold text-gray-200 mb-1">Which models process your requests</h3>
              <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
                Requests are processed by Groq (Llama / GPT-OSS models) as the primary
                provider. If a Groq call fails with a retriable error — a rate limit, a
                5xx server error, a timeout, or a connection error — the request
                automatically falls back to Google Gemini 2.5 Flash so your query still
                completes. Both providers receive only the text needed to answer your
                question (and retrieved document/data excerpts); database connection
                strings and credentials are never included.
              </p>
            </div>
          </section>

          <section className="flex gap-3">
            <EyeOff className="w-4 h-4 text-[var(--color-success-500)] shrink-0 mt-0.5" />
            <div>
              <h3 className="text-xs font-semibold text-gray-200 mb-1">PII masking</h3>
              <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
                Personally identifiable information is automatically detected and masked
                in streamed chat responses, sandbox (Python/pandas) output, and audit log
                entries before they reach your screen or storage.
              </p>
            </div>
          </section>

          <section className="flex gap-3">
            <KeyRound className="w-4 h-4 text-[var(--color-danger-500)] shrink-0 mt-0.5" />
            <div>
              <h3 className="text-xs font-semibold text-gray-200 mb-1">Credential handling</h3>
              <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
                Database and connector credentials are encrypted at rest (Fernet
                symmetric encryption) before being stored. They are never written to
                application logs and never included in any prompt sent to an LLM
                provider.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
