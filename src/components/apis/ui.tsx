"use client"

// DOCBOT-3905: small shared pieces for the /apis portal.

import { useState } from "react"
import type { ParsedApiError } from "@/lib/schemas/developer"

export const btnPrimaryCx =
  "inline-flex items-center justify-center gap-2 h-9 px-4 rounded-md bg-[var(--ap-accent-strong)] text-[var(--ap-on-accent)] text-sm font-semibold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
export const btnSecondaryCx =
  "inline-flex items-center justify-center gap-2 h-9 px-3 rounded-md border border-[var(--ap-border)] text-[var(--ap-text)] text-sm font-medium hover:bg-[var(--ap-inset)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
export const cardCx = "rounded-xl border border-[var(--ap-border)] bg-[var(--ap-surface)] p-5"

export function ErrorNote({ error }: { error: ParsedApiError | null }) {
  if (!error) return null
  return (
    <p role="alert" className="mt-3 text-sm text-[var(--ap-danger)]">
      {error.code && <code className="font-mono mr-2">[{error.code}]</code>}
      {error.message}
      {error.requestId && <span className="block text-xs text-[var(--ap-muted)]">Request ID: {error.requestId}</span>}
    </p>
  )
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      className={btnSecondaryCx}
      aria-label={label}
      onClick={() => {
        navigator.clipboard.writeText(text).then(
          () => {
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
          },
          () => setCopied(false),
        )
      }}
    >
      <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
    </button>
  )
}

export function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="relative mt-3">
      <pre
        aria-label={label}
        tabIndex={0}
        className="overflow-x-auto rounded-lg bg-[var(--ap-inset)] border border-[var(--ap-border)] p-3 pr-20 text-xs leading-relaxed font-mono text-[var(--ap-text)]"
      >
        <code>{code}</code>
      </pre>
      <div className="absolute top-2 right-2">
        <CopyButton text={code} label={`Copy ${label}`} />
      </div>
    </div>
  )
}
