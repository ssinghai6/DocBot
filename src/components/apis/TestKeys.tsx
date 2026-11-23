"use client"

// DOCBOT-3905: mint, list and revoke the signed-in user's own test keys.
// The plaintext key lives only in the parent's React state (never storage/URL).

import { useCallback, useEffect, useState } from "react"
import { apiCall } from "@/lib/developerApi"
import {
  TestKeyCreateResponseSchema,
  TestKeyListResponseSchema,
  TestKeyRevokeResponseSchema,
  type ParsedApiError,
  type TestKeyCreateResponse,
  type TestKeySummary,
} from "@/lib/schemas/developer"
import { CopyButton, ErrorNote, btnPrimaryCx, btnSecondaryCx, cardCx } from "./ui"

type Props = {
  activeKey: TestKeyCreateResponse | null
  onKey: (key: TestKeyCreateResponse | null) => void
}

const fmt = (iso: string | null) => (iso ? new Date(iso).toLocaleDateString() : "never")

export default function TestKeys({ activeKey, onKey }: Props) {
  const [keys, setKeys] = useState<TestKeySummary[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ParsedApiError | null>(null)

  const refresh = useCallback(
    () =>
      apiCall("/api/developer/test-keys", { method: "GET" }, TestKeyListResponseSchema).then((res) =>
        res.ok ? setKeys(res.data.keys) : setError(res.error),
      ),
    [],
  )

  useEffect(() => {
    void refresh()
  }, [refresh])

  async function create() {
    setBusy(true)
    setError(null)
    const res = await apiCall("/api/developer/test-key", { method: "POST" }, TestKeyCreateResponseSchema)
    setBusy(false)
    if (!res.ok) return setError(res.error)
    onKey(res.data)
    void refresh()
  }

  async function revoke(k: TestKeySummary) {
    if (!window.confirm(`Revoke test key ${k.prefix}…? Requests using it will fail immediately.`)) return
    setError(null)
    const res = await apiCall(`/api/developer/test-key/${encodeURIComponent(k.id)}`, { method: "DELETE" }, TestKeyRevokeResponseSchema)
    if (!res.ok) return setError(res.error)
    if (activeKey?.id === k.id) onKey(null)
    void refresh()
  }

  return (
    <section aria-labelledby="keys-heading" className={cardCx}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="keys-heading" className="text-lg font-semibold">Test keys</h2>
          <p className="text-sm text-[var(--ap-muted)]">
            Free, sandbox-only keys. Up to 2 active, each valid for 30 days.
          </p>
        </div>
        <button type="button" className={btnPrimaryCx} onClick={create} disabled={busy}>
          {busy ? "Creating…" : "Get test key"}
        </button>
      </div>

      <ErrorNote error={error} />

      {activeKey && (
        <div className="mt-4 rounded-lg border border-[var(--ap-accent)] bg-[var(--ap-inset)] p-4">
          <label htmlFor="active-key" className="block text-sm font-medium">
            Your new test key (shown once)
          </label>
          <div className="mt-2 flex gap-2">
            <input
              id="active-key"
              readOnly
              value={activeKey.key}
              autoComplete="off"
              spellCheck={false}
              className="flex-1 min-w-0 h-9 rounded-md border border-[var(--ap-border)] bg-[var(--ap-bg)] px-3 font-mono text-xs"
              onFocus={(e) => e.currentTarget.select()}
            />
            <CopyButton text={activeKey.key} label="Copy test key" />
          </div>
          <p className="mt-2 text-xs text-[var(--ap-muted)]">
            Kept only in this tab&apos;s memory and used by the quickstart and Try it below. Copy it now: reloading the
            page forgets it and it cannot be shown again.
          </p>
        </div>
      )}

      <div className="mt-4">
        {keys === null ? (
          <p className="text-sm text-[var(--ap-muted)]">Loading keys…</p>
        ) : keys.length === 0 ? (
          <p className="text-sm text-[var(--ap-muted)]">No active test keys yet.</p>
        ) : (
          <ul className="divide-y divide-[var(--ap-border)]" aria-label="Active test keys">
            {keys.map((k) => (
              <li key={k.id} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
                <span>
                  <code className="font-mono">{k.prefix}…</code>
                  {activeKey?.id === k.id && <span className="ml-2 text-xs text-[var(--ap-success)]">in use on this page</span>}
                  <span className="block text-xs text-[var(--ap-muted)]">
                    Created {fmt(k.created_at)} · expires {fmt(k.expires_at)} · last used {fmt(k.last_used_at)}
                  </span>
                </span>
                <button
                  type="button"
                  className={btnSecondaryCx}
                  onClick={() => revoke(k)}
                  aria-label={`Revoke test key ${k.prefix}`}
                >
                  Revoke
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
