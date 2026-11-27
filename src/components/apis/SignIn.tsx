"use client"

// DOCBOT-3905: sign in without leaving /apis (same POST /api/auth/login as
// /onboarding), so there is no redirect param and no open-redirect surface.
// GitHub/Google/SSO return to /chat (backend-owned redirect), hence the link.

import { useState } from "react"
import Link from "next/link"
import { parseApiError, type ParsedApiError } from "@/lib/schemas/developer"
import { ErrorNote, btnPrimaryCx, btnSecondaryCx, cardCx } from "./ui"

const inputCx =
  "w-full h-9 rounded-md border border-[var(--ap-border)] bg-[var(--ap-bg)] px-3 text-sm text-[var(--ap-text)]"

export default function SignIn({ onSignedIn }: { onSignedIn: () => void }) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ParsedApiError | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      })
      if (res.ok) {
        setPassword("")
        onSignedIn()
      } else {
        setError(parseApiError(await res.json().catch(() => null), res.status))
      }
    } catch {
      setError({ code: null, message: "Network error. Please try again.", requestId: null })
    } finally {
      setBusy(false)
    }
  }

  return (
    <section aria-labelledby="cta-heading" className={cardCx}>
      <h2 id="cta-heading" className="text-lg font-semibold">Try the API with a free test key</h2>
      <p className="mt-1 text-sm text-[var(--ap-muted)]">
        The reference below is read-only. Sign in to get a sandbox test key, run the quickstart and use Try it.
      </p>
      <form onSubmit={submit} className="mt-4 grid max-w-md gap-3">
        <div>
          <label htmlFor="apis-email" className="block text-sm font-medium mb-1">Email</label>
          <input id="apis-email" type="email" autoComplete="email" required value={email}
            onChange={(e) => setEmail(e.target.value)} className={inputCx} />
        </div>
        <div>
          <label htmlFor="apis-password" className="block text-sm font-medium mb-1">Password</label>
          <input id="apis-password" type="password" autoComplete="current-password" required value={password}
            onChange={(e) => setPassword(e.target.value)} className={inputCx} />
        </div>
        <div className="flex flex-wrap gap-3">
          <button type="submit" className={btnPrimaryCx} disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <Link href="/signup" className={btnSecondaryCx}>Create a free account</Link>
          <Link href="/chat?login" className={btnSecondaryCx}>GitHub, Google or SSO</Link>
        </div>
      </form>
      <ErrorNote error={error} />
    </section>
  )
}
