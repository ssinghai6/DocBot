"use client"

// DOCBOT-1712: two-question setup after signup. Skippable. Verifying an email
// does not open a session, so a signed-out visitor signs in here first.

import React, { useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import LogoMark from "@/components/LogoMark"
import { authInputCx, authPrimaryBtnCx } from "@/components/OAuthButtons"
import { AuthMeSchema } from "@/components/types"
import {
  OnboardingRequestSchema,
  OnboardingResponseSchema,
  USE_CASE_OPTIONS,
  apiErrorDetail,
} from "@/lib/schemas/signup"

type UseCase = (typeof USE_CASE_OPTIONS)[number]["value"]

const labelCx = "block text-[12px] font-medium text-[var(--color-text-secondary)] mb-1"

export default function OnboardingPage() {
  const router = useRouter()
  const [signedIn, setSignedIn] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [workspace, setWorkspace] = useState("")
  const [useCase, setUseCase] = useState<UseCase | "">("")

  useEffect(() => {
    fetch("/api/auth/me", { credentials: "include" })
      .then(async (res) => setSignedIn(res.ok && AuthMeSchema.safeParse(await res.json()).success))
      .catch(() => setSignedIn(false))
  }, [])

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      })
      if (res.ok) {
        setSignedIn(true)
        setPassword("")
      } else {
        setError(apiErrorDetail(await res.json(), "Sign-in failed. Please try again."))
      }
    } catch {
      setError("Network error. Please try again.")
    } finally {
      setSubmitting(false)
    }
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    const parsed = OnboardingRequestSchema.safeParse({
      ...(workspace.trim() ? { workspace_name: workspace.trim() } : {}),
      ...(useCase ? { use_case: useCase } : {}),
    })
    if (!parsed.success) {
      setError("Workspace name must be 80 characters or fewer.")
      return
    }
    setSubmitting(true)
    try {
      const res = await fetch("/api/auth/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(parsed.data),
      })
      const data: unknown = await res.json()
      if (res.ok && OnboardingResponseSchema.safeParse(data).success) {
        router.push("/chat")
        return
      }
      if (res.status === 401) setSignedIn(false)
      setError(apiErrorDetail(data, "Couldn't save. You can skip this for now."))
    } catch {
      setError("Network error. Please try again.")
    } finally {
      setSubmitting(false)
    }
  }

  const errorBox = (
    <div aria-live="assertive">
      {error && (
        <p role="alert" className="text-[12px] text-[var(--color-danger-500)] bg-[var(--color-danger-500)]/10 border border-[var(--color-danger-500)]/30 rounded-[5px] px-2.5 py-1.5">
          {error}
        </p>
      )}
    </div>
  )

  return (
    <main className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-[400px] rounded-[8px] border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)] p-6 space-y-5">
        <div className="flex items-center gap-2">
          <span className="w-8 h-8 rounded-lg bg-[#0F1E22] border border-white/10 flex items-center justify-center">
            <LogoMark className="w-4 h-4" />
          </span>
          <span className="font-semibold">DocBot</span>
        </div>

        {signedIn === null && (
          <p role="status" className="text-[13px] text-[var(--color-text-tertiary)]">Loading…</p>
        )}

        {signedIn === false && (
          <form onSubmit={handleSignIn} className="space-y-3">
            <div className="space-y-1">
              <h1 className="text-lg font-semibold">Sign in to finish setup</h1>
              <p className="text-[13px] text-[var(--color-text-tertiary)]">Your email is verified. Sign in once to continue.</p>
            </div>
            <div>
              <label htmlFor="onb-email" className={labelCx}>Email</label>
              <input id="onb-email" type="email" autoComplete="email" required value={email}
                onChange={(e) => setEmail(e.target.value)} className={authInputCx} />
            </div>
            <div>
              <label htmlFor="onb-password" className={labelCx}>Password</label>
              <input id="onb-password" type="password" autoComplete="current-password" required value={password}
                onChange={(e) => setPassword(e.target.value)} className={authInputCx} />
            </div>
            {errorBox}
            <button type="submit" disabled={submitting || !email || !password} className={authPrimaryBtnCx}>
              {submitting ? "Signing in…" : "Sign in"}
            </button>
          </form>
        )}

        {signedIn === true && (
          <form onSubmit={handleSave} className="space-y-3">
            <div className="space-y-1">
              <h1 className="text-lg font-semibold">Set up your workspace</h1>
              <p className="text-[13px] text-[var(--color-text-tertiary)]">Two quick questions. You can skip them.</p>
            </div>
            <div>
              <label htmlFor="onb-workspace" className={labelCx}>Workspace name</label>
              <input id="onb-workspace" type="text" maxLength={80} placeholder="e.g. Acme Finance" value={workspace}
                onChange={(e) => setWorkspace(e.target.value)} className={authInputCx} />
            </div>
            <div>
              <label htmlFor="onb-use-case" className={labelCx}>Primary use case</label>
              <select id="onb-use-case" value={useCase}
                onChange={(e) => setUseCase(e.target.value as UseCase | "")}
                className={authInputCx}>
                <option value="">Choose one…</option>
                {USE_CASE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
            {errorBox}
            <button type="submit" disabled={submitting || (!workspace.trim() && !useCase)} className={authPrimaryBtnCx}>
              {submitting ? "Saving…" : "Continue"}
            </button>
          </form>
        )}

        <Link href="/chat" className="block text-center text-[12px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] outline-none focus-visible:underline">
          Skip for now
        </Link>
      </div>
    </main>
  )
}
