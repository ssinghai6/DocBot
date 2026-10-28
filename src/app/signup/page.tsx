"use client"

// DOCBOT-1712: standalone signup — form on the left, a sample tie-out on the right.

import React, { useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { MailCheck, TriangleAlert } from "lucide-react"
import LogoMark from "@/components/LogoMark"
import TurnstileWidget from "@/components/TurnstileWidget"
import OAuthButtons, { authInputCx, authPrimaryBtnCx } from "@/components/OAuthButtons"
import { LegalLinks } from "@/components/LegalPage"
import { RegisterRequestSchema, RegisterResponseSchema } from "@/lib/schemas/emailVerify"
import { AuthConfigSchema, apiErrorDetail, type AuthConfig } from "@/lib/schemas/signup"

// Mirrors api/email_verification_service.RESEND_COOLDOWN_SECONDS.
const RESEND_COOLDOWN_SECONDS = 60

const labelCx = "block text-[12px] font-medium text-[var(--color-text-secondary)] mb-1"
const linkCx = "text-[var(--color-cyan-500)] hover:text-[var(--color-cyan-200)] underline-offset-2 hover:underline outline-none focus-visible:underline"

export default function SignupPage() {
  const router = useRouter()
  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY ?? ""

  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [acceptedTerms, setAcceptedTerms] = useState(false)
  const [turnstileToken, setTurnstileToken] = useState("")
  const [turnstileReset, setTurnstileReset] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [inboxEmail, setInboxEmail] = useState<string | null>(null)
  const [cooldown, setCooldown] = useState(0)
  const [resendStatus, setResendStatus] = useState<string | null>(null)

  useEffect(() => {
    fetch("/api/auth/config")
      .then((r) => r.json())
      .then((d: unknown) => {
        const parsed = AuthConfigSchema.safeParse(d)
        if (parsed.success) setAuthConfig(parsed.data)
      })
      .catch(() => setAuthConfig(null))
  }, [])

  useEffect(() => {
    if (cooldown <= 0) return
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000)
    return () => clearTimeout(t)
  }, [cooldown])

  const botCheckMissing = turnstileSiteKey.length > 0 && !turnstileToken

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    const parsed = RegisterRequestSchema.safeParse({
      email,
      password,
      ...(name.trim() ? { name: name.trim() } : {}),
      ...(turnstileToken ? { turnstile_token: turnstileToken } : {}),
      accepted_terms: acceptedTerms,
    })
    if (!parsed.success) {
      setError("Please enter a valid email and password.")
      return
    }
    if (!acceptedTerms) {
      setError("Please accept the Terms of Service and Privacy Policy.")
      return
    }
    setSubmitting(true)
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(parsed.data),
      })
      const data: unknown = await res.json()
      if (!res.ok) {
        setError(apiErrorDetail(data, "Something went wrong. Please try again."))
        setTurnstileReset((n) => n + 1)
        return
      }
      const result = RegisterResponseSchema.safeParse(data)
      if (result.success && result.data.status === "verification_required") {
        setInboxEmail(result.data.email)
        setCooldown(RESEND_COOLDOWN_SECONDS)
        return
      }
      router.push("/onboarding")
    } catch {
      setError("Network error. Please try again.")
      setTurnstileReset((n) => n + 1)
    } finally {
      setSubmitting(false)
    }
  }

  // No dedicated resend endpoint: signing in with an unverified address
  // re-sends the link (server-throttled to once per cooldown) and returns 403.
  async function handleResend() {
    if (!inboxEmail || cooldown > 0) return
    setResendStatus(null)
    setCooldown(RESEND_COOLDOWN_SECONDS)
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: inboxEmail, password }),
      })
      if (res.ok) {
        router.push("/onboarding")
        return
      }
      if (res.status === 403) {
        setResendStatus("If your last link is more than a minute old, a fresh one is on its way.")
        return
      }
      setResendStatus(apiErrorDetail(await res.json(), "Couldn't resend the link. Try signing in instead."))
    } catch {
      setResendStatus("Network error. Please try again.")
    }
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] flex flex-col">
      <header className="h-16 px-4 sm:px-6 lg:px-8 flex items-center border-b border-[var(--color-border-subtle)]">
        <Link href="/" className="flex items-center gap-2 rounded-md outline-none focus-visible:shadow-[var(--glow-cyan)]">
          <span className="w-8 h-8 rounded-lg bg-[#0F1E22] border border-white/10 flex items-center justify-center">
            <LogoMark className="w-4 h-4" />
          </span>
          <span className="font-semibold text-lg">DocBot</span>
        </Link>
      </header>

      <main className="flex-1 grid lg:grid-cols-2">
        <section className="flex items-center justify-center px-4 sm:px-6 py-12">
          <div className="w-full max-w-[380px]">
            {inboxEmail ? (
              <div className="space-y-4">
                <MailCheck className="w-8 h-8 text-[var(--color-cyan-500)]" aria-hidden="true" />
                <h1 className="text-xl font-semibold">Check your inbox</h1>
                <p className="text-[13px] text-[var(--color-text-secondary)]">
                  We sent a verification link to{" "}
                  <span className="font-medium text-[var(--color-text-primary)]">{inboxEmail}</span>.
                  Open it to activate your account. The link expires in 24 hours.
                </p>
                <button
                  type="button"
                  onClick={handleResend}
                  disabled={cooldown > 0}
                  className={authPrimaryBtnCx}
                >
                  {cooldown > 0 ? `Resend link in ${cooldown}s` : "Resend link"}
                </button>
                <p aria-live="polite" className="text-[12px] text-[var(--color-text-tertiary)] min-h-[1em]">
                  {resendStatus}
                </p>
                <p className="text-[12px] text-[var(--color-text-tertiary)]">
                  Wrong address?{" "}
                  <button type="button" className={linkCx} onClick={() => { setInboxEmail(null); setResendStatus(null) }}>
                    Start over
                  </button>
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-1">
                  <h1 className="text-xl font-semibold">Create your DocBot account</h1>
                  <p className="text-[13px] text-[var(--color-text-tertiary)]">
                    Tie out every number in your draft against the books.
                  </p>
                </div>

                <div className="space-y-2.5">
                  <OAuthButtons
                    github={authConfig?.github ?? false}
                    google={authConfig?.google ?? false}
                    onError={setError}
                  />
                </div>

                <form onSubmit={handleSubmit} className="space-y-3">
                  <div>
                    <label htmlFor="signup-name" className={labelCx}>Name <span className="text-[var(--color-text-quaternary)]">(optional)</span></label>
                    <input id="signup-name" type="text" autoComplete="name" value={name}
                      onChange={(e) => setName(e.target.value)} className={authInputCx} />
                  </div>
                  <div>
                    <label htmlFor="signup-email" className={labelCx}>Work email</label>
                    <input id="signup-email" type="email" autoComplete="email" required value={email}
                      onChange={(e) => setEmail(e.target.value)} className={authInputCx} />
                  </div>
                  <div>
                    <label htmlFor="signup-password" className={labelCx}>Password</label>
                    <input id="signup-password" type="password" autoComplete="new-password" required minLength={8}
                      aria-describedby="signup-password-hint" value={password}
                      onChange={(e) => setPassword(e.target.value)} className={authInputCx} />
                    <p id="signup-password-hint" className="mt-1 text-[11px] text-[var(--color-text-quaternary)]">
                      At least 8 characters. Breached passwords are rejected.
                    </p>
                  </div>
                  <div className="flex items-start gap-2">
                    <input id="signup-terms" type="checkbox" required checked={acceptedTerms}
                      onChange={(e) => setAcceptedTerms(e.target.checked)}
                      className="mt-0.5 w-4 h-4 accent-[var(--color-cyan-500)] outline-none focus-visible:shadow-[var(--glow-cyan)]" />
                    <label htmlFor="signup-terms" className="text-[12px] text-[var(--color-text-secondary)]">
                      I agree to the <Link href="/terms" target="_blank" className={linkCx}>Terms of Service</Link> and{" "}
                      <Link href="/privacy" target="_blank" className={linkCx}>Privacy Policy</Link>.
                    </label>
                  </div>
                  {turnstileSiteKey && (
                    <div className="flex justify-center">
                      <TurnstileWidget siteKey={turnstileSiteKey} onToken={setTurnstileToken} resetSignal={turnstileReset} />
                    </div>
                  )}
                  <div aria-live="assertive">
                    {error && (
                      <p role="alert" className="text-[12px] text-[var(--color-danger-500)] bg-[var(--color-danger-500)]/10 border border-[var(--color-danger-500)]/30 rounded-[5px] px-2.5 py-1.5">
                        {error}
                      </p>
                    )}
                  </div>
                  <button type="submit" disabled={submitting || !email || !password || !acceptedTerms || botCheckMissing} className={authPrimaryBtnCx}>
                    {submitting ? "Creating account…" : "Create account"}
                  </button>
                </form>

                <p className="text-[12px] text-[var(--color-text-tertiary)]">
                  Already have an account? <Link href="/chat?login" className={linkCx}>Sign in</Link>
                </p>
              </div>
            )}
          </div>
        </section>

        <aside
          aria-label="Example tie-out"
          className="hidden lg:flex items-center justify-center px-8 py-12 bg-[var(--color-bg-surface)] border-l border-[var(--color-border-subtle)]"
        >
          <SampleTieOutCard />
        </aside>
      </main>

      <footer className="border-t border-[var(--color-border-subtle)] py-4 px-4 sm:px-6 lg:px-8 text-[12px] text-[var(--color-text-tertiary)] flex items-center justify-between">
        <span>&copy; 2026 DocBot</span>
        <LegalLinks />
      </footer>
    </div>
  )
}

function SampleTieOutCard() {
  const rows: { label: string; source: string; value: string }[] = [
    { label: "Draft report", source: "Q3 board deck, p. 4", value: "$418.2M" },
    { label: "Books", source: "GL revenue accounts, Q3", value: "$412.6M" },
  ]
  return (
    <div className="w-full max-w-[420px] space-y-4">
      <p className="text-[11px] uppercase tracking-wider text-[var(--color-text-quaternary)]">What DocBot catches</p>
      <div className="rounded-[8px] border border-[var(--color-amber-500)]/40 bg-[var(--color-bg-elevated)] shadow-[0_0_24px_var(--amber-glow)] p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-[14px] font-semibold">Total revenue</h2>
          <span className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full bg-[var(--color-amber-900)] text-[var(--color-amber-500)]">
            <TriangleAlert className="w-3 h-3" aria-hidden="true" />
            Doesn&apos;t tie
          </span>
        </div>
        <dl className="space-y-2.5">
          {rows.map((r) => (
            <div key={r.label} className="flex items-baseline justify-between gap-4">
              <dt>
                <span className="block text-[13px] text-[var(--color-text-secondary)]">{r.label}</span>
                <span className="block text-[11px] text-[var(--color-text-quaternary)]">{r.source}</span>
              </dt>
              <dd className="font-mono text-[15px] tabular-nums">{r.value}</dd>
            </div>
          ))}
          <div className="flex items-baseline justify-between gap-4 pt-2.5 border-t border-[var(--color-border-subtle)]">
            <dt className="text-[13px] text-[var(--color-text-secondary)]">Variance</dt>
            <dd className="font-mono text-[15px] tabular-nums text-[var(--color-amber-500)]">+$5.6M (1.4%)</dd>
          </div>
        </dl>
        <p className="text-[12px] text-[var(--color-text-tertiary)]">
          The deck includes a $5.6M deal that was booked in October. Every figure links back to its page and query.
        </p>
      </div>
      <p className="text-[11px] text-[var(--color-text-quaternary)]">Sample data for illustration.</p>
    </div>
  )
}
