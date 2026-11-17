"use client"

// DOCBOT-2581: waitlist form (used by OutOfCreditsModal and the landing page).

import React, { useState } from "react"
import { authInputCx, authPrimaryBtnCx } from "@/components/OAuthButtons"
import { WaitlistRequestSchema, WaitlistResponseSchema } from "@/lib/schemas/waitlist"
import type { WaitlistSource } from "@/lib/schemas/waitlist"

const labelCx = "block text-[12px] font-medium text-[var(--color-text-secondary)] mb-1"

interface Props {
  source: WaitlistSource
  defaultEmail?: string
}

export default function WaitlistForm({ source, defaultEmail = "" }: Props) {
  const [email, setEmail] = useState(defaultEmail)
  const [name, setName] = useState("")
  const [company, setCompany] = useState("")
  const [note, setNote] = useState("")
  const [website, setWebsite] = useState("") // honeypot
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    const parsed = WaitlistRequestSchema.safeParse({
      email, name, company, source,
      ...(note.trim() ? { note } : {}),
      ...(website ? { website } : {}),
    })
    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? "Check the form and try again.")
      return
    }
    setSubmitting(true)
    try {
      const res = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(parsed.data),
      })
      if (res.ok && WaitlistResponseSchema.safeParse(await res.json()).success) {
        setDone(true)
      } else {
        setError(res.status === 429 ? "Too many attempts. Please try again in a minute." : "Something went wrong. Please try again.")
      }
    } catch {
      setError("Something went wrong. Please try again.")
    }
    setSubmitting(false)
  }

  if (done) {
    return (
      <p role="status" className="text-[13px] text-[var(--color-text-primary)]">
        Thanks, you&apos;re on the list. We&apos;ll get back to you within 1-2 business days.
      </p>
    )
  }

  return (
    <form onSubmit={submit} className="space-y-3 text-left" noValidate>
      <div>
        <label htmlFor={`wl-email-${source}`} className={labelCx}>Work email</label>
        <input id={`wl-email-${source}`} type="email" autoComplete="email" required className={authInputCx}
          value={email} onChange={e => setEmail(e.target.value)} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label htmlFor={`wl-name-${source}`} className={labelCx}>Name</label>
          <input id={`wl-name-${source}`} autoComplete="name" required maxLength={120} className={authInputCx}
            value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <label htmlFor={`wl-company-${source}`} className={labelCx}>Company</label>
          <input id={`wl-company-${source}`} autoComplete="organization" required maxLength={120} className={authInputCx}
            value={company} onChange={e => setCompany(e.target.value)} />
        </div>
      </div>
      <div>
        <label htmlFor={`wl-note-${source}`} className={labelCx}>Anything we should know? (optional)</label>
        <textarea id={`wl-note-${source}`} rows={2} maxLength={1000} className={`${authInputCx} h-auto py-2`}
          value={note} onChange={e => setNote(e.target.value)} />
      </div>
      {/* Honeypot: hidden from people and assistive tech; bots fill it. */}
      <div aria-hidden="true" className="absolute -left-[9999px] h-0 w-0 overflow-hidden">
        <label>Website<input tabIndex={-1} autoComplete="off" value={website} onChange={e => setWebsite(e.target.value)} /></label>
      </div>
      {error && <p role="alert" className="text-[12px] text-[var(--color-error-500,#ef4444)]">{error}</p>}
      <button type="submit" disabled={submitting} className={authPrimaryBtnCx}>
        {submitting ? "Sending…" : "Join the waitlist"}
      </button>
    </form>
  )
}
