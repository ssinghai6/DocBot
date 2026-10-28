"use client"

// DOCBOT-1712: landing page for the emailed link (/verify?token=...).
// POSTs the token once, then sends the user to /onboarding. The legacy
// /chat?verify_token= landing still works for links sent before this page.

import React, { useEffect, useRef, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { CircleCheck, CircleX, LoaderCircle } from "lucide-react"
import LogoMark from "@/components/LogoMark"
import { VerifyRequestSchema, VerifyResponseSchema } from "@/lib/schemas/emailVerify"

type Status = "verifying" | "verified" | "invalid"

const GENERIC_INVALID = "That verification link is invalid or has expired."

export default function VerifyPage() {
  const router = useRouter()
  const [status, setStatus] = useState<Status>("verifying")
  const [email, setEmail] = useState("")
  const posted = useRef(false) // Strict Mode runs effects twice; the token is single-use.

  useEffect(() => {
    if (posted.current) return
    posted.current = true
    const token = new URLSearchParams(window.location.search).get("token") ?? ""
    // Drop the token from the address bar so a refresh can't replay it.
    window.history.replaceState({}, "", window.location.pathname)
    const parsed = VerifyRequestSchema.safeParse({ token })
    if (!parsed.success) {
      setStatus("invalid")
      return
    }
    fetch("/api/auth/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed.data),
    })
      .then(async (res) => {
        const body = VerifyResponseSchema.safeParse(await res.json())
        if (res.ok && body.success) {
          setEmail(body.data.email)
          setStatus("verified")
        } else {
          setStatus("invalid")
        }
      })
      .catch(() => setStatus("invalid"))
  }, [])

  useEffect(() => {
    if (status !== "verified") return
    const t = setTimeout(() => router.push("/onboarding"), 1500)
    return () => clearTimeout(t)
  }, [status, router])

  return (
    <main className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] flex items-center justify-center px-4">
      <div className="w-full max-w-[380px] rounded-[8px] border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)] p-6 space-y-4 text-center">
        <span className="mx-auto w-10 h-10 rounded-lg bg-[#0F1E22] border border-white/10 flex items-center justify-center">
          <LogoMark className="w-5 h-5" />
        </span>
        <div role="status" aria-live="polite" className="space-y-3">
          {status === "verifying" && (
            <>
              <LoaderCircle className="mx-auto w-6 h-6 animate-spin text-[var(--color-cyan-500)]" aria-hidden="true" />
              <h1 className="text-lg font-semibold">Verifying your email…</h1>
            </>
          )}
          {status === "verified" && (
            <>
              <CircleCheck className="mx-auto w-6 h-6 text-[var(--color-success-500)]" aria-hidden="true" />
              <h1 className="text-lg font-semibold">Email verified</h1>
              <p className="text-[13px] text-[var(--color-text-secondary)]">
                {email} is confirmed. Taking you to setup…
              </p>
              <Link href="/onboarding" className="inline-block text-[13px] text-[var(--color-cyan-500)] hover:underline outline-none focus-visible:underline">
                Continue
              </Link>
            </>
          )}
          {status === "invalid" && (
            <>
              <CircleX className="mx-auto w-6 h-6 text-[var(--color-danger-500)]" aria-hidden="true" />
              <h1 className="text-lg font-semibold">Link didn&apos;t work</h1>
              <p className="text-[13px] text-[var(--color-text-secondary)]">
                {GENERIC_INVALID} Sign in and we&apos;ll email you a new one.
              </p>
              <Link href="/chat?login" className="inline-block text-[13px] text-[var(--color-cyan-500)] hover:underline outline-none focus-visible:underline">
                Back to sign in
              </Link>
            </>
          )}
        </div>
      </div>
    </main>
  )
}
