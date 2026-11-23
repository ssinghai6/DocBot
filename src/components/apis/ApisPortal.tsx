"use client"

// DOCBOT-3905: /apis developer portal shell. Signed out: read-only reference +
// sign-in CTA. Signed in: test keys, quickstart and Scalar Try it.
// The plaintext test key is React state only: never localStorage,
// sessionStorage, cookies or the URL. A reload forgets it by design.

import { useEffect, useState, useSyncExternalStore } from "react"
import dynamic from "next/dynamic"
import Link from "next/link"
import LogoMark from "@/components/LogoMark"
import { AuthMeSchema } from "@/components/types"
import type { TestKeyCreateResponse } from "@/lib/schemas/developer"
import Quickstart from "./Quickstart"
import TestKeys from "./TestKeys"
import SignIn from "./SignIn"
import { btnSecondaryCx } from "./ui"
// Static import: CSS imported only inside the ssr:false chunk is not loaded.
import "@scalar/api-reference-react/style.css"
import "./apis.css"

// Scalar touches window/document on import; render it client-side only.
const ScalarReference = dynamic(() => import("./ScalarReference"), {
  ssr: false,
  loading: () => <p className="p-6 text-sm text-[var(--ap-muted)]">Loading API reference…</p>,
})

type Mode = "light" | "dark"

const LIGHT_QUERY = "(prefers-color-scheme: light)"
function subscribeScheme(cb: () => void) {
  const mq = window.matchMedia(LIGHT_QUERY)
  mq.addEventListener("change", cb)
  return () => mq.removeEventListener("change", cb)
}
const systemMode = (): Mode => (window.matchMedia(LIGHT_QUERY).matches ? "light" : "dark")

export default function ApisPortal() {
  const [signedIn, setSignedIn] = useState<boolean | null>(null)
  // Follows the OS theme until the user picks one with the toggle.
  const system = useSyncExternalStore(subscribeScheme, systemMode, (): Mode => "dark")
  const [override, setOverride] = useState<Mode | null>(null)
  const mode = override ?? system
  const [activeKey, setActiveKey] = useState<TestKeyCreateResponse | null>(null)

  useEffect(() => {
    fetch("/api/auth/me", { credentials: "include" })
      .then(async (res) => setSignedIn(res.ok && AuthMeSchema.safeParse(await res.json()).success))
      .catch(() => setSignedIn(false))
  }, [])

  return (
    <div className="apis-portal" data-mode={mode}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-[var(--ap-surface)] focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="border-b border-[var(--ap-border)]">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link href="/" className="flex items-center gap-2 rounded-md">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-[#0F1E22]">
              <LogoMark className="h-3.5 w-3.5" />
            </span>
            <span className="font-semibold">DocBot</span>
            <span className="text-[var(--ap-muted)]">API</span>
          </Link>
          <nav aria-label="Developer portal" className="flex items-center gap-2">
            <button
              type="button"
              className={btnSecondaryCx}
              onClick={() => setOverride(mode === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${mode === "dark" ? "light" : "dark"} mode`}
            >
              {mode === "dark" ? "Light" : "Dark"} mode
            </button>
            <Link href="/chat" className={btnSecondaryCx}>
              Open app
            </Link>
          </nav>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">DocBot API</h1>
        <p className="mt-3 max-w-2xl text-[var(--ap-muted)]">
          Ask questions across PDFs and live databases from your own code. Every answer comes with citations to the
          page and the SQL row, plus any discrepancies between them.
        </p>

        <div className="mt-8 space-y-6">
          {signedIn === null && <p className="text-sm text-[var(--ap-muted)]">Checking your session…</p>}

          {signedIn === false && <SignIn onSignedIn={() => setSignedIn(true)} />}

          {signedIn && (
            <>
              <TestKeys activeKey={activeKey} onKey={setActiveKey} />
              <Quickstart testKey={activeKey?.key ?? null} />
            </>
          )}
        </div>

        <section aria-labelledby="ref-heading" className="mt-12">
          <h2 id="ref-heading" className="text-2xl font-semibold">API reference</h2>
          <p className="mt-1 text-sm text-[var(--ap-muted)]">
            {activeKey
              ? "Try it is on and sends requests to this site's /v1 with your test key as the Bearer token."
              : "Read-only. Get a test key to send requests with Try it."}
          </p>
          <div className="mt-4 overflow-hidden rounded-xl border border-[var(--ap-border)]">
            <ScalarReference testKey={activeKey?.key ?? null} mode={mode} />
          </div>
        </section>
      </main>
    </div>
  )
}
