"use client"

// Cloudflare Turnstile widget — DOCBOT-1713.
//
// Rendered only when NEXT_PUBLIC_TURNSTILE_SITE_KEY is set; the widget mints
// a token the signup POST carries as `turnstile_token`, verified server-side
// against TURNSTILE_SECRET_KEY. Render nothing when the key is absent.
//
// The script is loaded once and the widget is rendered explicitly (not the
// auto-scan `.cf-turnstile` class) so React state observes the callback. The
// token reset uses a bump-key (resetSignal) rather than internal effects, so
// a failed submit can clear the single-use token for the next attempt.

import { useEffect, useRef } from "react"

const SCRIPT_ID = "cf-turnstile-api-script"

type TurnstileRenderOptions = {
  sitekey: string
  callback: (token: string) => void
  "expired-callback"?: () => void
  "error-callback"?: (code: string) => string
  theme?: "auto"
}

type TurnstileApi = {
  render: (element: HTMLElement, options: TurnstileRenderOptions) => string
  reset: (widgetId?: string) => void
  remove: (widgetId: string) => void
}

function turnstileGlobal(): TurnstileApi | null {
  if (typeof window === "undefined") return null
  return (window as Window & { turnstile?: TurnstileApi }).turnstile ?? null
}

function loadTurnstileApi(): Promise<TurnstileApi | null> {
  const existing = turnstileGlobal()
  if (existing) return Promise.resolve(existing)
  if (typeof document === "undefined") return Promise.resolve(null)

  return new Promise((resolve) => {
    let script = document.getElementById(SCRIPT_ID) as HTMLScriptElement | null
    if (!script) {
      script = document.createElement("script")
      script.id = SCRIPT_ID
      script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js"
      script.async = true
      document.head.appendChild(script)
    }
    const settle = () => resolve(turnstileGlobal())
    script.addEventListener("load", settle, { once: true })
    script.addEventListener("error", () => resolve(null), { once: true })
  })
}

interface TurnstileWidgetProps {
  siteKey: string
  onToken: (token: string) => void
  /** Bump this counter to clear a consumed/expired token (widget reset). */
  resetSignal: number
}

export default function TurnstileWidget({
  siteKey,
  onToken,
  resetSignal,
}: TurnstileWidgetProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const widgetIdRef = useRef<string | null>(null)
  const onTokenRef = useRef(onToken)
  useEffect(() => {
    onTokenRef.current = onToken
  }, [onToken])

  useEffect(() => {
    let cancelled = false
    loadTurnstileApi().then((api) => {
      if (!api || cancelled || !containerRef.current) return
      if (widgetIdRef.current !== null) return // React 18 Strict Mode double-run guard
      widgetIdRef.current = api.render(containerRef.current, {
        sitekey: siteKey,
        callback: (token: string) => onTokenRef.current(token),
        "expired-callback": () => onTokenRef.current(""),
        "error-callback": () => {
          onTokenRef.current("")
          return "bot-check-failed"
        },
      })
    })
    return () => {
      cancelled = true
    }
  }, [siteKey])

  useEffect(() => {
    if (resetSignal <= 0 || widgetIdRef.current === null) return
    try {
      turnstileGlobal()?.reset(widgetIdRef.current)
    } catch {
      // Widget is mid-teardown — a stale reset is harmless.
    }
  }, [resetSignal])

  return <div ref={containerRef} data-testid="turnstile-widget" />
}
