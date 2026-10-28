"use client"

// DOCBOT-1805: paywall shown when a chat request is refused with HTTP 402
// (insufficient_credits / account_frozen) or 429 (rate / demo cap).

import React, { useEffect, useState } from "react"
import { Dialog, DialogContent } from "@/components/ui"
import {
  CheckoutSessionRequestSchema,
  CheckoutSessionResponseSchema,
  CreditPackCatalogSchema,
  parseQuotaError,
} from "@/lib/schemas/billing"
import type { CreditPack, QuotaErrorPayload } from "@/lib/schemas/billing"

export type Paywall =
  | { kind: "credits"; payload: QuotaErrorPayload }
  | { kind: "rate"; message: string }

const PAYMENTS_SOON = "Payments open soon — reply to your invite email for more credits."

/** Map a refused chat response to a paywall, or null when it is a normal error. */
export async function readPaywall(response: Response): Promise<Paywall | null> {
  if (response.status !== 402 && response.status !== 429) return null
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    // non-JSON body — fall through
  }
  const payload = parseQuotaError(body)
  if (response.status === 402) return payload ? { kind: "credits", payload } : null
  const detail =
    body !== null && typeof body === "object" && "detail" in body && typeof body.detail === "string"
      ? body.detail
      : null
  return { kind: "rate", message: payload?.message ?? detail ?? "You've reached the free demo limit." }
}

const fmtCredits = (n: number) => n.toLocaleString()
const fmtPrice = (p: CreditPack) =>
  new Intl.NumberFormat(undefined, { style: "currency", currency: p.currency.toUpperCase() }).format(p.price_cents / 100)

const btnPrimary =
  "h-9 px-4 rounded-[5px] bg-[var(--color-cyan-500)] text-white text-[13px] font-medium disabled:opacity-50 transition-opacity"
const btnGhost =
  "h-9 px-4 rounded-[5px] border border-[var(--color-border-default)] text-[13px] text-[var(--color-text-secondary)]"

interface Props {
  paywall: Paywall
  isSignedIn: boolean
  onSignIn: () => void
  onClose: () => void
}

export default function OutOfCreditsModal({ paywall, isSignedIn, onSignIn, onClose }: Props) {
  const [packs, setPacks] = useState<CreditPack[] | null>(null)
  const [buying, setBuying] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const needsPacks = paywall.kind === "credits" && paywall.payload.error === "insufficient_credits"

  useEffect(() => {
    if (!needsPacks) return
    fetch("/api/billing/packs")
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        const parsed = CreditPackCatalogSchema.safeParse(data)
        setPacks(parsed.success ? parsed.data.packs : [])
      })
      .catch(() => setPacks([]))
  }, [needsPacks])

  const buy = async (packId: string) => {
    setBuying(packId)
    setNotice(null)
    try {
      const res = await fetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        // Come back to the chat so the toast + kept question are right there.
        body: JSON.stringify(CheckoutSessionRequestSchema.parse({ pack_id: packId, return_path: "/chat" })),
      })
      const parsed = CheckoutSessionResponseSchema.safeParse(res.ok ? await res.json() : null)
      if (parsed.success) {
        window.location.assign(parsed.data.url)
        return
      }
      setNotice(
        res.status === 503 ? PAYMENTS_SOON
        : res.status === 403 ? "Only a workspace admin can buy credits. Ask your admin to top up."
        : res.status === 401 ? "Sign in to buy credits."
        : "Checkout failed. Please try again.",
      )
    } catch {
      setNotice("Checkout failed. Please try again.")
    }
    setBuying(null)
  }

  let title: string
  let body: React.ReactNode
  if (paywall.kind === "rate") {
    title = isSignedIn ? "Slow down a moment" : "Sign in (free) to keep going"
    body = (
      <>
        <p className="text-[13px] text-[var(--color-text-secondary)]">{paywall.message}</p>
        {!isSignedIn && (
          <p className="text-[12px] text-[var(--color-text-tertiary)]">
            A free account comes with monthly credits — no card required.
          </p>
        )}
        <div className="flex justify-end gap-2 pt-2">
          <button className={btnGhost} onClick={onClose}>Close</button>
          {!isSignedIn && <button className={btnPrimary} onClick={onSignIn}>Sign in</button>}
        </div>
      </>
    )
  } else if (paywall.payload.error === "account_frozen") {
    title = "Account on hold"
    body = (
      <>
        <p className="text-[13px] text-[var(--color-text-secondary)]">{paywall.payload.message}</p>
        <p className="text-[12px] text-[var(--color-text-tertiary)]">
          Reply to your invite email or contact support and we&apos;ll sort it out.
        </p>
        <div className="flex justify-end pt-2">
          <button className={btnGhost} onClick={onClose}>Close</button>
        </div>
      </>
    )
  } else {
    const { required_credits: required, available_credits: available } = paywall.payload
    const buyable = (packs ?? []).some(p => p.purchasable)
    title = "You're out of credits"
    body = (
      <>
        {required !== null && available !== null && (
          <p className="text-[13px] text-[var(--color-text-secondary)]">
            This request needs about <strong>{fmtCredits(required)}</strong> credits; you have{" "}
            <strong>{fmtCredits(available)}</strong>. Your question is still in the box — resend it after topping up.
          </p>
        )}
        {packs === null ? (
          <p className="text-[12px] text-[var(--color-text-tertiary)]">Loading credit packs…</p>
        ) : (
          <ul className="space-y-2">
            {packs.map(p => (
              <li
                key={p.id}
                className={`flex items-center justify-between gap-3 rounded-[6px] border px-3 py-2 ${
                  p.highlight ? "border-[var(--color-cyan-500)]" : "border-[var(--color-border-default)]"
                }`}
              >
                <div>
                  <div className="text-[13px] font-medium text-[var(--color-text-primary)]">{p.name}</div>
                  <div className="text-[12px] text-[var(--color-text-tertiary)]">
                    {fmtCredits(p.credits)} credits · {fmtPrice(p)}
                  </div>
                </div>
                {p.purchasable ? (
                  <button className={btnPrimary} disabled={buying !== null} onClick={() => buy(p.id)}>
                    {buying === p.id ? "Opening…" : "Buy"}
                  </button>
                ) : (
                  <span className="text-[12px] text-[var(--color-text-quaternary)]">Coming soon</span>
                )}
              </li>
            ))}
          </ul>
        )}
        {(notice || (packs !== null && !buyable)) && (
          <p role="status" className="text-[12px] text-[var(--color-text-tertiary)]">{notice ?? PAYMENTS_SOON}</p>
        )}
        <div className="flex justify-end pt-2">
          <button className={btnGhost} onClick={onClose}>Close</button>
        </div>
      </>
    )
  }

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-[440px]">
        <h2 className="text-[15px] font-semibold text-[var(--color-text-primary)] mb-3">{title}</h2>
        <div className="space-y-3">{body}</div>
      </DialogContent>
    </Dialog>
  )
}
