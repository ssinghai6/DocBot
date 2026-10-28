"use client"

import React from "react"
import { ShieldCheck } from "lucide-react"
import { Dialog, DialogContent } from "@/components/ui"
import TurnstileWidget from "@/components/TurnstileWidget"
import OAuthButtons, { authInputCx } from "@/components/OAuthButtons"

interface AuthConfig {
  email: boolean
  github: boolean
  google: boolean
  saml: boolean
}

interface AuthModalProps {
  authConfig: AuthConfig | null
  authModalTab: "login" | "register"
  setAuthModalTab: (tab: "login" | "register") => void
  authEmail: string
  setAuthEmail: (v: string) => void
  authPassword: string
  setAuthPassword: (v: string) => void
  authName: string
  setAuthName: (v: string) => void
  authSubmitting: boolean
  authError: string | null
  setAuthError: (v: string | null) => void
  handleEmailAuth: (e: React.FormEvent) => void
  onClose: () => void
  // DOCBOT-1713: Turnstile on the register tab (rendered only when the
  // NEXT_PUBLIC_TURNSTILE_SITE_KEY is set; token sent with the signup POST).
  turnstileSiteKey: string
  turnstileToken: string
  setTurnstileToken: (v: string) => void
  turnstileResetSignal: number
}

const inputCx = authInputCx

export default function AuthModal({
  authConfig,
  authModalTab,
  setAuthModalTab,
  authEmail,
  setAuthEmail,
  authPassword,
  setAuthPassword,
  authName,
  setAuthName,
  authSubmitting,
  authError,
  setAuthError,
  handleEmailAuth,
  onClose,
  turnstileSiteKey,
  turnstileToken,
  setTurnstileToken,
  turnstileResetSignal,
}: AuthModalProps) {
  const turnstileRequired = turnstileSiteKey.length > 0
  const botCheckMissing = turnstileRequired && authModalTab === "register" && !turnstileToken
  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-[420px]">
        <div className="space-y-1 mb-4">
          <h2 className="text-[15px] font-semibold text-[var(--color-text-primary)]">Welcome to DocBot</h2>
          <p className="text-[12px] text-[var(--color-text-tertiary)]">Sign in to save your work across sessions</p>
        </div>

        <div className="space-y-2.5">
          <OAuthButtons
            github={authConfig?.github ?? false}
            google={authConfig?.google ?? false}
            onError={setAuthError}
          />

          {/* Segmented tab */}
          <div className="flex rounded-[5px] bg-[var(--color-bg-inset)] border border-[var(--color-border-default)] overflow-hidden text-[11px] h-8">
            <button
              type="button"
              onClick={() => { setAuthModalTab("login"); setAuthError(null) }}
              className={`flex-1 font-medium transition-colors ${authModalTab === "login" ? "bg-[var(--color-bg-overlay)] text-[var(--color-text-primary)]" : "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"}`}
            >
              Sign in
            </button>
            <button
              type="button"
              onClick={() => { setAuthModalTab("register"); setAuthError(null) }}
              className={`flex-1 font-medium transition-colors ${authModalTab === "register" ? "bg-[var(--color-bg-overlay)] text-[var(--color-text-primary)]" : "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"}`}
            >
              Create account
            </button>
          </div>

          <form onSubmit={handleEmailAuth} className="space-y-2">
            {authModalTab === "register" && (
              <input
                type="text"
                placeholder="Your name (optional)"
                value={authName}
                onChange={e => setAuthName(e.target.value)}
                className={inputCx}
              />
            )}
            <input
              type="email"
              placeholder="Email address"
              value={authEmail}
              onChange={e => setAuthEmail(e.target.value)}
              required
              className={inputCx}
            />
            <input
              type="password"
              placeholder="Password"
              value={authPassword}
              onChange={e => setAuthPassword(e.target.value)}
              required
              className={inputCx}
            />
            {authModalTab === "register" && (
              <label className="flex items-start gap-2 text-[11px] text-[var(--color-text-tertiary)]">
                <input type="checkbox" required className="mt-0.5 accent-[var(--color-cyan-500)]" />
                <span>
                  I agree to the{" "}
                  <a href="/terms" target="_blank" rel="noopener noreferrer" className="text-[var(--color-cyan-500)] hover:underline">Terms</a>
                  {" "}and{" "}
                  <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-[var(--color-cyan-500)] hover:underline">Privacy Policy</a>
                </span>
              </label>
            )}
            {turnstileRequired && authModalTab === "register" && (
              <div className="flex justify-center py-0.5">
                <TurnstileWidget
                  siteKey={turnstileSiteKey}
                  onToken={setTurnstileToken}
                  resetSignal={turnstileResetSignal}
                />
              </div>
            )}
            {authError && (
              <p role="alert" className="text-[11px] text-[var(--color-danger-500)] bg-[var(--color-danger-500)]/10 border border-[var(--color-danger-500)]/30 rounded-[5px] px-2.5 py-1.5">
                {authError}
              </p>
            )}
            <button
              type="submit"
              disabled={authSubmitting || !authEmail || !authPassword || botCheckMissing}
              className="w-full h-9 rounded-[5px] bg-[var(--color-cyan-500)] hover:bg-[var(--color-cyan-600)] disabled:opacity-40 disabled:cursor-not-allowed text-[var(--color-bg-base)] text-[13px] font-semibold transition-colors"
            >
              {authSubmitting ? "Please wait…" : authModalTab === "register" ? "Create account" : "Sign in"}
            </button>
          </form>

          <button
            onClick={onClose}
            className="w-full py-1.5 text-[11px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors"
          >
            Continue as guest (no account needed)
          </button>

          {authConfig?.saml && (
            <a
              href="/api/auth/saml/login"
              className="flex items-center justify-center gap-1.5 w-full text-[11px] text-[var(--color-text-tertiary)] hover:text-[var(--color-cyan-500)] transition-colors"
            >
              <ShieldCheck className="w-3 h-3" />
              Sign in with enterprise SSO
            </a>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
