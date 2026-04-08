"use client"

import React from "react"
import { ShieldCheck } from "lucide-react"
import { Dialog, DialogContent } from "@/components/ui"

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
}

const inputCx =
  "w-full px-3 h-9 rounded-[5px] bg-[var(--color-bg-inset)] border border-[var(--color-border-default)] text-[13px] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-quaternary)] outline-none focus:border-[var(--color-cyan-500)] focus:shadow-[0_0_0_3px_var(--glow-cyan)] transition-[border-color,box-shadow] duration-150"

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
}: AuthModalProps) {
  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-[420px]">
        <div className="space-y-1 mb-4">
          <h2 className="text-[15px] font-semibold text-[var(--color-text-primary)]">Welcome to DocBot</h2>
          <p className="text-[12px] text-[var(--color-text-tertiary)]">Sign in to save your work across sessions</p>
        </div>

        <div className="space-y-2.5">
          {authConfig?.github && (
            <button
              onClick={async () => {
                try {
                  const res = await fetch("/api/auth/github")
                  const { url } = await res.json()
                  window.location.href = url
                } catch { setAuthError("Failed to start GitHub sign-in.") }
              }}
              className="flex items-center justify-center gap-2 w-full h-9 rounded-[5px] bg-[var(--color-bg-inset)] hover:bg-[var(--color-bg-overlay)] border border-[var(--color-border-default)] text-[var(--color-text-primary)] text-[13px] font-medium transition-colors"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              Continue with GitHub
            </button>
          )}

          {authConfig?.google && (
            <button
              onClick={async () => {
                try {
                  const res = await fetch("/api/auth/google")
                  const { url } = await res.json()
                  window.location.href = url
                } catch { setAuthError("Failed to start Google sign-in.") }
              }}
              className="flex items-center justify-center gap-2 w-full h-9 rounded-[5px] bg-[var(--color-bg-inset)] hover:bg-[var(--color-bg-overlay)] border border-[var(--color-border-default)] text-[var(--color-text-primary)] text-[13px] font-medium transition-colors"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
              Continue with Google
            </button>
          )}

          {(authConfig?.github || authConfig?.google) && (
            <div className="flex items-center gap-3 py-1">
              <div className="flex-1 h-px bg-[var(--color-border-subtle)]" />
              <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-quaternary)]">or</span>
              <div className="flex-1 h-px bg-[var(--color-border-subtle)]" />
            </div>
          )}

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
            {authError && (
              <p className="text-[11px] text-[var(--color-danger-500)] bg-[var(--color-danger-500)]/10 border border-[var(--color-danger-500)]/30 rounded-[5px] px-2.5 py-1.5">
                {authError}
              </p>
            )}
            <button
              type="submit"
              disabled={authSubmitting || !authEmail || !authPassword}
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
