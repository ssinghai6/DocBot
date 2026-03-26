"use client"

import React from "react"
import { X, ShieldCheck } from "lucide-react"

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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative z-10 w-full max-w-sm bg-[#12121a] border border-[#ffffff10] rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#ffffff08]">
          <div>
            <h2 className="text-base font-bold text-white">Welcome to DocBot</h2>
            <p className="text-xs text-gray-500 mt-0.5">Sign in to save your work across sessions</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-[#ffffff10] text-gray-500 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-3">
          {/* OAuth buttons */}
          {authConfig?.github && (
            <button
              onClick={async () => {
                try {
                  const res = await fetch("/api/auth/github");
                  const { url } = await res.json();
                  window.location.href = url;
                } catch { setAuthError("Failed to start GitHub sign-in."); }
              }}
              className="flex items-center justify-center gap-2.5 w-full px-4 py-2.5 rounded-xl bg-[#24292e] hover:bg-[#2f363d] border border-[#ffffff15] text-white text-sm font-medium transition-all"
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
                  const res = await fetch("/api/auth/google");
                  const { url } = await res.json();
                  window.location.href = url;
                } catch { setAuthError("Failed to start Google sign-in."); }
              }}
              className="flex items-center justify-center gap-2.5 w-full px-4 py-2.5 rounded-xl bg-[#1a1a24] hover:bg-[#22222e] border border-[#ffffff10] text-white text-sm font-medium transition-all"
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
            <div className="flex items-center gap-3 my-1">
              <div className="flex-1 h-px bg-[#ffffff10]" />
              <span className="text-[11px] text-gray-600">or</span>
              <div className="flex-1 h-px bg-[#ffffff10]" />
            </div>
          )}

          {/* Email / Password */}
          <div className="flex rounded-xl bg-[#1a1a24] border border-[#ffffff08] overflow-hidden text-xs">
            <button
              onClick={() => { setAuthModalTab("login"); setAuthError(null); }}
              className={`flex-1 py-2 font-medium transition-colors ${authModalTab === "login" ? "bg-[#667eea]/20 text-[#a5b4fc]" : "text-gray-500 hover:text-gray-300"}`}
            >
              Sign in
            </button>
            <button
              onClick={() => { setAuthModalTab("register"); setAuthError(null); }}
              className={`flex-1 py-2 font-medium transition-colors ${authModalTab === "register" ? "bg-[#667eea]/20 text-[#a5b4fc]" : "text-gray-500 hover:text-gray-300"}`}
            >
              Create account
            </button>
          </div>

          <form onSubmit={handleEmailAuth} className="space-y-2.5">
            {authModalTab === "register" && (
              <input
                type="text"
                placeholder="Your name (optional)"
                value={authName}
                onChange={e => setAuthName(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-xl bg-[#1a1a24] border border-[#ffffff10] text-white text-sm placeholder-gray-600 outline-none focus:border-[#667eea]/50 transition-colors"
              />
            )}
            <input
              type="email"
              placeholder="Email address"
              value={authEmail}
              onChange={e => setAuthEmail(e.target.value)}
              required
              className="w-full px-3.5 py-2.5 rounded-xl bg-[#1a1a24] border border-[#ffffff10] text-white text-sm placeholder-gray-600 outline-none focus:border-[#667eea]/50 transition-colors"
            />
            <input
              type="password"
              placeholder="Password"
              value={authPassword}
              onChange={e => setAuthPassword(e.target.value)}
              required
              className="w-full px-3.5 py-2.5 rounded-xl bg-[#1a1a24] border border-[#ffffff10] text-white text-sm placeholder-gray-600 outline-none focus:border-[#667eea]/50 transition-colors"
            />
            {authError && (
              <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{authError}</p>
            )}
            <button
              type="submit"
              disabled={authSubmitting || !authEmail || !authPassword}
              className="w-full py-2.5 rounded-xl bg-[#667eea] hover:bg-[#5a6fd4] disabled:opacity-50 text-white text-sm font-medium transition-colors"
            >
              {authSubmitting ? "Please wait\u2026" : authModalTab === "register" ? "Create account" : "Sign in"}
            </button>
          </form>

          {/* Guest option */}
          <button
            onClick={onClose}
            className="w-full py-2 text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            Continue as guest (no account needed)
          </button>

          {authConfig?.saml && (
            <a
              href="/api/auth/saml/login"
              className="flex items-center justify-center gap-1.5 w-full text-xs text-gray-600 hover:text-[#a5b4fc] transition-colors"
            >
              <ShieldCheck className="w-3 h-3" />
              Sign in with enterprise SSO
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
