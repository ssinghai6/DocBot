// DOCBOT-1707: shared shell + constants for /terms and /privacy.
//
// LEGAL REVIEW REQUIRED: the Terms and Privacy text are founder-written
// drafts. Counsel must review them before public launch.

import React from "react"
import Link from "next/link"
import LogoMark from "@/components/LogoMark"

export const COMPANY_NAME = process.env.NEXT_PUBLIC_COMPANY_NAME || "DocBot"
export const CONTACT_EMAIL = process.env.NEXT_PUBLIC_CONTACT_EMAIL || "sanshrit.singhai@docbot.space"
export const LEGAL_LAST_UPDATED = "October 28, 2026"

const linkCx =
  "hover:text-[var(--color-text-primary)] transition-colors duration-150 outline-none focus-visible:underline"

/** Footer links to the legal pages (landing, signup, legal pages). */
export function LegalLinks({ className = "" }: { className?: string }) {
  return (
    <nav aria-label="Legal" className={`flex items-center gap-4 ${className}`}>
      <Link href="/terms" className={linkCx}>Terms</Link>
      <Link href="/privacy" className={linkCx}>Privacy</Link>
      <a href={`mailto:${CONTACT_EMAIL}`} className={linkCx}>Contact us</a>
    </nav>
  )
}

export function ContactLink() {
  return (
    <a href={`mailto:${CONTACT_EMAIL}`} className="text-[var(--color-cyan-500)] hover:underline">
      {CONTACT_EMAIL}
    </a>
  )
}

export function LegalSection({ id, title, children }: { id?: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="space-y-3 scroll-mt-20">
      <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">{title}</h2>
      <div className="space-y-3 text-[14px] leading-relaxed text-[var(--color-text-secondary)] [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1.5 [&_strong]:text-[var(--color-text-primary)] [&_strong]:font-medium">
        {children}
      </div>
    </section>
  )
}

export default function LegalPage({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] flex flex-col">
      <header className="h-16 px-4 sm:px-6 lg:px-8 flex items-center border-b border-[var(--color-border-subtle)]">
        <Link href="/" className="flex items-center gap-2 rounded-md outline-none focus-visible:shadow-[var(--glow-cyan)]">
          <span className="w-8 h-8 rounded-lg bg-[#0F1E22] border border-white/10 flex items-center justify-center">
            <LogoMark className="w-4 h-4" />
          </span>
          <span className="font-semibold text-lg">{COMPANY_NAME}</span>
        </Link>
      </header>
      <main className="flex-1 w-full max-w-[720px] mx-auto px-4 sm:px-6 py-12 space-y-8">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold">{title}</h1>
          <p className="inline-block text-[12px] font-medium px-2 py-0.5 rounded-full bg-[var(--color-amber-900)] text-[var(--color-amber-500)]">
            Draft — last updated {LEGAL_LAST_UPDATED}
          </p>
        </div>
        {children}
      </main>
      <footer className="border-t border-[var(--color-border-subtle)] py-6 px-4 text-sm text-[var(--color-text-tertiary)]">
        <div className="max-w-[720px] mx-auto flex items-center justify-between gap-4">
          <span>&copy; 2026 {COMPANY_NAME}</span>
          <LegalLinks />
        </div>
      </footer>
    </div>
  )
}
