"use client"

import type { ReactNode } from "react"

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
      {children}
    </div>
  )
}

export function EmptyTab({ label, hint }: { label: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center h-full min-w-0">
      <p className="text-[13px] text-[var(--color-text-secondary)] mb-1">No {label.toLowerCase()} yet</p>
      <p className="text-[11px] text-[var(--color-text-tertiary)] max-w-[240px] leading-relaxed break-words">{hint}</p>
    </div>
  )
}

type ChipTone = "neutral" | "cyan" | "amber" | "red" | "green"

const TONE: Record<ChipTone, string> = {
  neutral: "text-[var(--color-text-tertiary)] border-[var(--color-border-subtle)]",
  cyan: "text-[var(--color-cyan-500)] border-[var(--color-cyan-500)]/40",
  amber: "text-[var(--color-amber-500)] border-[var(--color-amber-500)]/40",
  red: "text-red-400 border-red-400/40",
  green: "text-emerald-400 border-emerald-400/40",
}

export function Chip({ children, tone = "neutral", title }: { children: ReactNode; tone?: ChipTone; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 text-[10px] px-1.5 h-[18px] rounded-[3px] border bg-[var(--color-bg-inset)] whitespace-nowrap ${TONE[tone]}`}
    >
      {children}
    </span>
  )
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "–"
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n == null) return "–"
  return Math.abs(n) >= 1000
    ? n.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : n.toLocaleString(undefined, { maximumFractionDigits: digits })
}

export function fmtUsd(n: number | null | undefined): string {
  if (n == null) return "–"
  return n < 0.01 ? `<$0.01` : `$${n.toFixed(2)}`
}

export function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="flex items-center gap-1.5" title={`${label}: ${value.toFixed(3)}`}>
      <span className="text-[9px] uppercase tracking-wider text-[var(--color-text-quaternary)] w-12 flex-none">{label}</span>
      <div className="flex-1 h-1 rounded-full bg-[var(--color-border-subtle)] overflow-hidden">
        <div className="h-full bg-[var(--color-cyan-500)]" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[9px] tabular-nums text-[var(--color-text-tertiary)] w-8 text-right">{value.toFixed(2)}</span>
    </div>
  )
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard?.writeText(text)
      }}
      className="text-[11px] px-2.5 h-7 rounded-[5px] border border-[var(--color-border-default)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-cyan-500)] transition-colors"
    >
      {label}
    </button>
  )
}
