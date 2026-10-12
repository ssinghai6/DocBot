"use client"

import React from "react"

export interface ChatEmptyStateProps {
  /** Icon rendered inside the fixed icon container (size/style is constant across states). */
  icon: React.ReactNode
  /** Headline — same size/weight regardless of which state triggered the empty view. */
  headline: string
  /** Subcopy under the headline — same size/treatment regardless of state. */
  subcopy: string
  /**
   * The only region that varies between empty states: action cards when no
   * source is connected, or suggestion chips once a source exists but no
   * messages have been sent yet. Keeping this as a single children slot
   * means the icon/headline/subcopy/layout shell never visually changes
   * when the user transitions from "no source" to "source connected".
   */
  children: React.ReactNode
}

/**
 * Unified empty-state shell for ChatArea (DOCBOT-1532).
 *
 * Previously ChatArea rendered two structurally different empty states
 * (different icon treatment, different headline size, different content
 * shape) depending on whether a source was connected. That meant a user
 * uploading a PDF saw the whole screen restructure mid-flow. This component
 * fixes the frame — icon container, headline, subcopy, max-width, vertical
 * spacing — so only the content below (passed as children) changes.
 */
export default function ChatEmptyState({ icon, headline, subcopy, children }: ChatEmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="w-12 h-12 rounded-[8px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/30 flex items-center justify-center mb-4">
        {icon}
      </div>
      <p className="text-[16px] font-semibold text-[var(--color-text-primary)] mb-1 text-center">{headline}</p>
      <p className="text-[12px] text-[var(--color-text-tertiary)] max-w-lg text-center mb-6">{subcopy}</p>
      {children}
    </div>
  )
}
