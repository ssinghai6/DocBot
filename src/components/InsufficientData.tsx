// DOCBOT-2727: autopilot step that found no usable data — amber chip, reason,
// and CTAs into the existing upload / connect-database UI. No chart is shown.

import { Database, Upload } from "lucide-react"
import type { AutopilotStep } from "@/components/types"
import { insufficientReason } from "@/lib/autopilotStep"

export type DataCtaHandlers = {
  onUploadClick?: () => void
  onConnectDatabase?: () => void
}

export function InsufficientDataChip() {
  return (
    <span className="shrink-0 px-1.5 py-0.5 rounded-[3px] text-[9px] font-semibold uppercase tracking-wider bg-[var(--color-amber-500)]/15 text-[var(--color-amber-500)] border border-[var(--color-amber-500)]/30">
      Insufficient data
    </span>
  )
}

const CTA_CLASS =
  "inline-flex items-center gap-1 h-6 px-2 rounded-[3px] text-[10px] font-semibold border border-[var(--color-amber-500)]/40 text-[var(--color-amber-500)] hover:bg-[var(--color-amber-500)]/10 transition-colors"

export function InsufficientDataBody({ step, onUploadClick, onConnectDatabase }: { step: AutopilotStep } & DataCtaHandlers) {
  return (
    <div className="mt-1">
      <p className="text-[var(--color-text-secondary)] leading-relaxed">{insufficientReason(step)}</p>
      {(onUploadClick || onConnectDatabase) && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {onUploadClick && (
            <button type="button" onClick={onUploadClick} className={CTA_CLASS}>
              <Upload className="w-3 h-3" /> Upload CSV
            </button>
          )}
          {onConnectDatabase && (
            <button type="button" onClick={onConnectDatabase} className={CTA_CLASS}>
              <Database className="w-3 h-3" /> Connect DB
            </button>
          )}
        </div>
      )}
    </div>
  )
}
