// DOCBOT-1528: small shared "relative time" + staleness-bucket helper used by
// ConnectionPanel (schema sync) and MarketplacePanel (connector sync badges).

/** Formats an ISO timestamp as "3m ago", "2h ago", "5d ago", etc. Returns
 * "never" for null/undefined, and "just now" for anything under a minute. */
export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "never"
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return "never"
  const diffMs = Date.now() - then
  if (diffMs < 0) return "just now"

  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour

  if (diffMs < minute) return "just now"
  if (diffMs < hour) {
    const m = Math.floor(diffMs / minute)
    return `${m}m ago`
  }
  if (diffMs < day) {
    const h = Math.floor(diffMs / hour)
    return `${h}h ago`
  }
  const d = Math.floor(diffMs / day)
  return `${d}d ago`
}

export type StalenessLevel = "fresh" | "stale" | "very_stale" | "unknown"

/** Buckets a timestamp into a freshness level for color-coding:
 *   fresh      — under 1 hour old (green)
 *   stale      — 1-24 hours old (amber)
 *   very_stale — over 24 hours old (red)
 *   unknown    — no timestamp, or an explicit failed status (red) */
export function getStalenessLevel(
  iso: string | null | undefined,
  failed = false
): StalenessLevel {
  if (failed) return "very_stale"
  if (!iso) return "unknown"
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return "unknown"
  const diffMs = Date.now() - then
  const hour = 60 * 60_000
  if (diffMs < hour) return "fresh"
  if (diffMs < 24 * hour) return "stale"
  return "very_stale"
}

export const STALENESS_COLOR_CLASSES: Record<StalenessLevel, string> = {
  fresh: "bg-[var(--color-success-500)]/15 text-[var(--color-success-500)] border-[var(--color-success-500)]/30",
  stale: "bg-[var(--color-amber-500)]/15 text-[var(--color-amber-500)] border-[var(--color-amber-500)]/30",
  very_stale: "bg-[var(--color-danger-500)]/15 text-[var(--color-danger-500)] border-[var(--color-danger-500)]/30",
  unknown: "bg-[#ffffff10] text-[var(--color-text-tertiary)] border-[var(--color-border-default)]",
}
