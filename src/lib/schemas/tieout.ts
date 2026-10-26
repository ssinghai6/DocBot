// TieOutCard API contract (Zod) — DOCBOT-2002.
//
// Mirrors api/utils/lineage.py::LineageDiscrepancy field-for-field (same
// snake_case names): one structured report-vs-books tie-out per card.
//   * ``report_value`` is the figure in the document (filing / report),
//     ``books_value`` the one in the connected database.
//   * ``delta`` = books - report; ``delta_pct`` is relative to the report
//     figure.
//   * ``doc_citation`` / ``db_citation`` carry the two sources every card
//     cites.
// Pre-2002 persisted rows ({label, doc_value, db_value, pct}) stay parseable:
// their keys are kept as optional legacy fallbacks so old lineage payloads
// (safeParse'd best-effort client-side) never vanish wholesale.

import { z } from "zod"

export const TieOutSeveritySchema = z.enum(["low", "medium", "high"])
export const TieOutUnitSchema = z.enum(["currency", "percent", "number"])

export const TieOutDocCitationSchema = z.object({
  file: z.string().nullish(),
  page: z.number().nullish(),
})

export const TieOutDbCitationSchema = z.object({
  table: z.string().nullish(),
  query: z.string().nullish(),
})

export const TieOutCardSchema = z.object({
  id: z.string().default(""),
  metric: z.string().nullish(),
  period: z.string().nullish(),
  report_value: z.number().nullish(),
  books_value: z.number().nullish(),
  delta: z.number().nullish(),
  delta_pct: z.number().nullish(),
  severity: TieOutSeveritySchema.default("medium"),
  unit: TieOutUnitSchema.default("number"),
  doc_citation: TieOutDocCitationSchema.default({}),
  db_citation: TieOutDbCitationSchema.default({}),
  // Legacy (pre-2002) keys — tolerated on read, never emitted by new backends.
  label: z.string().nullish(),
  doc_value: z.number().nullish(),
  db_value: z.number().nullish(),
  pct: z.number().nullish(),
})

export type TieOutCard = z.infer<typeof TieOutCardSchema>

/** Display label across legacy and structured rows. */
export function tieOutMetric(card: TieOutCard): string {
  return card.metric ?? card.label ?? "Discrepancy"
}

/** Inline marker the backend answer text carries for this card. */
export function tieOutMarker(card: TieOutCard): string {
  return card.id ? `[[discrepancy:${card.id}]]` : ""
}

export const TIE_OUT_MARKER_RE = /\[\[discrepancy:(d-[0-9a-f]{8})\]\]/g
