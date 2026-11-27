// Dropped-answer recovery contract (Zod) — DOCBOT-2636.
//
// Mirrors api/answer_models.py field-for-field (same snake_case names):
// GET /api/hybrid/answer/{run_id} -> RecoveredAnswer.

import { z } from "zod"
import { LineageSchema } from "@/components/types"

export const AnswerCitationSchema = z.object({
  source: z.string().default(""),
  page: z.number().nullish(),
  text: z.string().default(""),
})

export const RecoveredAnswerSchema = z.object({
  run_id: z.string(),
  status: z.enum(["pending", "ready"]),
  answer: z.string().nullish(),
  citations: z.array(AnswerCitationSchema).default([]),
  trace_id: z.string().nullish(),
  lineage: LineageSchema.nullish(),
})
export type RecoveredAnswer = z.infer<typeof RecoveredAnswerSchema>
