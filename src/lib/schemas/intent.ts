// Intent router contract (Zod) — DOCBOT-2728.
//
// Mirrors api/intent_models.py field-for-field (same snake_case names):
// POST /api/route-intent (RouteIntentRequest) -> RouteIntentResponse.

import { z } from "zod"

export const IntentTaskSchema = z.enum(["lookup", "compare", "forecast", "what_if", "diagnose", "viz"])
export type IntentTask = z.infer<typeof IntentTaskSchema>

export const RouteIntentRequestSchema = z.object({
  question: z.string().min(1).max(4000),
  history: z.array(z.object({ role: z.string(), content: z.string() })).max(20).default([]),
  session_id: z.string().nullish(),
  connection_id: z.string().nullish(),
})
export type RouteIntentRequest = z.infer<typeof RouteIntentRequestSchema>

export const RouteIntentResponseSchema = z.object({
  route: z.enum(["fast", "autopilot"]),
  source: z.enum(["doc", "sql", "hybrid"]),
  task: IntentTaskSchema,
  data_ready: z.boolean(),
  missing: z.string().nullish(),
  confidence: z.number().min(0).max(1),
})
export type RouteIntentResponse = z.infer<typeof RouteIntentResponseSchema>
