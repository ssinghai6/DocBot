// Autopilot "step" SSE event contract (Zod) — DOCBOT-2005.
// Mirrors the step event yielded by api/autopilot_service.py. ``summary`` is a
// short deterministic line ("Queried trial_balance — 12 rows"); ``content`` is
// the cleaned raw tool output shown behind a "Details" toggle. ``summary`` is
// optional so pre-2005 events still parse.

import { z } from "zod"

export const AutopilotStepEventSchema = z.object({
  type: z.literal("step"),
  step_num: z.number(),
  tool: z.string(),
  step_label: z.string().nullish(),
  summary: z.string().nullish(),
  content: z.string().nullish(),
  artifact_id: z.string().nullish(),
  chart_b64: z.string().nullish(),
  sql: z.string().nullish(),
  explanation: z.string().nullish(),
  code: z.string().nullish(),
  error: z.string().nullish(),
  // DOCBOT-2727: "insufficient_data" when the step found no usable data (DOCBOT-2724/2725).
  failure_reason: z.string().nullish(),
})

export type AutopilotStepEvent = z.infer<typeof AutopilotStepEventSchema>
