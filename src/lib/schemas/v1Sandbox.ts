// POST /v1/sandbox/sessions contracts (Zod) — DOCBOT-2597.
//
// Mirrors api/v1_sandbox_models.py field-for-field (same snake_case names).

import { z } from "zod"

export const SandboxDatasetSchema = z.enum(["quickbite", "fuel", "tieout"])

export const V1SandboxSessionRequestSchema = z
  .object({
    dataset: SandboxDatasetSchema.default("quickbite"),
  })
  .strict()

export const V1SandboxSessionResponseSchema = z.object({
  session_id: z.string(),
  connection_id: z.string(),
  dataset: SandboxDatasetSchema,
  expires_at: z.iso.datetime({ offset: true }),
})
export type V1SandboxSessionResponse = z.infer<typeof V1SandboxSessionResponseSchema>
