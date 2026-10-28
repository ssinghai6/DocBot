// Signup / onboarding API contracts (Zod) — DOCBOT-1712.
// Mirrors GET /api/auth/config and api/routes/onboarding_routes.py.

import { z } from "zod"

export const AuthConfigSchema = z.object({
  email: z.boolean(),
  github: z.boolean(),
  google: z.boolean(),
  saml: z.boolean(),
})
export type AuthConfig = z.infer<typeof AuthConfigSchema>

export const OAuthStartSchema = z.object({ url: z.string().url() })

export const ApiErrorSchema = z.object({ detail: z.string() })

/** The `detail` string FastAPI puts on errors, else the fallback. */
export function apiErrorDetail(data: unknown, fallback: string): string {
  const parsed = ApiErrorSchema.safeParse(data)
  return parsed.success ? parsed.data.detail : fallback
}

export const USE_CASE_OPTIONS = [
  { value: "tie_out", label: "Tie out a draft report" },
  { value: "close_reconciliation", label: "Close / reconciliation" },
  { value: "investor_reporting", label: "Investor / board reporting" },
  { value: "other", label: "Other" },
] as const

export const OnboardingRequestSchema = z
  .object({
    workspace_name: z.string().trim().max(80).optional(),
    use_case: z.enum(["tie_out", "close_reconciliation", "investor_reporting", "other"]).optional(),
  })
  .strict()

export const OnboardingResponseSchema = z.object({
  status: z.literal("ok"),
  workspace_renamed: z.boolean(),
})
