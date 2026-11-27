// Self-serve test API key contracts (Zod) — DOCBOT-2598.
//
// Mirrors api/developer_models.py field-for-field (same snake_case names).
// `key` (the raw dk_test_ secret) only appears in the create response; hold it
// in memory, never persist it.

import { z } from "zod"
import { V1ErrorResponseSchema } from "./v1"

const timestamp = z.iso.datetime({ offset: true })

export const TestKeySummarySchema = z.object({
  id: z.string(),
  prefix: z.string(),
  scopes: z.array(z.string()),
  created_at: timestamp,
  expires_at: timestamp.nullable(),
  last_used_at: timestamp.nullable(),
})
export type TestKeySummary = z.infer<typeof TestKeySummarySchema>

export const TestKeyCreateResponseSchema = TestKeySummarySchema.extend({
  key: z.string().startsWith("dk_test_"),
  mode: z.literal("test"),
})
export type TestKeyCreateResponse = z.infer<typeof TestKeyCreateResponseSchema>

export const TestKeyListResponseSchema = z.object({
  keys: z.array(TestKeySummarySchema),
})
export type TestKeyListResponse = z.infer<typeof TestKeyListResponseSchema>

export const TestKeyRevokeResponseSchema = z.object({
  status: z.literal("revoked"),
  id: z.string(),
})
export type TestKeyRevokeResponse = z.infer<typeof TestKeyRevokeResponseSchema>

// ── Errors (DOCBOT-3905 /apis) ───────────────────────────────────────────
// /api/developer/*: FastAPI {"detail": {code, message}} (or a plain string).
// /v1/*: the shared envelope, V1ErrorResponseSchema in ./v1.

export const ApiDetailErrorSchema = z.object({
  detail: z.union([z.string(), z.object({ code: z.string(), message: z.string() })]),
})

export type ParsedApiError = { code: string | null; message: string; requestId: string | null }

const FRIENDLY: Record<string, string> = {
  email_not_verified: "Verify your email address first, then try again. Check your inbox for the verification link.",
  test_key_limit: "You already have 2 active test keys. Revoke one below to create another.",
}

/** Turn any backend error body into a code + human message for display. */
export function parseApiError(data: unknown, status: number): ParsedApiError {
  const v1 = V1ErrorResponseSchema.safeParse(data)
  if (v1.success) {
    const { code, message, request_id } = v1.data.error
    return { code, message: FRIENDLY[code] ?? message, requestId: request_id }
  }
  const api = ApiDetailErrorSchema.safeParse(data)
  if (api.success) {
    const d = api.data.detail
    if (typeof d === "string") return { code: null, message: d, requestId: null }
    return { code: d.code, message: FRIENDLY[d.code] ?? d.message, requestId: null }
  }
  return { code: null, message: `Request failed (HTTP ${status}).`, requestId: null }
}
