// Self-serve test API key contracts (Zod) — DOCBOT-2598.
//
// Mirrors api/developer_models.py field-for-field (same snake_case names).
// `key` (the raw dk_test_ secret) only appears in the create response; hold it
// in memory, never persist it.

import { z } from "zod"

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
