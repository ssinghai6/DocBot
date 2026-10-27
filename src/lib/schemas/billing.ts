// Billing API contracts (Zod) — DOCBOT-1800, EPIC-18.
//
// Mirrors api/billing_models.py field-for-field (same snake_case names).
// tests/unit/test_billing_models.py fails if a Pydantic field is missing here.
//
// Units (see api/billing_models.py docstring):
//   * credits are integers — 1 credit = $0.001 retail
//   * fiat prices are integer cents + lowercase ISO-4217 currency
//   * timestamps are ISO-8601 strings with offset
// Server-only models (CreditPackConfig, UsageEventRecord) are intentionally
// not mirrored: they carry Stripe price ids / raw cost.
//
// Response schemas use z.object (unknown keys stripped) so additive backend
// changes don't break old clients; request schemas are .strict() to match
// Pydantic extra="forbid".

import { z } from "zod"

// ── Units ──────────────────────────────────────────────────────────────────

export const CREDITS_PER_USD = 1_000
export const MICRO_USD_PER_CREDIT = 1_000

const credits = z.number().int().nonnegative()
const signedCredits = z.number().int()
const timestamp = z.iso.datetime({ offset: true })
const currencyCode = z.string().regex(/^[a-z]{3}$/)
const packId = z.string().regex(/^[a-z0-9][a-z0-9_-]{0,63}$/)
// Same-origin relative path: leading "/" not followed by "/" or "\".
const relativePath = z.string().max(512).regex(/^\/(?![/\\])[A-Za-z0-9/_\-?=&.%]*$/)

// ── Enums ──────────────────────────────────────────────────────────────────

export const LedgerEntryKindSchema = z.enum([
  "free_grant",
  "purchase",
  "usage",
  "refund",
  "dispute_hold",
  "dispute_release",
  "adjustment",
])
export type LedgerEntryKind = z.infer<typeof LedgerEntryKindSchema>

export const UsageEventKindSchema = z.enum(["llm", "sandbox", "upload", "request"])
export type UsageEventKind = z.infer<typeof UsageEventKindSchema>

export const KeySourceSchema = z.enum(["platform", "byok"])
export type KeySource = z.infer<typeof KeySourceSchema>

export const QuotaErrorCodeSchema = z.enum([
  "insufficient_credits",
  "demo_quota_exceeded",
  "rate_limited",
  "email_unverified",
  "account_frozen",
  "platform_capacity",
])
export type QuotaErrorCode = z.infer<typeof QuotaErrorCodeSchema>

// ── Balance ────────────────────────────────────────────────────────────────

export const CreditBalanceSchema = z.object({
  org_id: z.string(),
  balance_credits: credits,
  reserved_credits: credits,
  available_credits: credits,
  free_monthly_credits: credits,
  free_grant_period: z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/).nullable(),
  frozen: z.boolean(),
  updated_at: timestamp,
})
export type CreditBalance = z.infer<typeof CreditBalanceSchema>

// ── Packs / catalog ────────────────────────────────────────────────────────

export const CreditPackSchema = z.object({
  id: packId,
  name: z.string().min(1).max(80),
  credits: z.number().int().positive(),
  price_cents: z.number().int().positive(),
  currency: currencyCode,
  description: z.string().max(280).nullable(),
  highlight: z.boolean(),
})
export type CreditPack = z.infer<typeof CreditPackSchema>

export const CreditPackCatalogSchema = z.object({
  packs: z.array(CreditPackSchema),
  currency: currencyCode,
  free_monthly_credits: credits,
  credit_usd_value: z.literal("0.001"),
})
export type CreditPackCatalog = z.infer<typeof CreditPackCatalogSchema>

// ── Checkout ───────────────────────────────────────────────────────────────

// Request: org_id is never sent — the server derives it from the session.
export const CheckoutSessionRequestSchema = z
  .object({
    pack_id: packId,
    return_path: relativePath.nullable().optional(),
  })
  .strict()
export type CheckoutSessionRequest = z.infer<typeof CheckoutSessionRequestSchema>

export const CheckoutSessionResponseSchema = z.object({
  checkout_url: z.string().regex(/^https:\/\/checkout\.stripe\.com\//),
  session_id: z.string().regex(/^cs_(test|live)_[A-Za-z0-9]+$/),
  expires_at: timestamp.nullable(),
})
export type CheckoutSessionResponse = z.infer<typeof CheckoutSessionResponseSchema>

// ── Ledger ─────────────────────────────────────────────────────────────────

export const LedgerEntrySchema = z.object({
  id: z.string(),
  org_id: z.string(),
  kind: LedgerEntryKindSchema,
  amount_credits: signedCredits,
  balance_after_credits: z.number().int(),
  ref: z.string().min(1).max(255),
  description: z.string().max(280).nullable(),
  actor_user_id: z.string().nullable(),
  created_at: timestamp,
})
export type LedgerEntry = z.infer<typeof LedgerEntrySchema>

export const LedgerPageSchema = z.object({
  entries: z.array(LedgerEntrySchema),
  next_cursor: z.string().nullable(),
})
export type LedgerPage = z.infer<typeof LedgerPageSchema>

// ── Usage ──────────────────────────────────────────────────────────────────

export const UsageEventSchema = z.object({
  id: z.string(),
  org_id: z.string(),
  user_id: z.string().nullable(),
  kind: UsageEventKindSchema,
  key_source: KeySourceSchema,
  run_id: z.string().nullable(),
  model: z.string().nullable(),
  prompt_tokens: z.number().int().nonnegative(),
  completion_tokens: z.number().int().nonnegative(),
  sandbox_ms: z.number().int().nonnegative(),
  credits_debited: credits,
  created_at: timestamp,
})
export type UsageEvent = z.infer<typeof UsageEventSchema>

export const UsageBreakdownItemSchema = z.object({
  kind: UsageEventKindSchema,
  event_count: z.number().int().nonnegative(),
  credits_used: credits,
})
export type UsageBreakdownItem = z.infer<typeof UsageBreakdownItemSchema>

export const UsageSummarySchema = z.object({
  org_id: z.string(),
  period_start: timestamp,
  period_end: timestamp,
  credits_used: credits,
  credits_granted: credits,
  credits_purchased: credits,
  event_count: z.number().int().nonnegative(),
  by_kind: z.array(UsageBreakdownItemSchema),
})
export type UsageSummary = z.infer<typeof UsageSummarySchema>

export const UsagePageSchema = z.object({
  events: z.array(UsageEventSchema),
  next_cursor: z.string().nullable(),
})
export type UsagePage = z.infer<typeof UsagePageSchema>

// ── Stripe webhook + errors ────────────────────────────────────────────────

export const StripeWebhookAckSchema = z.object({
  received: z.literal(true),
  event_id: z.string().regex(/^evt_[A-Za-z0-9]+$/),
  duplicate: z.boolean(),
  handled: z.boolean(),
})
export type StripeWebhookAck = z.infer<typeof StripeWebhookAckSchema>

export const QuotaErrorPayloadSchema = z.object({
  error: QuotaErrorCodeSchema,
  message: z.string().min(1).max(500),
  required_credits: credits.nullable(),
  available_credits: credits.nullable(),
  retry_after_seconds: z.number().int().nonnegative().nullable(),
  action_path: relativePath.nullable(),
})
export type QuotaErrorPayload = z.infer<typeof QuotaErrorPayloadSchema>

// Full error body: FastAPI wraps HTTPException.detail as { detail: ... }.
export const QuotaErrorResponseSchema = z.object({
  detail: QuotaErrorPayloadSchema,
})
export type QuotaErrorResponse = z.infer<typeof QuotaErrorResponseSchema>

/**
 * Parse a non-2xx billing/quota response body. Returns null when the body is
 * not a structured quota error (e.g. a plain `{detail: "..."}` string), so
 * callers can fall back to generic error handling (DOCBOT-1805 / 2305).
 */
export function parseQuotaError(body: unknown): QuotaErrorPayload | null {
  const result = QuotaErrorResponseSchema.safeParse(body)
  return result.success ? result.data.detail : null
}
