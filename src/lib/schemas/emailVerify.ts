// Email verification + Turnstile API contracts (Zod) — DOCBOT-1713.
//
// Mirrors api/email_verify_models.py field-for-field (same snake_case names).
// The signup request shape (RegisterRequestSchema) mirrors the inline
// `RegisterRequest` model in api/index.py, which gained the optional
// `turnstile_token` field for Cloudflare bot protection.
//
// Response schemas use z.object (unknown keys stripped); the request schema
// is .strict() to keep accidental fields out of the signup POST.

import { z } from "zod"

// ── Signup (POST /api/auth/register) ─────────────────────────────────────

export const RegisterRequestSchema = z
  .object({
    email: z.string(),
    password: z.string(),
    name: z.string().optional(),
    turnstile_token: z.string().optional(),
    // DOCBOT-1707: register only; the backend refuses signup unless true.
    accepted_terms: z.boolean().optional(),
  })
  .strict()

export const RegisterResponseSchema = z.object({
  status: z.enum(["registered", "verification_required"]),
  email: z.string(),
})
export type RegisterResponse = z.infer<typeof RegisterResponseSchema>

// ── Link confirmation (POST /api/auth/verify) ─────────────────────────────

export const VerifyRequestSchema = z
  .object({
    token: z.string().min(1, "Missing verification token."),
  })
  .strict()

export const VerifyResponseSchema = z.object({
  status: z.literal("verified"),
  email: z.string(),
})
export type VerifyResponse = z.infer<typeof VerifyResponseSchema>
