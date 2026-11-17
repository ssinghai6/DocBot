// Waitlist API contracts (Zod) — DOCBOT-2581.
//
// Mirrors api/waitlist_models.py field-for-field (same snake_case names).

import { z } from "zod"

export const WaitlistSourceSchema = z.enum(["out_of_credits", "landing"])
export type WaitlistSource = z.infer<typeof WaitlistSourceSchema>

export const WaitlistRequestSchema = z
  .object({
    email: z.string().trim().max(254).regex(/^[^@\s]+@[^@\s]+\.[^@\s]+$/, "Enter a valid email."),
    name: z.string().trim().min(1, "Name is required.").max(120),
    company: z.string().trim().min(1, "Company is required.").max(120),
    role: z.string().max(120).optional(),
    note: z.string().max(1000).optional(),
    source: WaitlistSourceSchema,
    // Honeypot: always empty from the real form.
    website: z.string().max(500).optional(),
  })
  .strict()
export type WaitlistRequest = z.infer<typeof WaitlistRequestSchema>

export const WaitlistResponseSchema = z.object({ ok: z.literal(true) })

export const WaitlistEntrySchema = z.object({
  id: z.string(),
  email: z.string(),
  name: z.string(),
  company: z.string(),
  role: z.string().nullable().optional(),
  note: z.string().nullable().optional(),
  source: WaitlistSourceSchema,
  user_id: z.string().nullable().optional(),
  org_id: z.string().nullable().optional(),
  created_at: z.string(),
  updated_at: z.string(),
})

export const WaitlistListSchema = z.object({ entries: z.array(WaitlistEntrySchema) })
