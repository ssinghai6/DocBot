// Owner unlock API contract (Zod) — DOCBOT-1605.
// Mirrors OwnerUnlockResponse in api/index.py (POST /api/auth/owner-unlock).

import { z } from "zod"

export const OwnerUnlockResponseSchema = z.object({ owner: z.boolean() })
