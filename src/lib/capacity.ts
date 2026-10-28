import { z } from "zod"

/**
 * DOCBOT-1615: once the platform daily spend cap is hit, free-tier callers get
 * HTTP 503 with `{"detail": {"error": "platform_capacity", "message": ...}}`.
 * Surface that message instead of a raw "HTTP 503" (and never retry it).
 */
const CapacitySchema = z.object({
  detail: z.object({ error: z.literal("platform_capacity"), message: z.string() }),
})

export class PlatformCapacityError extends Error {}

/** Throws PlatformCapacityError for a capacity 503. Does not consume the body. */
export async function throwIfAtCapacity(response: Response): Promise<void> {
  if (response.status !== 503) return
  const body: unknown = await response.clone().json().catch(() => null)
  const parsed = CapacitySchema.safeParse(body)
  if (parsed.success) throw new PlatformCapacityError(parsed.data.detail.message)
}
