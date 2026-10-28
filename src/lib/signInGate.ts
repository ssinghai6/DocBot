import { z } from "zod"

/**
 * DOCBOT-1711: the backend asks an anonymous visitor to sign in with a 401
 * (APP_AUTH_MODE=required, own data) or a 429 whose detail says
 * "sign in (free)" (DEMO_DAILY_USD_CAP reached). Both should open the
 * AuthModal instead of surfacing as a raw HTTP error.
 */
const ErrorDetailSchema = z.object({ detail: z.string() })

export class SignInGatedError extends Error {}

/** The sign-in prompt for a gated response, else null. Does not consume the body. */
export async function signInGateMessage(response: Response): Promise<string | null> {
  if (response.status !== 401 && response.status !== 429) return null
  const body: unknown = await response.clone().json().catch(() => null)
  const parsed = ErrorDetailSchema.safeParse(body)
  const detail = parsed.success ? parsed.data.detail : ""
  if (response.status === 429 && !/sign in \(free\)/i.test(detail)) return null
  return detail || "Sign in (free) to continue."
}
