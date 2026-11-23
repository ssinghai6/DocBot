// DOCBOT-3905: same-origin fetch helper for the /apis developer portal.
// /api/developer/* uses the session cookie; /v1/* uses the in-memory test key.

import type { z } from "zod"
import { parseApiError, type ParsedApiError } from "@/lib/schemas/developer"

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ParsedApiError }

/** Fetch + Zod-validate the JSON body; errors come back parsed for display. */
export async function apiCall<S extends z.ZodType>(
  url: string,
  init: RequestInit & { bearer?: string },
  schema: S,
): Promise<ApiResult<z.infer<S>>> {
  const { bearer, ...rest } = init
  const headers = new Headers(rest.headers)
  if (bearer) headers.set("Authorization", `Bearer ${bearer}`)
  if (rest.body) headers.set("Content-Type", "application/json")
  let res: Response
  try {
    res = await fetch(url, { ...rest, headers, credentials: "include", cache: "no-store" })
  } catch {
    return { ok: false, error: { code: null, message: "Network error. Check your connection and try again.", requestId: null } }
  }
  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) return { ok: false, error: parseApiError(body, res.status) }
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return { ok: false, error: { code: "bad_response", message: "Unexpected response from the server.", requestId: null } }
  }
  return { ok: true, data: parsed.data }
}
