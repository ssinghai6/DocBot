import { z } from "zod"
import { parseQuotaError } from "@/lib/schemas/billing"
import type { QuotaErrorPayload } from "@/lib/schemas/billing"

/**
 * DOCBOT-2305: one classifier for every refused API response, so chat,
 * uploads, DB connect, demo init, EDGAR and connectors all react the same way.
 * Merges DOCBOT-1711 (sign-in gate), DOCBOT-1615 (platform capacity) and
 * DOCBOT-1805 (paywall).
 *
 *   signin   401; 403 asking to sign in / verify email; 429 "sign in (free)"
 *   rate     other 429 — backend `detail` shown verbatim
 *   credits  402 with a structured quota payload -> OutOfCreditsModal
 *   capacity 503 {detail:{error:"platform_capacity"}} — sticky toast, never retried
 *   other    everything else
 */
export type HttpGate =
  | { kind: "signin"; message: string }
  | { kind: "rate"; message: string }
  | { kind: "credits"; payload: QuotaErrorPayload }
  | { kind: "capacity"; message: string }
  | { kind: "other"; status: number; message: string }

const DetailSchema = z.object({ detail: z.string() })
const CapacitySchema = z.object({
  detail: z.object({ error: z.literal("platform_capacity"), message: z.string() }),
})

const SIGN_IN_403 = /sign[ -]?in|verify your email/i
const SIGN_IN_429 = /sign in \(free\)/i

/** Pure: classify a refused response from its status and parsed JSON body. */
export function classifyGate(status: number, body: unknown): HttpGate {
  const parsedDetail = DetailSchema.safeParse(body)
  const detail = parsedDetail.success ? parsedDetail.data.detail : ""
  if (status === 401) return { kind: "signin", message: detail || "Sign in (free) to continue." }
  if (status === 403 && SIGN_IN_403.test(detail)) return { kind: "signin", message: detail }
  const quota = parseQuotaError(body) // structured billing detail (402/429/403-unverified)
  if (status === 403 && quota?.error === "email_unverified") return { kind: "signin", message: quota.message }
  if (status === 429) {
    if (SIGN_IN_429.test(detail)) return { kind: "signin", message: detail }
    const message = quota?.message ?? detail
    return { kind: "rate", message: message || "Too many requests. Please try again shortly." }
  }
  if (status === 402 && quota) return { kind: "credits", payload: quota }
  if (status === 503) {
    const capacity = CapacitySchema.safeParse(body)
    if (capacity.success) return { kind: "capacity", message: capacity.data.detail.message }
  }
  return { kind: "other", status, message: detail || `HTTP ${status}` }
}

/** Classify a non-OK response (null when OK). Reads a clone, so the body stays unread. */
export async function classifyResponse(response: Response): Promise<HttpGate | null> {
  if (response.ok) return null
  const body: unknown = await response.clone().json().catch(() => null)
  return classifyGate(response.status, body)
}

/** Thrown after a gate has already been shown to the user — callers skip their own error toast. */
export class HttpGateError extends Error {}
export class SignInGatedError extends HttpGateError {}
export class PlatformCapacityError extends HttpGateError {}

// The chat page registers one listener that maps gates to UI (AuthModal, toast, paywall).
type GateListener = (gate: HttpGate) => void
let gateListener: GateListener | null = null

export function onHttpGate(listener: GateListener): () => void {
  gateListener = listener
  return () => {
    if (gateListener === listener) gateListener = null
  }
}

export function reportGate(gate: HttpGate): void {
  gateListener?.(gate)
}

export function gateMessage(gate: HttpGate): string {
  return gate.kind === "credits" ? gate.payload.message : gate.message
}

/**
 * Throw for any non-OK response. Sign-in / rate / credits / capacity gates are
 * shown via the page listener first and thrown as HttpGateError; anything else
 * throws a plain Error carrying the backend `detail` (or `fallback`).
 */
export async function ensureOk(response: Response, fallback: string): Promise<void> {
  const gate = await classifyResponse(response)
  if (gate === null) return
  if (gate.kind === "other") throw new Error(gate.message.startsWith("HTTP ") ? fallback : gate.message)
  reportGate(gate)
  throw new HttpGateError(gateMessage(gate))
}

/** True (and shown via the page listener) when the response is a sign-in / rate / credits / capacity gate. */
export async function gateResponse(response: Response): Promise<boolean> {
  const gate = await classifyResponse(response)
  if (gate === null || gate.kind === "other") return false
  reportGate(gate)
  return true
}
