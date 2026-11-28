// DOCBOT-2729: per-message routing via POST /api/route-intent, with the old
// keyword regex kept only as the fallback when the router times out or fails.

import { RouteIntentResponseSchema } from "@/lib/schemas/intent"
import type { IntentTask, RouteIntentRequest, RouteIntentResponse } from "@/lib/schemas/intent"

export const ROUTER_TIMEOUT_MS = 1500

// Pre-2729 auto-trigger. Simple single-intent viz stays on the fast path.
const SIMPLE_VIZ = /^\s*(plot|show|graph|chart|draw|visualize|visualise|display|histogram|heatmap|render|what is|tell me)\b/i
const MULTI_STEP_TRIGGER = /\b(predict|forecast|what.?if|simulat|root.?cause|deep.?dive|break.?down|investigat|diagnos|scenario|correlat|regress|cluster|anomal|outli)\b/i
const COMPLEX_COMPARE = /\bcompar\w*\s+.+\s+(vs|versus|against|with|to)\b/i

export function regexWantsAutopilot(input: string): boolean {
  if (SIMPLE_VIZ.test(input)) return false
  return MULTI_STEP_TRIGGER.test(input) || COMPLEX_COMPARE.test(input)
}

/** The router's verdict, or null on timeout / HTTP error / schema mismatch. */
export async function fetchRouteIntent(body: RouteIntentRequest): Promise<RouteIntentResponse | null> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), ROUTER_TIMEOUT_MS)
  try {
    const res = await fetch("/api/route-intent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(body),
      signal: ctrl.signal,
    })
    if (!res.ok) return null
    const parsed = RouteIntentResponseSchema.safeParse(await res.json())
    return parsed.success ? parsed.data : null
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

const TASK_VERB: Record<IntentTask, string> = {
  lookup: "look this up",
  compare: "compare this",
  forecast: "forecast",
  what_if: "run a what-if on",
  diagnose: "diagnose",
  viz: "chart this",
}

/** Tasks that need data the sources may not have; data_ready=false blocks them. */
export function blocksOnMissingData(intent: RouteIntentResponse): boolean {
  return !intent.data_ready && (intent.task === "forecast" || intent.task === "what_if" || intent.task === "diagnose")
}

export function insufficientDataReply(intent: RouteIntentResponse): string {
  const missing = intent.missing?.trim() || "your sources have no historical values for it"
  return `I can't ${TASK_VERB[intent.task]} this yet: ${missing}. Upload a CSV, connect a database, or ingest a filing with the historical values.`
}

export function autopilotToast(task: IntentTask): string {
  return `Autopilot: ${task.replace("_", "-")} detected`
}
