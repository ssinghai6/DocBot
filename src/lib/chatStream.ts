import type { Citation, Lineage, Message } from "@/components/types"

/** DOCBOT-2303: shown when a chat stream ends with no tokens and no error. */
export const EMPTY_ANSWER_MESSAGE = "No answer was returned — please retry."

/**
 * DOCBOT-2303: an SSE `error` event. Thrown from inside the per-line parse
 * try/catch, which must re-throw it instead of skipping it as a malformed line.
 */
export class StreamErrorEvent extends Error {}

/**
 * Put `content` into the trailing assistant bubble if it is still empty
 * (no text, no charts). Bubbles that already show something are left alone.
 */
export function fillEmptyAssistant(messages: Message[], content: string): Message[] {
  const last = messages[messages.length - 1]
  if (!last || last.role !== "assistant" || last.content.trim() !== "" || (last.charts?.length ?? 0) > 0) {
    return messages
  }
  return [...messages.slice(0, -1), { ...last, content }]
}

/** DOCBOT-2635: shown when the answer stream breaks off (network drop, proxy reset). */
export function streamDroppedNote(error: unknown): string {
  const cause = error instanceof Error ? `${error.name}: ${error.message}` : String(error)
  return `The connection dropped before the full answer arrived (${cause}). Partial results shown here were received; send the question again for the full answer.`
}

/**
 * DOCBOT-2635: put a failure note on the trailing assistant bubble without
 * discarding what already streamed in (answer text, SQL, code, charts).
 */
export function keepPartialAnswer(messages: Message[], note: string): Message[] {
  const last = messages[messages.length - 1]
  if (!last || last.role !== "assistant") return messages
  const content = last.content.trim() ? `${last.content}\n\n> ${note}` : note
  return [...messages.slice(0, -1), { ...last, content }]
}

/** DOCBOT-2636: shown while a dropped hybrid answer is fetched from the backend. */
export const RECOVERING_MESSAGE = "Connection dropped, recovering answer…"

/** DOCBOT-2636: a finished answer read back from `GET /api/hybrid/answer/{run_id}`. */
export type RecoveredReply = {
  answer: string
  citations: Citation[]
  lineage: Lineage | null
  traceId: string | null
}

/** One poll: the answer, `"pending"` while the turn still runs, or null (gone / not the caller's). */
export type RecoveryPoll = () => Promise<RecoveredReply | "pending" | null>

/**
 * DOCBOT-2636: poll until the backend has persisted the dropped answer.
 * Null when it never shows up within `timeoutMs`; the caller then keeps the partial answer.
 */
export async function pollRecoveredAnswer(
  poll: RecoveryPoll,
  { intervalMs = 2000, timeoutMs = 60000, sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms)) } = {},
): Promise<RecoveredReply | null> {
  for (let waited = 0; ; waited += intervalMs) {
    const result = await poll()
    if (result !== "pending") return result
    if (waited + intervalMs > timeoutMs) return null
    await sleep(intervalMs)
  }
}

/** DOCBOT-2636: swap the recovered answer into the trailing assistant bubble (SQL, code, charts kept). */
export function applyRecoveredAnswer(messages: Message[], reply: RecoveredReply): Message[] {
  const last = messages[messages.length - 1]
  if (!last || last.role !== "assistant") return messages
  return [...messages.slice(0, -1), {
    ...last,
    content: reply.answer,
    citations: reply.citations.length > 0 ? reply.citations : last.citations,
    lineage: reply.lineage ?? last.lineage,
    traceId: reply.traceId ?? last.traceId,
  }]
}

const HISTORY_TURNS = 6
const HISTORY_MAX_CHARS = 4000
const MD_DATA_IMAGE = /!\[[^\]]*\]\(\s*data:image\/[^)]*\)/gi
const DATA_IMAGE = /data:image\/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]*/gi

/** Drop inline base64 images/charts from a history message, then cap its length. */
export function trimHistoryContent(content: string): string {
  return content
    .replace(MD_DATA_IMAGE, "[chart]")
    .replace(DATA_IMAGE, "[chart]")
    .slice(0, HISTORY_MAX_CHARS)
}

/**
 * DOCBOT-2305: last 6 messages (3 turns) of follow-up context for every chat
 * path, as `{role, content}` only — charts/citations never leave the browser.
 */
export function buildRecentHistory(messages: Message[]): { role: Message["role"]; content: string }[] {
  return messages
    .filter(m => m.content.trim())
    .slice(-HISTORY_TURNS)
    .map(m => ({ role: m.role, content: trimHistoryContent(m.content) }))
}
