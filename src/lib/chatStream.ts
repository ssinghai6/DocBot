import type { Message } from "@/components/types"

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
