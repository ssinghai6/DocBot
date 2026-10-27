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
