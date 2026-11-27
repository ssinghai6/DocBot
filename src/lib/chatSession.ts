export type ChatMode = "docs" | "database" | "hybrid"

/**
 * Fall back when the chosen mode's source is gone. Never promotes: a manual
 * Docs/Database pick sticks; hybrid promotion is a one-time effect in chat/page.tsx.
 * - "database" without a connection -> "docs"
 * - "hybrid" that lost a source -> whichever source remains ("database" or "docs")
 * Idempotent: resolveChatMode(resolveChatMode(m, c, s), c, s) === resolveChatMode(m, c, s).
 */
export function resolveChatMode(
  mode: ChatMode,
  connectionId: string | null,
  sessionId: string | null,
): ChatMode {
  if (mode === "database" && !connectionId) return "docs"
  if (mode === "hybrid" && !(connectionId && sessionId)) return connectionId ? "database" : "docs"
  return mode
}

/** Random ID for chats that have no PDF session yet. Call once (e.g. a useState initializer). */
export function createAnonymousSessionId(): string {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Math.random().toString(36).substring(2)
}
