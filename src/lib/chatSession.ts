export type ChatMode = "docs" | "database" | "hybrid"

/**
 * Pick the chat mode the available sources allow.
 * - doc session + DB/CSV connection -> "hybrid"
 * - "database" without a connection -> "docs"
 * - "hybrid" that lost a source -> whichever source remains ("database" or "docs")
 * Idempotent: resolveChatMode(resolveChatMode(m, c, s), c, s) === resolveChatMode(m, c, s).
 */
export function resolveChatMode(
  mode: ChatMode,
  connectionId: string | null,
  sessionId: string | null,
): ChatMode {
  if (connectionId && sessionId) return "hybrid"
  if (mode === "database" && !connectionId) return "docs"
  if (mode === "hybrid") return connectionId ? "database" : "docs"
  return mode
}

/** Random ID for chats that have no PDF session yet. Call once (e.g. a useState initializer). */
export function createAnonymousSessionId(): string {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Math.random().toString(36).substring(2)
}
