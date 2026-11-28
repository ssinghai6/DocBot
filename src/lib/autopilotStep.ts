// DOCBOT-2727: backend contract (DOCBOT-2724/2725) for a step that found no
// usable data: failure_reason === "insufficient_data" and the result text
// starts with "INSUFFICIENT_DATA:".

import type { AutopilotStep } from "@/components/types"

const MARKER = "INSUFFICIENT_DATA:"

export function isInsufficientStep(step: Pick<AutopilotStep, "failure_reason" | "content" | "summary">): boolean {
  return step.failure_reason === "insufficient_data" || [step.content, step.summary].some(t => t?.trimStart().startsWith(MARKER))
}

/** Human reason line, marker stripped. */
export function insufficientReason(step: Pick<AutopilotStep, "content" | "summary">): string {
  const text = [step.content, step.summary].find(t => t?.trimStart().startsWith(MARKER)) ?? step.summary ?? step.content ?? ""
  return text.trimStart().replace(MARKER, "").trim() || "Not enough data in your sources for this step."
}
