"use client"

import React, { useState, useMemo } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  Brain, Clock, Maximize2, Download,
  ChevronDown, BookOpen, Wand2, AlertTriangle,
  Copy, Code2, Sparkles,
} from "lucide-react"

import type { Message, ChartMeta, AutopilotStep, Citation } from "./types"
import { useUIStore } from "@/store/uiStore"

// ── Expert personas accent colors ─────────────────────────────────────────
// cyan = data/interactive, amber = AI/finance/analytical, info = other
const PERSONA_ACCENT: Record<string, "cyan" | "amber"> = {
  Generalist: "cyan",
  "Finance Expert": "amber",
  "Data Analyst": "amber",
  Engineer: "cyan",
  "AI/ML Expert": "cyan",
  Consultant: "cyan",
  Doctor: "cyan",
  Lawyer: "cyan",
}

function accentVar(p: string | undefined): string {
  return PERSONA_ACCENT[p ?? ""] === "amber" ? "var(--color-amber-500)" : "var(--color-cyan-500)"
}

// ── Internal helper components ────────────────────────────────────────────────

function DiscrepancyBlock({ content }: { content: string }) {
  return (
    <div className="mt-3 p-3 rounded-[5px] border border-[var(--color-warning-500)]/40 bg-[var(--color-warning-500)]/5">
      <div className="flex items-center gap-1.5 mb-1.5">
        <AlertTriangle className="w-3.5 h-3.5 text-[var(--color-warning-500)] shrink-0" />
        <span className="text-[10px] font-semibold text-[var(--color-warning-500)] uppercase tracking-wider">Discrepancy Detected</span>
      </div>
      <div className="text-[11px] text-[var(--color-text-secondary)] whitespace-pre-line leading-relaxed font-mono">
        {content.trim()}
      </div>
    </div>
  )
}

function ChartThumbnail({
  charts, chartMetas, messageId,
}: { charts: string[]; chartMetas?: (ChartMeta | null)[]; messageId: string }) {
  const selectArtifact = useUIStore((s) => s.selectArtifact)

  if (!charts || charts.length === 0) return null

  return (
    <div className="mt-3 space-y-2">
      {charts.map((b64, i) => {
        const meta = chartMetas?.[i]
        return (
          <button
            key={i}
            type="button"
            onClick={() =>
              selectArtifact({
                messageId: `${messageId}-chart-${i}`,
                type: "chart",
                payload: { b64, meta },
              })
            }
            className="group block w-full text-left border-t-2 border-[var(--color-amber-500)]/60 bg-[var(--color-bg-surface)] border border-[var(--color-border-subtle)] rounded-[5px] overflow-hidden hover:border-[var(--color-amber-500)]/50 transition-colors"
          >
            {/* Header */}
            {meta && (meta.title || meta.x_label || meta.y_label) && (
              <div className="flex items-center justify-between px-3 pt-2 pb-1">
                <span className="text-[11px] font-medium text-[var(--color-text-primary)] truncate">{meta.title || `Chart ${i + 1}`}</span>
                <Maximize2 className="w-3 h-3 text-[var(--color-text-tertiary)] group-hover:text-[var(--color-amber-500)] transition-colors" />
              </div>
            )}
            {/* Image */}
            <div className="px-3 py-2">
              <img
                src={`data:image/png;base64,${b64}`}
                alt={meta?.title || `Chart ${i + 1}`}
                className="rounded-[3px] max-w-full"
              />
            </div>
            {/* Footer */}
            {meta && (meta.x_label || meta.y_label) && (
              <div className="px-3 pb-2 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-[var(--color-text-tertiary)] tabular-nums">
                {meta.x_label && <span>x: {meta.x_label}</span>}
                {meta.y_label && <span>y: {meta.y_label}</span>}
                {meta.series_count > 1 && <span>{meta.series_count} series</span>}
              </div>
            )}
          </button>
        )
      })}
    </div>
  )
}

function SqlCard({ sql, explanation, messageId }: { sql: string; explanation?: string; messageId: string }) {
  const selectArtifact = useUIStore((s) => s.selectArtifact)
  return (
    <button
      type="button"
      onClick={() =>
        selectArtifact({
          messageId: `${messageId}-sql`,
          type: "sql",
          payload: { sql, explanation },
        })
      }
      className="group mt-3 w-full text-left border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] rounded-[5px] overflow-hidden hover:border-[var(--color-cyan-500)]/50 transition-colors"
    >
      <div className="flex items-center justify-between px-3 h-7 bg-[var(--color-bg-surface)] border-b border-[var(--color-border-subtle)]">
        <div className="flex items-center gap-1.5">
          <Code2 className="w-3 h-3 text-[var(--color-cyan-500)]" />
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-cyan-500)]">SQL</span>
        </div>
        <span className="text-[10px] text-[var(--color-text-tertiary)] group-hover:text-[var(--color-cyan-500)] transition-colors">Open in inspector →</span>
      </div>
      <pre className="p-3 text-[11px] font-mono leading-relaxed text-[var(--color-text-secondary)] overflow-x-auto whitespace-pre-wrap max-h-32 line-clamp-5">
        <code>{sql}</code>
      </pre>
    </button>
  )
}

function CollapsibleCode({ code }: { code: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-2 border border-[var(--color-border-subtle)] rounded-[5px] overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-3 h-7 bg-[var(--color-bg-surface)] hover:bg-[var(--color-bg-elevated)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] text-[10px] font-mono uppercase tracking-wider transition-colors"
        onClick={() => setOpen(o => !o)}
      >
        <span className="flex items-center gap-1.5">
          <Code2 className="w-3 h-3" />
          Analysis code
        </span>
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <pre className="p-3 overflow-x-auto bg-[var(--color-bg-inset)] text-[var(--color-cyan-500)] text-[11px] font-mono leading-relaxed">
          <code>{code}</code>
        </pre>
      )}
    </div>
  )
}

function AutopilotStepsList({ steps }: { steps: AutopilotStep[] }) {
  return (
    <div className="mt-3 pt-3 border-t border-[var(--color-border-subtle)] space-y-1.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Wand2 className="w-3 h-3 text-[var(--color-amber-500)]" />
        <span className="text-[10px] font-semibold text-[var(--color-amber-500)] uppercase tracking-wider">
          Investigation Steps
        </span>
        <span className="text-[10px] text-[var(--color-text-quaternary)] tabular-nums">
          ({steps.length})
        </span>
      </div>
      {steps.map((step) => (
        <div key={step.step_num} className="bg-[var(--color-bg-inset)] rounded-[5px] p-2 text-[11px] border border-[var(--color-border-subtle)]">
          <div className="flex items-center gap-1.5 mb-1">
            <span className={`px-1.5 py-0.5 rounded-[3px] text-[9px] font-semibold uppercase tracking-wider ${
              step.tool === "sql_query"  ? "bg-[var(--color-cyan-500)]/15 text-[var(--color-cyan-500)]" :
              step.tool === "doc_search" ? "bg-[var(--color-amber-500)]/15 text-[var(--color-amber-500)]" :
                                          "bg-[var(--color-success-500)]/15 text-[var(--color-success-500)]"
            }`}>
              {step.tool.replace("_", " ")}
            </span>
            <span className="text-[var(--color-text-tertiary)] truncate flex-1">{step.step_label}</span>
          </div>
          {step.content && (
            <p className="text-[var(--color-text-tertiary)] leading-relaxed">{step.content}</p>
          )}
          {step.chart_b64 && (
            <img src={`data:image/png;base64,${step.chart_b64}`} alt="chart" className="mt-2 rounded-[3px] max-h-32 object-contain" />
          )}
          {step.error && (
            <p className="text-[var(--color-danger-500)] mt-1">⚠ {step.error}</p>
          )}
        </div>
      ))}
    </div>
  )
}

function CitationsBlock({ citations, messageId }: { citations: Citation[]; messageId: string }) {
  const selectArtifact = useUIStore((s) => s.selectArtifact)
  return (
    <div className="mt-3 pt-3 border-t border-[var(--color-border-subtle)]">
      <div className="flex items-center gap-1.5 mb-2">
        <BookOpen className="w-3 h-3 text-[var(--color-cyan-500)]" />
        <span className="text-[10px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">References</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((citation, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() =>
              selectArtifact({
                messageId: `${messageId}-cite-${idx}`,
                type: "citations",
                payload: { citations },
              })
            }
            className="text-[10px] px-2 py-1 bg-[var(--color-bg-surface)] rounded-[3px] border border-[var(--color-border-subtle)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-cyan-500)]/40 transition-colors"
            title={`${citation.source} — Page ${citation.page}`}
          >
            <span className="text-[var(--color-cyan-500)] font-medium">{citation.source}</span>
            <span className="text-[var(--color-text-quaternary)] mx-1">·</span>
            <span className="text-[var(--color-text-tertiary)] tabular-nums">p{citation.page}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Agent formatting helpers ──────────────────────────────────────────────────

function renderMessageContent(content: string): React.ReactNode[] | null {
  const parts = content.split(/\[DISCREPANCY\]([\s\S]*?)\[\/DISCREPANCY\]/g)
  if (parts.length === 1) return null
  return parts.map((part, i) =>
    i % 2 === 0
      ? part.trim()
        ? <ReactMarkdown key={i} remarkPlugins={[remarkGfm]}>{part}</ReactMarkdown>
        : null
      : <DiscrepancyBlock key={i} content={part} />
  ).filter(Boolean) as React.ReactNode[]
}

function AgentMessageContent({ msg }: { msg: Message }) {
  // Use globals.css .prose token styling; no more inline prose-* utilities
  return (
    <div className="prose max-w-none">
      {renderMessageContent(msg.content) ?? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {msg.content}
        </ReactMarkdown>
      )}
    </div>
  )
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface ChatMessageProps {
  message: Message
  isLoading: boolean
  isLastMessage: boolean
  onCopy: (text: string) => void
}

// ── Main export ───────────────────────────────────────────────────────────────

export default function ChatMessage({
  message: msg,
  isLoading,
  isLastMessage,
  onCopy,
}: ChatMessageProps) {
  // Stable per-message id for inspector artifact keys
  const messageId = useMemo(
    () => `m-${msg.timestamp?.getTime() ?? Math.random().toString(36).slice(2, 10)}`,
    [msg.timestamp]
  )

  // User message — right-aligned minimal card
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-[5px] bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] px-3 py-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">You</span>
            {msg.timestamp && (
              <span className="text-[10px] text-[var(--color-text-quaternary)] flex items-center gap-1 ml-auto tabular-nums">
                <Clock className="w-2.5 h-2.5" />
                {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            )}
          </div>
          <p className="text-[13px] leading-relaxed whitespace-pre-wrap text-[var(--color-text-primary)]">{msg.content}</p>
        </div>
      </div>
    )
  }

  // Assistant message — left 2px rule, no bubble
  const accent = accentVar(msg.agentPersona)
  const hasAutopilot = msg.autopilotSteps && msg.autopilotSteps.length > 0
  const ruleColor = hasAutopilot ? "var(--color-amber-500)" : "var(--color-border-default)"

  return (
    <div
      className="flex justify-start group"
      style={{ borderLeft: `2px solid ${ruleColor}`, paddingLeft: "12px" }}
    >
      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          <div
            className="w-5 h-5 rounded-[3px] flex items-center justify-center"
            style={{ backgroundColor: `${accent.replace(")", "/10)").replace("var(", "rgba(").replace("--color-", "")}` }}
          >
            <Brain className="w-3 h-3" style={{ color: accent }} />
          </div>

          {msg.agentPersonas && msg.agentPersonas.length > 1 ? (
            <div className="flex items-center gap-1">
              {msg.agentPersonas.map((ap) => {
                const c = accentVar(ap)
                return (
                  <span
                    key={ap}
                    className="text-[9px] px-1.5 py-0.5 rounded-[3px] font-semibold uppercase tracking-wider"
                    style={{ color: c, border: `1px solid ${c}66`, background: "transparent" }}
                  >
                    {ap}
                  </span>
                )
              })}
            </div>
          ) : (
            <span
              className="text-[9px] px-1.5 py-0.5 rounded-[3px] font-semibold uppercase tracking-wider"
              style={{ color: accent, border: `1px solid ${accent}66`, background: "transparent" }}
            >
              {msg.agentPersona ?? "DocBot"}
            </span>
          )}

          {msg.timestamp && (
            <span className="text-[10px] text-[var(--color-text-quaternary)] flex items-center gap-1 tabular-nums">
              <Clock className="w-2.5 h-2.5" />
              {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
        </div>

        {/* Content */}
        {msg.content === "" && isLoading && isLastMessage ? (
          <div className="flex items-center gap-2 py-1">
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-cyan-500)] animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-cyan-500)] animate-bounce" style={{ animationDelay: "180ms" }} />
              <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-cyan-500)] animate-bounce" style={{ animationDelay: "360ms" }} />
            </div>
            <span className="text-[11px] text-[var(--color-text-tertiary)]">Thinking…</span>
          </div>
        ) : (
          <AgentMessageContent msg={msg} />
        )}

        {/* SQL card */}
        {msg.sql && <SqlCard sql={msg.sql} explanation={msg.explanation} messageId={messageId} />}

        {/* Analysis Code */}
        {msg.analysisCode && <CollapsibleCode code={msg.analysisCode} />}

        {/* Chart thumbnail(s) */}
        {msg.charts && msg.charts.length > 0 && (
          <ChartThumbnail charts={msg.charts} chartMetas={msg.chartMetas} messageId={messageId} />
        )}

        {/* Autopilot steps */}
        {hasAutopilot && <AutopilotStepsList steps={msg.autopilotSteps!} />}

        {/* Citations */}
        {msg.citations && msg.citations.length > 0 && (
          <CitationsBlock citations={msg.citations} messageId={messageId} />
        )}

        {/* Actions row */}
        <div className="flex gap-3 mt-2 pt-2 border-t border-[var(--color-border-subtle)] opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onCopy(msg.content)}
            className="flex items-center gap-1 text-[10px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors uppercase tracking-wider"
          >
            <Copy className="w-3 h-3" />
            Copy
          </button>
          {msg.charts && msg.charts.length > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-[var(--color-text-quaternary)] uppercase tracking-wider">
              <Sparkles className="w-3 h-3 text-[var(--color-amber-500)]" />
              AI-generated
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
