"use client"

import React, { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  Brain, Clock, Maximize2, Download, Minimize2,
  Terminal, ChevronDown, BookOpen, Wand2, AlertTriangle,
  Copy,
} from "lucide-react"

import type { Message, ChartMeta, AutopilotStep, Citation } from "./types"

// ── Expert personas accent colors (duplicated here to avoid circular import) ─
// This map drives per-persona badge styling in message headers.
const PERSONA_ACCENT_COLORS: Record<string, string> = {
  Generalist: "#667eea",
  Doctor: "#10b981",
  "Finance Expert": "#f59e0b",
  Engineer: "#3b82f6",
  "AI/ML Expert": "#8b5cf6",
  Lawyer: "#ef4444",
  Consultant: "#06b6d4",
  "Data Analyst": "#f97316",
}

const PERSONA_RESPONSE_FORMATS: Record<string, string> = {
  "Finance Expert": "finance",
  Doctor: "clinical",
  Lawyer: "legal",
  "Data Analyst": "data",
  Engineer: "technical",
  "AI/ML Expert": "research",
  Consultant: "consulting",
  Generalist: "general",
}

const PERSONA_HIGHLIGHT_PATTERNS: Record<string, string | null> = {
  Doctor: "\\b(WARNING|CRITICAL|CONTRAINDICATED|ABNORMAL|RED FLAG)\\b",
  Lawyer: "\\b(RISK|WARNING|VOID|BREACH|PENALTY|PROHIBITED|LIMITATION OF LIABILITY)\\b",
  "Data Analyst": "\\b(NULL|ERROR|WARNING|OUTLIER|MISSING)\\b",
}

// ── Internal helper components ────────────────────────────────────────────────

function DiscrepancyBlock({ content }: { content: string }) {
  return (
    <div className="mt-3 p-3 rounded-xl border border-[#f59e0b]/40 bg-[#f59e0b]/8">
      <div className="flex items-center gap-2 mb-2">
        <AlertTriangle className="w-4 h-4 text-[#f59e0b] shrink-0" />
        <span className="text-xs font-semibold text-[#f59e0b] uppercase tracking-wider">Discrepancy Detected</span>
      </div>
      <div className="text-sm text-gray-300 whitespace-pre-line leading-relaxed font-mono text-xs">
        {content.trim()}
      </div>
    </div>
  )
}

function ChartDisplay({ charts, chartMetas }: { charts: string[]; chartMetas?: (ChartMeta | null)[] }) {
  const [expanded, setExpanded] = useState<number | null>(null)

  if (!charts || charts.length === 0) return null

  return (
    <div className="mt-3 space-y-3">
      {charts.map((b64, i) => {
        const meta = chartMetas?.[i]
        return (
          <div key={i} className="space-y-1">
            <div className="relative group">
              <img
                src={`data:image/png;base64,${b64}`}
                alt={meta?.title || `Analysis chart ${i + 1}`}
                className="rounded-lg border border-[#ffffff10] max-w-full cursor-pointer hover:opacity-90 transition-opacity"
                onClick={() => setExpanded(i)}
              />
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                <button
                  onClick={() => setExpanded(i)}
                  className="bg-[#12121a]/90 backdrop-blur-sm rounded-md p-1.5 text-gray-400 hover:text-gray-200 border border-[#ffffff10]"
                  title="Expand"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
                <a
                  href={`data:image/png;base64,${b64}`}
                  download={meta?.title ? `${meta.title.replace(/\s+/g, "_")}.png` : `chart-${i + 1}.png`}
                  className="bg-[#12121a]/90 backdrop-blur-sm rounded-md px-2 py-1.5 text-xs text-gray-400 hover:text-gray-200 border border-[#ffffff10] flex items-center gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Download className="w-3 h-3" />
                  PNG
                </a>
              </div>
            </div>
            {meta && (meta.title || meta.x_label || meta.y_label) && (
              <div className="px-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-gray-500">
                {meta.title && <span className="font-medium text-gray-400">{meta.title}</span>}
                {meta.x_label && <span>x: {meta.x_label}</span>}
                {meta.y_label && <span>y: {meta.y_label}</span>}
                {meta.series_count > 1 && <span>{meta.series_count} series</span>}
              </div>
            )}
          </div>
        )
      })}

      {expanded !== null && (
        <div
          className="fixed inset-0 bg-black/85 z-50 flex flex-col items-center justify-center p-4 gap-3"
          onClick={() => setExpanded(null)}
        >
          <img
            src={`data:image/png;base64,${charts[expanded]}`}
            alt="Chart full view"
            className="max-w-full max-h-[85vh] rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
          <div className="flex items-center gap-3">
            <a
              href={`data:image/png;base64,${charts[expanded]}`}
              download={chartMetas?.[expanded]?.title
                ? `${chartMetas[expanded]!.title.replace(/\s+/g, "_")}.png`
                : `chart-${expanded + 1}.png`}
              className="bg-[#1a1a24] border border-[#ffffff15] rounded-lg px-3 py-1.5 text-xs text-gray-300 hover:text-white flex items-center gap-1.5 transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              <Download className="w-3 h-3" /> Download PNG
            </a>
            <button
              onClick={() => setExpanded(null)}
              className="bg-[#1a1a24] border border-[#ffffff15] rounded-lg px-3 py-1.5 text-xs text-gray-300 hover:text-white flex items-center gap-1.5 transition-colors"
            >
              <Minimize2 className="w-3 h-3" /> Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function CollapsibleCode({ code }: { code: string }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-2 border border-[#ffffff10] rounded-lg overflow-hidden text-sm">
      <button
        className="w-full flex items-center justify-between px-3 py-2 bg-[#1a1a24]/80 hover:bg-[#1a1a24] text-gray-500 hover:text-gray-300 text-xs font-mono transition-colors"
        onClick={() => setOpen(o => !o)}
      >
        <span className="flex items-center gap-1.5">
          <Terminal className="w-3 h-3" />
          Analysis code
        </span>
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <pre className="p-3 overflow-x-auto bg-[#0a0a0f] text-[#10b981] text-xs leading-relaxed">
          <code>{code}</code>
        </pre>
      )}
    </div>
  )
}

function AutopilotStepsList({ steps }: { steps: AutopilotStep[] }) {
  return (
    <div className="mt-3 pt-3 border-t border-[#ffffff08] space-y-2">
      <div className="flex items-center gap-1.5 mb-2">
        <Wand2 className="w-3 h-3 text-[#a5b4fc]" />
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Investigation Steps
        </span>
        <span className="text-[10px] text-gray-600 ml-1">
          ({steps.length})
        </span>
      </div>
      {steps.map((step) => (
        <div key={step.step_num} className="bg-[#ffffff04] rounded-xl p-2.5 text-xs border border-[#ffffff08]">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${
              step.tool === "sql_query"  ? "bg-[#0ea5e9]/20 text-[#38bdf8]" :
              step.tool === "doc_search" ? "bg-[#f59e0b]/20 text-[#fbbf24]" :
                                          "bg-[#10b981]/20 text-[#34d399]"
            }`}>
              {step.tool.replace("_", " ")}
            </span>
            <span className="text-gray-500 truncate flex-1">{step.step_label}</span>
          </div>
          {step.content && (
            <p className="text-gray-400 leading-relaxed">{step.content}</p>
          )}
          {step.chart_b64 && (
            <div className="mt-2">
              <ChartDisplay charts={[step.chart_b64]} chartMetas={undefined} />
            </div>
          )}
          {step.error && (
            <p className="text-red-400 mt-1 text-xs">&#9888; {step.error}</p>
          )}
        </div>
      ))}
    </div>
  )
}

function CitationsBlock({ citations }: { citations: Citation[] }) {
  return (
    <div className="mt-4 pt-3 border-t border-[#ffffff08]">
      <div className="flex items-center gap-2 mb-3">
        <BookOpen className="w-3 h-3 text-[#667eea]" />
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">References</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation, idx) => (
          <div
            key={idx}
            className="text-xs px-3 py-2 bg-[#1a1a24]/80 rounded-lg border border-[#ffffff10] text-gray-400 hover:text-gray-300 hover:border-[#667eea]/30 hover:bg-[#1a1a24] transition-all cursor-pointer group"
            title={`${citation.source} - Page ${citation.page}`}
          >
            <span className="text-[#667eea] font-medium">{citation.source}</span>
            <span className="text-gray-600 mx-1.5">•</span>
            <span className="text-gray-500">Page {citation.page}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Agent formatting helpers ──────────────────────────────────────────────────

function applyAgentFormatting(content: string, agentPersona: string | undefined): string {
  if (!agentPersona) return content
  const highlightPattern = PERSONA_HIGHLIGHT_PATTERNS[agentPersona]
  const responseFormat = PERSONA_RESPONSE_FORMATS[agentPersona]

  let processed = content

  if (highlightPattern) {
    try {
      const re = new RegExp(highlightPattern, "g")
      processed = processed.replace(re, "**$&**")
    } catch {
      // invalid regex — skip
    }
  }

  if (responseFormat === "clinical") {
    processed = processed.replace(
      /## Medical Disclaimer\n/g,
      "\n---\n> **Medical Disclaimer**\n>\n> "
    )
  }

  return processed
}

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
  const responseFormat = PERSONA_RESPONSE_FORMATS[msg.agentPersona ?? ""] ?? "general"
  const processedContent = applyAgentFormatting(msg.content, msg.agentPersona)

  const wrapperClass = [
    "prose prose-invert prose-sm max-w-none",
    "prose-p:leading-7 prose-p:text-gray-300",
    "prose-headings:text-white prose-headings:font-semibold prose-headings:mt-0 prose-headings:mb-2",
    "prose-strong:text-white prose-strong:font-medium",
    "prose-ul:text-gray-300 prose-ul:my-2 prose-li:my-1",
    "prose-ol:text-gray-300 prose-ol:my-2 prose-li:my-1",
    "prose-li:marker:text-[#667eea]",
    "prose-a:text-[#667eea] prose-a:no-underline hover:prose-a:underline",
    "prose-blockquote:border-l-[#667eea] prose-blockquote:bg-[#1a1a24]/50 prose-blockquote:py-1 prose-blockquote:px-3 prose-blockquote:rounded-r-lg prose-blockquote:text-gray-400 prose-blockquote:not-italic",
    "prose-code:text-[#764ba2] prose-code:bg-[#1a1a24]/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none",
    "prose-pre:bg-[#1a1a24]/80 prose-pre:border prose-pre:border-[#ffffff08]",
    responseFormat === "finance" ? "agent-finance" : "",
    responseFormat === "legal" ? "agent-legal" : "",
    responseFormat === "clinical" ? "agent-clinical" : "",
  ].filter(Boolean).join(" ")

  const financeStyle: React.CSSProperties | undefined =
    responseFormat === "finance"
      ? { borderLeft: "3px solid rgba(245,158,11,0.3)", paddingLeft: "0.75rem" }
      : undefined

  return (
    <div className={wrapperClass} style={financeStyle}>
      {renderMessageContent(processedContent) ?? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {processedContent}
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
  return (
    <div
      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-in slide-in-from-bottom-2 duration-300`}
    >
      <div
        className={`max-w-[85%] lg:max-w-[80%] rounded-2xl p-4 shadow-lg border backdrop-blur-xl group
          ${msg.role === "user"
            ? "bg-gradient-to-br from-[#667eea] to-[#764ba2] text-white border-transparent rounded-br-sm"
            : "bg-[#12121a]/80 border-[#ffffff08] text-[#e0e0e0] rounded-bl-sm hover:border-[#667eea]/20 transition-colors"
          }`}
      >
        {/* Message Header — assistant */}
        {msg.role === "assistant" && (
          <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[#ffffff08]">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-[#667eea] to-[#764ba2] flex items-center justify-center">
              <Brain className="w-3 h-3 text-white" />
            </div>

            {/* DOCBOT-804: Dynamic agent badge */}
            {msg.agentPersonas && msg.agentPersonas.length > 1 ? (
              <div className="flex items-center gap-1">
                {msg.agentPersonas.map((ap) => {
                  const color = PERSONA_ACCENT_COLORS[ap] ?? "#667eea"
                  return (
                    <span
                      key={ap}
                      className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                      style={{ backgroundColor: color + "20", color, border: `1px solid ${color}40` }}
                    >
                      {ap}
                    </span>
                  )
                })}
              </div>
            ) : (
              (() => {
                const agentName = msg.agentPersona ?? "DocBot"
                const accentColor = PERSONA_ACCENT_COLORS[agentName] ?? "#667eea"
                return (
                  <span
                    className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                    style={{ backgroundColor: accentColor + "20", color: accentColor, border: `1px solid ${accentColor}40` }}
                  >
                    {agentName}
                  </span>
                )
              })()
            )}

            {msg.timestamp && (
              <>
                <span className="text-[10px] text-gray-600">•</span>
                <span className="text-[10px] text-gray-600 flex items-center gap-1">
                  <Clock className="w-2 h-2" />
                  {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </>
            )}
          </div>
        )}

        {/* Message Header — user */}
        {msg.role === "user" && (
          <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[#ffffff10]">
            <span className="text-xs font-medium text-white/80">You</span>
            {msg.timestamp && (
              <span className="text-[10px] text-white/40 flex items-center gap-1 ml-auto">
                <Clock className="w-2 h-2" />
                {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            )}
          </div>
        )}

        {/* Message Content */}
        {msg.role === "assistant" ? (
          msg.content === "" && isLoading && isLastMessage ? (
            /* Thinking animation */
            <div className="flex flex-col gap-2.5 py-1">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-[#667eea] animate-bounce" style={{ animationDelay: "0ms", animationDuration: "1s" }} />
                  <div className="w-2 h-2 rounded-full bg-[#764ba2] animate-bounce" style={{ animationDelay: "180ms", animationDuration: "1s" }} />
                  <div className="w-2 h-2 rounded-full bg-[#8b5cf6] animate-bounce" style={{ animationDelay: "360ms", animationDuration: "1s" }} />
                </div>
                <span className="text-xs text-gray-500 animate-pulse">Thinking…</span>
              </div>
              <div className="space-y-2 opacity-40">
                <div className="h-2.5 rounded-full bg-gradient-to-r from-[#667eea]/30 to-transparent animate-pulse" style={{ width: "72%", animationDelay: "0ms" }} />
                <div className="h-2.5 rounded-full bg-gradient-to-r from-[#667eea]/20 to-transparent animate-pulse" style={{ width: "55%", animationDelay: "200ms" }} />
                <div className="h-2.5 rounded-full bg-gradient-to-r from-[#667eea]/10 to-transparent animate-pulse" style={{ width: "40%", animationDelay: "400ms" }} />
              </div>
            </div>
          ) : (
            <AgentMessageContent msg={msg} />
          )
        ) : (
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
        )}

        {/* Analysis Code (DOCBOT-303) */}
        {msg.role === "assistant" && msg.analysisCode && (
          <CollapsibleCode code={msg.analysisCode} />
        )}

        {/* Charts (DOCBOT-303) */}
        {msg.role === "assistant" && msg.charts && msg.charts.length > 0 && (
          <ChartDisplay charts={msg.charts} chartMetas={msg.chartMetas} />
        )}

        {/* DOCBOT-405: Autopilot investigation steps */}
        {msg.role === "assistant" && msg.autopilotSteps && msg.autopilotSteps.length > 0 && (
          <AutopilotStepsList steps={msg.autopilotSteps} />
        )}

        {/* Citations */}
        {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
          <CitationsBlock citations={msg.citations} />
        )}

        {/* Message Actions */}
        {msg.role === "assistant" && (
          <div className="flex gap-2 mt-2 pt-2 border-t border-[#ffffff08] opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => onCopy(msg.content)}
              className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
            >
              <Copy className="w-3 h-3" />
              Copy
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
