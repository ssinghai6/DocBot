"use client"

import React, { useState } from "react"
import {
  Send, Upload, Loader2,
  FileText, MessageSquare,
  Layers,
  Database, BookOpen,
  Download, FileJson, FileText as FileTxt,
  Sparkles, HelpCircle,
  Wand2, CheckCircle2, Hash, XCircle,
  ChevronDown,
} from "lucide-react"
import ChatMessage from "@/components/ChatMessage"
import type { Message, AutopilotStep, Toast } from "@/components/types"

const PERSONA_OPTIONS = ["Generalist", "Finance Expert", "Data Analyst", "Consultant"] as const;
const PERSONA_DISPLAY: Record<string, string> = { Consultant: "Strategy Analyst" };

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-3 py-2">
      <div className="flex gap-1">
        <div className="w-1.5 h-1.5 bg-[var(--color-cyan-500)] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-1.5 h-1.5 bg-[var(--color-cyan-500)] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-1.5 h-1.5 bg-[var(--color-cyan-500)] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
      <span className="text-[11px] text-[var(--color-text-tertiary)] ml-1">Analyzing</span>
    </div>
  )
}

function SessionInfo({ sessionId, fileCount, persona, onClear }: {
  sessionId: string | null;
  fileCount: number;
  persona: string;
  onClear: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copySessionId = () => {
    if (sessionId) {
      navigator.clipboard.writeText(sessionId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!sessionId) return null;

  return (
    <div className="flex items-center gap-2.5 px-2.5 h-7 bg-[var(--color-bg-elevated)] rounded-[5px] border border-[var(--color-border-subtle)]">
      <div className="flex items-center gap-1.5">
        <div className="w-1.5 h-1.5 bg-[var(--color-success-500)] rounded-full animate-pulse" />
        <span className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider font-medium">Active</span>
      </div>
      <div className="w-px h-3 bg-[var(--color-border-default)]" />
      <div className="flex items-center gap-1">
        <FileText className="w-3 h-3 text-[var(--color-cyan-500)]" />
        <span className="text-[11px] text-[var(--color-text-secondary)] tabular-nums">{fileCount}</span>
      </div>
      <div className="w-px h-3 bg-[var(--color-border-default)]" />
      <div className="flex items-center gap-1">
        <Sparkles className="w-3 h-3 text-[var(--color-amber-500)]" />
        <span className="text-[11px] text-[var(--color-text-secondary)]">{persona}</span>
      </div>
      <button
        onClick={copySessionId}
        className="ml-0.5 p-0.5 hover:bg-[var(--color-bg-overlay)] rounded-[3px] transition-colors"
        title="Copy session ID"
      >
        {copied ? <CheckCircle2 className="w-3 h-3 text-[var(--color-success-500)]" /> : <Hash className="w-3 h-3 text-[var(--color-text-tertiary)]" />}
      </button>
      <button
        onClick={onClear}
        className="p-0.5 hover:bg-[var(--color-danger-500)]/10 rounded-[3px] transition-colors"
        title="Clear session"
      >
        <XCircle className="w-3 h-3 text-[var(--color-text-tertiary)] hover:text-[var(--color-danger-500)]" />
      </button>
    </div>
  )
}

export interface ChatAreaProps {
  sessionId: string | null
  uploadedFiles: File[]
  selectedPersona: string
  setSelectedPersona?: React.Dispatch<React.SetStateAction<string>>
  isAutoMode?: boolean
  setAutoMode?: React.Dispatch<React.SetStateAction<boolean>>
  isDbConnected: boolean
  connectionId: string | null
  chatMode: "docs" | "database" | "hybrid"
  setChatMode: React.Dispatch<React.SetStateAction<"docs" | "database" | "hybrid">>
  messages: Message[]
  isLoading: boolean
  input: string
  setInput: React.Dispatch<React.SetStateAction<string>>
  chartType: string
  setChartType: React.Dispatch<React.SetStateAction<string>>
  autopilotMode: boolean
  setAutopilotMode: React.Dispatch<React.SetStateAction<boolean>>
  autopilotRunning: boolean
  autopilotSteps: AutopilotStep[]
  autopilotPlan: string[]
  deepResearch: boolean
  drProgress: { step: string; message: string } | null

  chatContainerRef: React.RefObject<HTMLDivElement | null>
  lastMessageRef: React.RefObject<HTMLDivElement | null>
  messagesEndRef: React.RefObject<HTMLDivElement | null>
  textareaRef: React.RefObject<HTMLTextAreaElement | null>

  handleSendMessage: (e?: React.FormEvent) => void
  clearSession: () => void
  exportChat: (format: 'txt' | 'markdown' | 'json') => void
  showToast: (type: Toast['type'], message: string) => void

  // Onboarding actions
  onUploadClick?: () => void
  onConnectDatabase?: () => void
  onBrowseEdgar?: () => void
  onTryDemo?: () => void
  demoLoading?: boolean
}

export default function ChatArea(props: ChatAreaProps) {
  const {
    sessionId,
    uploadedFiles,
    selectedPersona,
    setSelectedPersona,
    isAutoMode,
    setAutoMode,
    isDbConnected,
    connectionId,
    chatMode,
    setChatMode,
    messages,
    isLoading,
    input,
    setInput,
    chartType,
    setChartType,
    autopilotRunning,
    autopilotSteps,
    autopilotPlan,
    deepResearch,
    drProgress,
    chatContainerRef,
    lastMessageRef,
    messagesEndRef,
    textareaRef,
    handleSendMessage,
    clearSession,
    exportChat,
    showToast,
    onUploadClick,
    onConnectDatabase,
    onBrowseEdgar,
    onTryDemo,
    demoLoading,
  } = props;

  const [personaDropdownOpen, setPersonaDropdownOpen] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    showToast("success", "Copied to clipboard");
  };

  return (
    <main className="h-full flex flex-col min-w-0 bg-[var(--color-bg-base)] relative">
      {/* Top Bar — 44px */}
      <header className="h-11 flex items-center px-4 border-b border-[var(--color-border-subtle)] flex-none gap-2">
        {/* Mode toggle */}
        <div className="flex items-center bg-[var(--color-bg-elevated)] rounded-[5px] border border-[var(--color-border-subtle)] p-0.5">
          {(["docs", "database", "hybrid"] as const).map((mode) => {
            const disabled =
              (mode === "database" && !connectionId) ||
              (mode === "hybrid" && (!connectionId || !sessionId));
            const labels = { docs: "Docs", database: "Database", hybrid: "Hybrid" };
            const icons = {
              docs: <FileText className="w-3 h-3" />,
              database: <Database className="w-3 h-3" />,
              hybrid: <Layers className="w-3 h-3" />,
            };
            return (
              <button
                key={mode}
                onClick={() => !disabled && setChatMode(mode)}
                title={
                  mode === "database" && !connectionId
                    ? "Connect a database first"
                    : mode === "hybrid" && !connectionId
                      ? "Connect a database first"
                      : mode === "hybrid" && !sessionId
                        ? "Upload a PDF first"
                        : undefined
                }
                className={`flex items-center gap-1.5 h-6 px-2 rounded-[3px] text-[11px] font-medium transition-colors ${
                  disabled
                    ? "opacity-40 cursor-not-allowed pointer-events-none text-[var(--color-text-quaternary)]"
                    : chatMode === mode
                      ? "bg-[var(--color-cyan-500)]/15 text-[var(--color-cyan-500)]"
                      : "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
                }`}
              >
                {icons[mode]}
                {labels[mode]}
              </button>
            );
          })}
        </div>

        {/* Session info */}
        <SessionInfo
          sessionId={sessionId}
          fileCount={uploadedFiles.length}
          persona={selectedPersona}
          onClear={clearSession}
        />

        {/* Spacer */}
        <div className="flex-1" />

        {/* Export */}
        {sessionId && messages.length > 0 && (
          <div className="relative group">
            <button className="flex items-center gap-1.5 h-7 px-2.5 rounded-[5px] text-[11px] text-[var(--color-text-secondary)] bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] hover:border-[var(--color-cyan-500)]/40 hover:text-[var(--color-text-primary)] transition-colors">
              <Download className="w-3 h-3" />
              Export
            </button>
            <div className="absolute top-full right-0 mt-1 py-1 bg-[var(--color-bg-elevated)] rounded-[5px] border border-[var(--color-border-default)] shadow-[var(--elev-3)] opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 min-w-[140px]">
              <button
                onClick={() => exportChat('txt')}
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[11px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-overlay)] hover:text-[var(--color-text-primary)] transition-colors"
              >
                <FileTxt className="w-3 h-3" />
                Text (.txt)
              </button>
              <button
                onClick={() => exportChat('markdown')}
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[11px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-overlay)] hover:text-[var(--color-text-primary)] transition-colors"
              >
                <FileText className="w-3 h-3" />
                Markdown (.md)
              </button>
              <button
                onClick={() => exportChat('json')}
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-[11px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-overlay)] hover:text-[var(--color-text-primary)] transition-colors"
              >
                <FileJson className="w-3 h-3" />
                JSON (.json)
              </button>
            </div>
          </div>
        )}
      </header>

      {/* Chat Output */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-5 lg:px-6 pb-4 max-w-5xl mx-auto w-full">
        {!sessionId && !isDbConnected && messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <p className="text-[22px] font-semibold text-[var(--color-text-primary)] mb-2 text-center">What do you want to analyze?</p>
            <p className="text-[13px] text-[var(--color-text-tertiary)] text-center max-w-md mb-10">
              Upload financial documents, connect a live database, or browse SEC filings.
            </p>

            {/* Action cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-xl w-full mb-8">
              <button
                onClick={onUploadClick}
                className="group flex flex-col items-center gap-3 p-5 bg-[var(--color-bg-surface)] rounded-[8px] border border-[var(--color-border-subtle)] hover:border-[var(--color-cyan-500)]/40 transition-colors"
              >
                <div className="w-10 h-10 rounded-[5px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/20 flex items-center justify-center">
                  <Upload className="w-4 h-4 text-[var(--color-cyan-500)]" />
                </div>
                <div className="text-center">
                  <p className="text-[13px] font-medium text-[var(--color-text-primary)]">Upload Document</p>
                  <p className="text-[11px] text-[var(--color-text-tertiary)] mt-0.5">PDF, CSV, or SQLite</p>
                </div>
              </button>

              <button
                onClick={onConnectDatabase}
                className="group flex flex-col items-center gap-3 p-5 bg-[var(--color-bg-surface)] rounded-[8px] border border-[var(--color-border-subtle)] hover:border-[var(--color-cyan-500)]/40 transition-colors"
              >
                <div className="w-10 h-10 rounded-[5px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/20 flex items-center justify-center">
                  <Database className="w-4 h-4 text-[var(--color-cyan-500)]" />
                </div>
                <div className="text-center">
                  <p className="text-[13px] font-medium text-[var(--color-text-primary)]">Connect Database</p>
                  <p className="text-[11px] text-[var(--color-text-tertiary)] mt-0.5">PostgreSQL, MySQL, Azure</p>
                </div>
              </button>

              <button
                onClick={onBrowseEdgar}
                className="group flex flex-col items-center gap-3 p-5 bg-[var(--color-bg-surface)] rounded-[8px] border border-[var(--color-border-subtle)] hover:border-[var(--color-cyan-500)]/40 transition-colors"
              >
                <div className="w-10 h-10 rounded-[5px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/20 flex items-center justify-center">
                  <BookOpen className="w-4 h-4 text-[var(--color-cyan-500)]" />
                </div>
                <div className="text-center">
                  <p className="text-[13px] font-medium text-[var(--color-text-primary)]">Browse SEC Filings</p>
                  <p className="text-[11px] text-[var(--color-text-tertiary)] mt-0.5">10-K, 10-Q, Annual Reports</p>
                </div>
              </button>
            </div>

            {/* Divider + Try Demo */}
            <div className="flex items-center gap-3 w-full max-w-xs mb-5">
              <div className="flex-1 h-px bg-[var(--color-border-subtle)]" />
              <span className="text-[10px] text-[var(--color-text-quaternary)] uppercase tracking-wider">or try instantly</span>
              <div className="flex-1 h-px bg-[var(--color-border-subtle)]" />
            </div>

            <button
              onClick={onTryDemo}
              disabled={demoLoading}
              className="flex items-center gap-2 h-9 px-4 rounded-[5px] bg-[var(--color-amber-500)]/15 border border-[var(--color-amber-500)]/40 text-[var(--color-amber-500)] text-[12px] font-medium hover:bg-[var(--color-amber-500)]/20 transition-colors disabled:opacity-60 disabled:cursor-not-allowed mb-8"
            >
              {demoLoading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Loading demo...
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  Try Demo — TechCorp 10-K Analysis
                </>
              )}
            </button>

            {/* Capability pills */}
            <div className="flex flex-wrap items-center justify-center gap-1.5 text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider">
              <span className="px-2 py-1 rounded-[3px] bg-[var(--color-bg-surface)] border border-[var(--color-border-subtle)]">Hybrid Doc + DB</span>
              <span className="px-2 py-1 rounded-[3px] bg-[var(--color-bg-surface)] border border-[var(--color-border-subtle)]">Discrepancy Detection</span>
              <span className="px-2 py-1 rounded-[3px] bg-[var(--color-bg-surface)] border border-[var(--color-border-subtle)]">Analytical Autopilot</span>
            </div>
          </div>
        ) : messages.length === 0 && (sessionId || isDbConnected) ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="w-12 h-12 rounded-[8px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/30 flex items-center justify-center mb-4">
              <MessageSquare className="w-5 h-5 text-[var(--color-cyan-500)]" />
            </div>
            <p className="text-[16px] font-semibold text-[var(--color-text-primary)] mb-1">
              {isDbConnected ? "Database connected" : "Ready to analyze"}
            </p>
            <p className="text-[12px] text-[var(--color-text-tertiary)]">
              {isDbConnected
                ? `Query financials, metrics, and trends in ${chatMode} mode`
                : "Ask about revenue, margins, risk factors, or any financial metric"}
            </p>

            {/* Suggested questions */}
            <div className="mt-6 grid gap-1.5 max-w-lg w-full">
              <p className="text-[10px] text-[var(--color-text-quaternary)] uppercase tracking-wider text-center mb-1">Try asking</p>
              {(isDbConnected
                ? [
                  "What are the total revenue figures by quarter?",
                  "Show me a breakdown of expenses by category",
                  "Which accounts have the highest growth rate?",
                  "Are there any anomalies or outliers in the data?",
                ]
                : [
                  "Summarize the key financial highlights",
                  "What are the revenue and net income figures?",
                  "Compare year-over-year performance metrics",
                  "What are the main risk factors mentioned?",
                ]
              ).map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(question)}
                  className="text-left px-3 py-2 bg-[var(--color-bg-surface)] border border-[var(--color-border-subtle)] rounded-[5px] text-[12px] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-cyan-500)]/30 transition-colors"
                >
                  <Sparkles className="w-3 h-3 inline mr-2 text-[var(--color-cyan-500)]" />
                  {question}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4 max-w-5xl mx-auto pb-20 pt-4">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                ref={idx === messages.length - 1 ? lastMessageRef : null}
              >
                <ChatMessage
                  message={msg}
                  isLoading={isLoading}
                  isLastMessage={idx === messages.length - 1}
                  onCopy={copyToClipboard}
                />
              </div>
            ))}
            {/* Autopilot live step-by-step progress panel */}
            {autopilotRunning && (autopilotPlan.length > 0 || autopilotSteps.length > 0) && (
              <div className="flex justify-start">
                <div className="max-w-[85%] lg:max-w-[80%] rounded-[8px] p-3 bg-[var(--color-bg-surface)] border border-[var(--color-amber-500)]/30 border-l-2 border-l-[var(--color-amber-500)] text-[var(--color-text-secondary)]">
                  <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[var(--color-border-subtle)]">
                    <Wand2 className="w-3.5 h-3.5 text-[var(--color-amber-500)]" />
                    <span className="text-[10px] font-semibold text-[var(--color-amber-500)] uppercase tracking-wider">Autopilot Investigation</span>
                    <span className="ml-auto text-[10px] text-[var(--color-text-tertiary)] tabular-nums">{autopilotSteps.length}/{autopilotPlan.length} steps</span>
                  </div>
                  {/* Plan overview */}
                  {autopilotPlan.length > 0 && (
                    <div className="mb-2 space-y-1">
                      {autopilotPlan.map((planStep, pi) => {
                        const completed = autopilotSteps.find(s => s.step_num === pi + 1);
                        const isCurrent = !completed && autopilotSteps.length === pi;
                        return (
                          <div key={pi} className={`flex items-start gap-2 text-[11px] py-0.5 ${completed ? "text-[var(--color-text-tertiary)]" : isCurrent ? "text-[var(--color-text-primary)]" : "text-[var(--color-text-quaternary)]"}`}>
                            <span className={`shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5 ${
                              completed ? "bg-[var(--color-success-500)]/15 text-[var(--color-success-500)]" : isCurrent ? "bg-[var(--color-amber-500)]/20 text-[var(--color-amber-500)]" : "bg-[var(--color-bg-overlay)] text-[var(--color-text-quaternary)]"
                            }`}>
                              {completed ? "✓" : pi + 1}
                            </span>
                            <span className="leading-tight">{planStep}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {/* Completed steps with results */}
                  {autopilotSteps.length > 0 && (
                    <div className="space-y-1.5 border-t border-[var(--color-border-subtle)] pt-2">
                      {autopilotSteps.map((step) => (
                        <div key={step.step_num} className="bg-[var(--color-bg-inset)] rounded-[5px] p-2 text-[11px]">
                          <div className="flex items-center gap-1.5 mb-1">
                            <span className={`px-1.5 py-0.5 rounded-[3px] text-[9px] font-semibold uppercase tracking-wider ${
                              step.tool === "sql_query" ? "bg-[var(--color-cyan-500)]/15 text-[var(--color-cyan-500)]" :
                              step.tool === "doc_search" ? "bg-[var(--color-amber-500)]/15 text-[var(--color-amber-500)]" :
                              "bg-[var(--color-success-500)]/15 text-[var(--color-success-500)]"
                            }`}>{step.tool.replace("_", " ")}</span>
                            <span className="text-[var(--color-text-tertiary)] truncate">{step.step_label}</span>
                          </div>
                          {step.content && (
                            <p className="text-[var(--color-text-tertiary)] line-clamp-2 leading-relaxed">{step.content}</p>
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
                  )}
                  {/* Spinner while running */}
                  <div className="flex items-center gap-2 mt-2 pt-2 border-t border-[var(--color-border-subtle)]">
                    <div className="w-2 h-2 rounded-full bg-[var(--color-amber-500)]/60 animate-pulse" />
                    <span className="text-[11px] text-[var(--color-text-tertiary)]">Investigating…</span>
                  </div>
                </div>
              </div>
            )}

            {/* Deep Research progress strip */}
            {isLoading && drProgress && deepResearch && (
              <div className="flex justify-start mb-2">
                <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-[5px] bg-[var(--color-bg-surface)] border border-[var(--color-cyan-500)]/30 text-[var(--color-cyan-500)] text-[11px] max-w-sm">
                  <span className="animate-pulse shrink-0">
                    {drProgress.step === "planning" && "🧠"}
                    {drProgress.step === "retrieving" && "🔍"}
                    {drProgress.step === "evaluating" && "✅"}
                    {drProgress.step === "gap_fill" && "🔄"}
                    {drProgress.step === "synthesizing" && "📝"}
                  </span>
                  <span className="truncate">{drProgress.message}</span>
                </div>
              </div>
            )}
            {isLoading && messages.length > 0 && messages[messages.length - 1].role === "user" && !autopilotRunning && !drProgress && (
              <div className="flex justify-start">
                <TypingIndicator />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 lg:px-6 flex-none w-full max-w-5xl mx-auto">
        {/* Chart type selector — shown only when DB connected */}
        {isDbConnected && (
          <div className="flex items-center gap-1.5 mb-2 flex-wrap">
            <span className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider shrink-0">Chart</span>
            {(["auto", "bar", "line", "scatter", "heatmap", "box", "multi"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setChartType(t)}
                className={`px-2 h-5 rounded-[3px] text-[10px] border transition-colors capitalize ${
                  chartType === t
                    ? "border-[var(--color-cyan-500)]/50 bg-[var(--color-cyan-500)]/10 text-[var(--color-cyan-500)]"
                    : "border-[var(--color-border-subtle)] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] hover:border-[var(--color-border-default)]"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        )}

        <form
          onSubmit={handleSendMessage}
          className="relative bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-[8px] shadow-[var(--elev-1)] focus-within:border-[var(--color-cyan-500)]/60 focus-within:shadow-[0_0_0_3px_var(--glow-cyan)] transition-all overflow-hidden"
        >
          {/* Header row: persona selector */}
          {setSelectedPersona && (
            <div className="flex items-center gap-2 px-3 pt-2 pb-1 border-b border-[var(--color-border-subtle)]">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setPersonaDropdownOpen(!personaDropdownOpen)}
                  className="flex items-center gap-1.5 h-6 px-2 rounded-[3px] text-[10px] uppercase tracking-wider font-semibold bg-[var(--color-bg-overlay)] hover:bg-[var(--color-bg-surface)] text-[var(--color-text-secondary)] transition-colors"
                >
                  <Sparkles className="w-3 h-3 text-[var(--color-amber-500)]" />
                  {isAutoMode ? "Auto" : (PERSONA_DISPLAY[selectedPersona] || selectedPersona)}
                  <ChevronDown className={`w-3 h-3 text-[var(--color-text-tertiary)] transition-transform ${personaDropdownOpen ? "rotate-180" : ""}`} />
                </button>
                {personaDropdownOpen && (
                  <div className="absolute bottom-full left-0 mb-1 py-1 bg-[var(--color-bg-elevated)] rounded-[5px] border border-[var(--color-border-default)] shadow-[var(--elev-3)] z-50 min-w-[160px]">
                    <button
                      type="button"
                      onClick={() => { setAutoMode?.(true); setPersonaDropdownOpen(false); }}
                      className={`w-full flex items-center gap-2 px-2.5 py-1.5 text-[11px] transition-colors ${
                        isAutoMode ? "text-[var(--color-amber-500)] bg-[var(--color-amber-500)]/10" : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-overlay)]"
                      }`}
                    >
                      <Wand2 className="w-3 h-3" />
                      Auto-routing
                    </button>
                    {PERSONA_OPTIONS.map((name) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => {
                          setAutoMode?.(false);
                          setSelectedPersona(name);
                          setPersonaDropdownOpen(false);
                        }}
                        className={`w-full flex items-center gap-2 px-2.5 py-1.5 text-[11px] transition-colors ${
                          !isAutoMode && selectedPersona === name
                            ? "text-[var(--color-cyan-500)] bg-[var(--color-cyan-500)]/10"
                            : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-overlay)]"
                        }`}
                      >
                        {PERSONA_DISPLAY[name] || name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="flex items-end">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder={
                  isDbConnected
                    ? "Ask about revenue, margins, trends..."
                    : sessionId
                      ? "Ask about financials, risk factors, key metrics..."
                      : "Upload a filing or connect a database to start..."
                }
                disabled={(!sessionId && !isDbConnected) || isLoading}
                className="w-full max-h-40 min-h-[56px] p-3 pr-12 bg-transparent outline-none resize-none text-[var(--color-text-primary)] placeholder-[var(--color-text-quaternary)] text-[13px] leading-relaxed disabled:opacity-50"
                rows={1}
              />
              <div className="absolute bottom-2.5 right-3 text-[10px] text-[var(--color-text-quaternary)] tabular-nums">
                {input.length > 0 && <span>{input.length}</span>}
              </div>
            </div>
            <button
              type="submit"
              disabled={!input.trim() || (!sessionId && !isDbConnected) || isLoading}
              className="m-2 h-8 w-8 flex items-center justify-center rounded-[5px] bg-[var(--color-cyan-500)] text-[var(--color-bg-base)] disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--color-cyan-600)] transition-colors"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </form>
        <div className="text-center mt-2 text-[10px] text-[var(--color-text-quaternary)] flex items-center justify-center gap-1.5">
          <HelpCircle className="w-3 h-3" />
          AI-generated analysis. Always verify against source documents.
        </div>
      </div>

    </main>
  );
}
