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
    <div className="flex items-center gap-1 px-4 py-3 bg-[#12121a]/80 border border-[#ffffff08] rounded-2xl rounded-bl-sm">
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-[#667eea] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 bg-[#764ba2] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 bg-[#667eea] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
      <span className="text-sm text-gray-400 ml-2">Analyzing document</span>
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
    <div className="flex items-center gap-3 px-3 py-1.5 bg-[#1a1a24]/60 rounded-lg border border-[#ffffff06]">
      <div className="flex items-center gap-1.5">
        <div className="w-2 h-2 bg-[#10b981] rounded-full animate-pulse" />
        <span className="text-xs text-gray-400">Active Session</span>
      </div>
      <div className="w-px h-4 bg-[#ffffff10]" />
      <div className="flex items-center gap-1.5">
        <FileText className="w-3 h-3 text-[#667eea]" />
        <span className="text-xs text-gray-300">{fileCount} file{fileCount !== 1 ? 's' : ''}</span>
      </div>
      <div className="w-px h-4 bg-[#ffffff10]" />
      <div className="flex items-center gap-1.5">
        <Sparkles className="w-3 h-3 text-[#764ba2]" />
        <span className="text-xs text-gray-300">{persona}</span>
      </div>
      <button
        onClick={copySessionId}
        className="ml-1 p-1 hover:bg-[#ffffff08] rounded transition-colors"
        title="Copy session ID"
      >
        {copied ? <CheckCircle2 className="w-3 h-3 text-[#10b981]" /> : <Hash className="w-3 h-3 text-gray-500" />}
      </button>
      <button
        onClick={onClear}
        className="p-1 hover:bg-red-500/10 rounded transition-colors"
        title="Clear session"
      >
        <XCircle className="w-3 h-3 text-gray-500 hover:text-red-400" />
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
  } = props;

  const [personaDropdownOpen, setPersonaDropdownOpen] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    showToast("success", "Copied to clipboard");
  };

  return (
    <main className="flex-1 flex flex-col min-w-0 bg-transparent relative z-10">
      {/* Compact Header Toolbar */}
      <header className="px-4 pt-14 md:pt-4 lg:pt-3 lg:px-6 flex-none">
        <div className="flex flex-wrap items-center gap-2 max-w-5xl mx-auto py-2">
          {/* Mode toggle */}
          <div className="flex items-center gap-0.5 p-1 bg-[#1a1a24]/80 rounded-xl border border-[#ffffff08]">
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
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${disabled
                      ? "opacity-40 cursor-not-allowed pointer-events-none text-gray-400"
                      : chatMode === mode
                        ? "bg-gradient-to-r from-[#667eea] to-[#764ba2] text-white shadow-sm"
                        : "text-gray-400 hover:text-gray-200 hover:bg-[#ffffff08]"
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
              <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-300 bg-[#1a1a24]/60 border border-[#ffffff08] hover:border-[#667eea]/30 hover:bg-[#1a1a24]/80 transition-all">
                <Download className="w-3 h-3" />
                Export
              </button>
              <div className="absolute top-full right-0 mt-1 py-1 bg-[#1a1a24]/95 backdrop-blur-xl rounded-lg border border-[#ffffff08] shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 min-w-[140px]">
                <button
                  onClick={() => exportChat('txt')}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-[#ffffff08] hover:text-white transition-colors"
                >
                  <FileTxt className="w-3 h-3" />
                  Text (.txt)
                </button>
                <button
                  onClick={() => exportChat('markdown')}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-[#ffffff08] hover:text-white transition-colors"
                >
                  <FileText className="w-3 h-3" />
                  Markdown (.md)
                </button>
                <button
                  onClick={() => exportChat('json')}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-[#ffffff08] hover:text-white transition-colors"
                >
                  <FileJson className="w-3 h-3" />
                  JSON (.json)
                </button>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Chat Output */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 md:px-5 lg:px-6 pb-4 max-w-5xl mx-auto w-full">
        {!sessionId && !isDbConnected && messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <p className="text-2xl font-semibold text-gray-200 mb-2 text-center">What do you want to analyze?</p>
            <p className="text-sm text-gray-500 text-center max-w-md mb-8">
              Upload financial documents, connect a live database, or browse SEC filings.
            </p>

            {/* Action buttons */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-xl w-full mb-10">
              <button
                onClick={onUploadClick}
                className="group flex flex-col items-center gap-3 p-5 bg-[#12121a]/60 rounded-2xl border border-[#ffffff08] hover:border-[#667eea]/40 hover:bg-[#667eea]/5 transition-all"
              >
                <div className="w-11 h-11 rounded-xl bg-[#667eea]/15 flex items-center justify-center group-hover:bg-[#667eea]/25 transition-colors">
                  <Upload className="w-5 h-5 text-[#667eea]" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-200">Upload Document</p>
                  <p className="text-xs text-gray-500 mt-0.5">PDF, CSV, or SQLite</p>
                </div>
              </button>

              <button
                onClick={onConnectDatabase}
                className="group flex flex-col items-center gap-3 p-5 bg-[#12121a]/60 rounded-2xl border border-[#ffffff08] hover:border-[#f97316]/40 hover:bg-[#f97316]/5 transition-all"
              >
                <div className="w-11 h-11 rounded-xl bg-[#f97316]/15 flex items-center justify-center group-hover:bg-[#f97316]/25 transition-colors">
                  <Database className="w-5 h-5 text-[#f97316]" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-200">Connect Database</p>
                  <p className="text-xs text-gray-500 mt-0.5">PostgreSQL, MySQL, Azure</p>
                </div>
              </button>

              <button
                onClick={onBrowseEdgar}
                className="group flex flex-col items-center gap-3 p-5 bg-[#12121a]/60 rounded-2xl border border-[#ffffff08] hover:border-[#10b981]/40 hover:bg-[#10b981]/5 transition-all"
              >
                <div className="w-11 h-11 rounded-xl bg-[#10b981]/15 flex items-center justify-center group-hover:bg-[#10b981]/25 transition-colors">
                  <BookOpen className="w-5 h-5 text-[#10b981]" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-200">Browse SEC Filings</p>
                  <p className="text-xs text-gray-500 mt-0.5">10-K, 10-Q, Annual Reports</p>
                </div>
              </button>
            </div>

            {/* Capability pills */}
            <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-gray-500">
              <span className="px-2.5 py-1 rounded-full bg-[#ffffff06] border border-[#ffffff08]">Hybrid Doc + DB Analysis</span>
              <span className="px-2.5 py-1 rounded-full bg-[#ffffff06] border border-[#ffffff08]">Discrepancy Detection</span>
              <span className="px-2.5 py-1 rounded-full bg-[#ffffff06] border border-[#ffffff08]">Analytical Autopilot</span>
            </div>
          </div>
        ) : messages.length === 0 && (sessionId || isDbConnected) ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#667eea]/20 to-[#764ba2]/20 flex items-center justify-center mb-4">
              <MessageSquare className="w-8 h-8 text-[#667eea]" />
            </div>
            <p className="text-lg font-medium text-gray-300 mb-2">
              {isDbConnected ? "Database connected — ready to analyze" : "Ready to analyze"}
            </p>
            <p className="text-sm text-gray-500">
              {isDbConnected
                ? `Query financials, metrics, and trends in ${chatMode} mode`
                : "Ask about revenue, margins, risk factors, or any financial metric"}
            </p>

            {/* Suggested questions */}
            <div className="mt-8 grid gap-2 max-w-lg w-full">
              <p className="text-xs text-gray-500 text-center mb-2">Try asking:</p>
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
                  className="text-left p-3 bg-[#12121a]/60 border border-[#ffffff08] rounded-xl text-sm text-gray-400 hover:text-white hover:border-[#667eea]/30 hover:bg-[#1a1a24]/60 transition-all"
                >
                  <Sparkles className="w-3 h-3 inline mr-2 text-[#667eea]" />
                  {question}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4 max-w-5xl mx-auto pb-20">
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
              <div className="flex justify-start animate-in slide-in-from-bottom-2 duration-300">
                <div className="max-w-[85%] lg:max-w-[80%] rounded-2xl p-4 bg-[#12121a]/80 border border-[#667eea]/20 text-[#e0e0e0] shadow-lg">
                  <div className="flex items-center gap-2 mb-3 pb-2 border-b border-[#ffffff08]">
                    <Wand2 className="w-4 h-4 text-[#a5b4fc]" />
                    <span className="text-xs font-semibold text-[#a5b4fc]">Autopilot Investigation</span>
                    <span className="ml-auto text-[10px] text-gray-500">{autopilotSteps.length}/{autopilotPlan.length} steps</span>
                  </div>
                  {/* Plan overview */}
                  {autopilotPlan.length > 0 && (
                    <div className="mb-3 space-y-1">
                      {autopilotPlan.map((planStep, pi) => {
                        const completed = autopilotSteps.find(s => s.step_num === pi + 1);
                        const isCurrent = !completed && autopilotSteps.length === pi;
                        return (
                          <div key={pi} className={`flex items-start gap-2 text-xs py-1 ${completed ? "text-gray-400" : isCurrent ? "text-white" : "text-gray-600"}`}>
                            <span className={`shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5 ${
                              completed ? "bg-[#10b981]/20 text-[#10b981]" : isCurrent ? "bg-[#667eea]/30 text-[#a5b4fc]" : "bg-[#ffffff08] text-gray-600"
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
                    <div className="space-y-2 border-t border-[#ffffff08] pt-2">
                      {autopilotSteps.map((step) => (
                        <div key={step.step_num} className="bg-[#ffffff05] rounded-xl p-2.5 text-xs">
                          <div className="flex items-center gap-1.5 mb-1">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${
                              step.tool === "sql_query" ? "bg-[#667eea]/20 text-[#a5b4fc]" :
                              step.tool === "doc_search" ? "bg-[#f97316]/20 text-[#fb923c]" :
                              "bg-[#10b981]/20 text-[#34d399]"
                            }`}>{step.tool.replace("_", " ")}</span>
                            <span className="text-gray-500 truncate">{step.step_label}</span>
                          </div>
                          {step.content && (
                            <p className="text-gray-400 line-clamp-2 leading-relaxed">{step.content}</p>
                          )}
                          {step.chart_b64 && (
                            <img src={`data:image/png;base64,${step.chart_b64}`} alt="chart" className="mt-2 rounded-lg max-h-32 object-contain" />
                          )}
                          {step.error && (
                            <p className="text-red-400 mt-1">⚠ {step.error}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Spinner while running */}
                  <div className="flex items-center gap-2 mt-2 pt-2 border-t border-[#ffffff08]">
                    <div className="w-3 h-3 rounded-full bg-[#667eea]/40 animate-pulse" />
                    <span className="text-xs text-gray-500">Investigating…</span>
                  </div>
                </div>
              </div>
            )}

            {/* Deep Research progress strip */}
            {isLoading && drProgress && deepResearch && (
              <div className="flex justify-start mb-2">
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-indigo-950/40 border border-indigo-500/20 text-indigo-300 text-xs max-w-sm">
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
      <div className="p-4 lg:p-6 pt-0 flex-none w-full">
        {/* Chart type selector — shown only when DB connected */}
        {isDbConnected && (
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="text-xs text-gray-500 shrink-0">Chart:</span>
            {(["auto", "bar", "line", "scatter", "heatmap", "box", "multi"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setChartType(t)}
                className={`px-2 py-0.5 rounded-full text-xs border transition-all capitalize ${
                  chartType === t
                    ? "border-[#667eea]/60 bg-[#667eea]/15 text-[#a5b4fc]"
                    : "border-[#ffffff10] text-gray-500 hover:text-gray-300 hover:border-[#ffffff20]"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        {/* Persona quick-switch */}
        {setSelectedPersona && (
          <div className="flex items-center gap-2 mb-2">
            <div className="relative">
              <button
                onClick={() => setPersonaDropdownOpen(!personaDropdownOpen)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs bg-[#1a1a24]/80 border border-[#ffffff10] hover:border-[#667eea]/30 transition-all text-gray-300"
              >
                <Sparkles className="w-3 h-3 text-[#667eea]" />
                {isAutoMode ? "Auto" : (PERSONA_DISPLAY[selectedPersona] || selectedPersona)}
                <ChevronDown className={`w-3 h-3 text-gray-500 transition-transform ${personaDropdownOpen ? "rotate-180" : ""}`} />
              </button>
              {personaDropdownOpen && (
                <div className="absolute bottom-full left-0 mb-1 py-1 bg-[#1a1a24]/95 backdrop-blur-xl rounded-lg border border-[#ffffff08] shadow-xl z-50 min-w-[160px]">
                  <button
                    onClick={() => { setAutoMode?.(true); setPersonaDropdownOpen(false); }}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-xs transition-colors ${
                      isAutoMode ? "text-[#a5b4fc] bg-[#667eea]/10" : "text-gray-300 hover:bg-[#ffffff08]"
                    }`}
                  >
                    <Wand2 className="w-3 h-3" />
                    Auto-routing
                  </button>
                  {PERSONA_OPTIONS.map((name) => (
                    <button
                      key={name}
                      onClick={() => {
                        setAutoMode?.(false);
                        setSelectedPersona(name);
                        setPersonaDropdownOpen(false);
                      }}
                      className={`w-full flex items-center gap-2 px-3 py-2 text-xs transition-colors ${
                        !isAutoMode && selectedPersona === name
                          ? "text-[#a5b4fc] bg-[#667eea]/10"
                          : "text-gray-300 hover:bg-[#ffffff08]"
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

        <form
          onSubmit={handleSendMessage}
          className="relative bg-[#12121a]/90 border border-[#ffffff10] rounded-2xl backdrop-blur-xl shadow-lg shadow-black/20 focus-within:border-[#667eea]/40 focus-within:shadow-[0_8px_30px_rgba(102,126,234,0.15)] transition-all overflow-hidden flex items-end"
        >
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
              className="w-full max-h-40 min-h-[60px] p-4 pr-12 bg-transparent outline-none resize-none text-white placeholder-gray-500 text-sm leading-relaxed disabled:opacity-50"
              rows={1}
            />
            <div className="absolute bottom-3 right-3 text-[10px] text-gray-600 flex items-center gap-1.5">
              {input.length > 0 && <span>{input.length} chars</span>}
            </div>
          </div>
          <button
            type="submit"
            disabled={!input.trim() || (!sessionId && !isDbConnected) || isLoading}
            className="m-2 p-3.5 rounded-xl bg-gradient-to-tr from-[#667eea] to-[#764ba2] text-white shadow-lg shadow-[#667eea]/20 disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-xl hover:shadow-[#667eea]/30 transition-all focus:outline-none focus:ring-2 focus:ring-[#764ba2]/50 hover:scale-105 active:scale-95"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </form>
        <div className="text-center mt-3 text-xs text-gray-600 font-medium flex items-center justify-center gap-2">
          <HelpCircle className="w-3 h-3" />
          AI-generated analysis. Always verify against source documents.
        </div>
      </div>

    </main>
  );
}
