"use client"

import React from "react"
import {
  Database, Loader2, RefreshCw, ChevronDown,
  XCircle, Wand2, Clock, ArrowRight, Eye, EyeOff,
} from "lucide-react"
import type { LiveDbForm, QueryHistoryItem } from "./types"

interface ConnectionPanelProps {
  // Connection state
  isDbConnected: boolean
  dbFileName: string | null
  dbUploadState: "idle" | "uploading" | "connected" | "error"

  // Live DB form
  showLiveDbForm: boolean
  liveDbForm: LiveDbForm
  showDbPassword: boolean
  liveDbConnectState: "idle" | "connecting" | "error"
  liveDbError: string | null

  // Entra (Azure AD) state
  entraToken: string | null
  entraEmail: string | null
  entraSignInState: "idle" | "signing_in" | "signed_in" | "error"

  // Autopilot
  autopilotMode: boolean
  hasDocSession: boolean

  // Query history
  queryHistory: QueryHistoryItem[]
  historyOpen: boolean
  expandedHistoryId: string | null

  // Callbacks
  onDbUpload: (file: File, type: "csv" | "sqlite") => void
  onDbDisconnect: () => void
  onToggleLiveDbForm: () => void
  onLiveDbFormChange: (form: LiveDbForm) => void
  onShowDbPasswordChange: (v: boolean) => void
  onLiveDbConnect: () => void
  onAutopilotToggle: () => void
  onHistoryToggle: () => void
  onExpandedHistoryChange: (id: string | null) => void
  onSetInput: (value: string) => void
  onMicrosoftSignIn: () => void
  onEntraReset: () => void
}

export default function ConnectionPanel({
  isDbConnected,
  dbFileName,
  dbUploadState,
  showLiveDbForm,
  liveDbForm,
  showDbPassword,
  liveDbConnectState,
  liveDbError,
  entraToken,
  entraEmail,
  entraSignInState,
  autopilotMode,
  hasDocSession,
  queryHistory,
  historyOpen,
  expandedHistoryId,
  onDbUpload,
  onDbDisconnect,
  onToggleLiveDbForm,
  onLiveDbFormChange,
  onShowDbPasswordChange,
  onLiveDbConnect,
  onAutopilotToggle,
  onHistoryToggle,
  onExpandedHistoryChange,
  onSetInput,
  onMicrosoftSignIn,
  onEntraReset,
}: ConnectionPanelProps) {
  return (
    <div className="mb-5">
      <h3 className="text-sm font-semibold mb-3 text-white flex items-center gap-2">
        <Database className="w-4 h-4 text-[#f97316]" />
        Database
        {isDbConnected && (
          <span className="ml-auto text-[10px] text-[#10b981] font-medium flex items-center gap-1">
            <div className="w-1.5 h-1.5 bg-[#10b981] rounded-full animate-pulse" />
            Connected
          </span>
        )}
      </h3>

      {isDbConnected ? (
        <div className="space-y-2">
          {/* Connected indicator */}
          <div className="flex items-center gap-2 px-3 py-2.5 bg-[#10b981]/10 rounded-xl border border-[#10b981]/20 text-xs">
            <Database className="w-3.5 h-3.5 text-[#10b981] shrink-0" />
            <span className="truncate text-gray-300 flex-1">{dbFileName}</span>
            <button
              onClick={onDbDisconnect}
              className="text-gray-500 hover:text-red-400 transition-colors shrink-0"
              title="Disconnect database"
            >
              <XCircle className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* DOCBOT-504: Query History Panel */}
          <div className="rounded-xl border border-[#ffffff10] overflow-hidden">
            <button
              onClick={onHistoryToggle}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-400 hover:text-white hover:bg-[#ffffff08] transition-colors"
            >
              <Clock className="w-3.5 h-3.5 text-[#f97316] shrink-0" />
              <span className="flex-1 text-left font-medium">Query History</span>
              {queryHistory.length > 0 && (
                <span className="bg-[#f97316]/20 text-[#f97316] text-[10px] px-1.5 py-0.5 rounded-full font-medium">
                  {queryHistory.length}
                </span>
              )}
              <ChevronDown className={`w-3 h-3 transition-transform ${historyOpen ? "rotate-180" : ""}`} />
            </button>

            {historyOpen && (
              <div className="border-t border-[#ffffff10] max-h-64 overflow-y-auto">
                {queryHistory.length === 0 ? (
                  <p className="px-3 py-3 text-xs text-gray-500 text-center">No queries yet</p>
                ) : (
                  queryHistory.map((item) => (
                    <div key={item.id} className="border-b border-[#ffffff08] last:border-b-0">
                      <button
                        className="w-full flex items-start gap-2 px-3 py-2 text-left hover:bg-[#ffffff06] transition-colors group"
                        onClick={() => onExpandedHistoryChange(expandedHistoryId === item.id ? null : item.id)}
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-gray-300 truncate leading-snug">{item.question}</p>
                          <div className="flex items-center gap-2 mt-0.5">
                            {item.row_count != null && (
                              <span className="text-[10px] text-gray-500">
                                {item.row_count} row{item.row_count !== 1 ? "s" : ""}
                              </span>
                            )}
                            {item.executed_at && (
                              <span className="text-[10px] text-gray-600">
                                {new Date(item.executed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            )}
                          </div>
                        </div>
                        <button
                          title="Re-run this query"
                          className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-gray-500 hover:text-[#f97316] mt-0.5"
                          onClick={(e) => {
                            e.stopPropagation()
                            onSetInput(item.question)
                          }}
                        >
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </button>

                      {expandedHistoryId === item.id && (
                        <div className="px-3 pb-2">
                          <pre className="text-[10px] text-gray-400 bg-[#0d0d14] rounded-lg p-2 overflow-x-auto whitespace-pre-wrap break-all leading-relaxed">
                            {item.sql}
                          </pre>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {/* Live DB connect toggle */}
          <button
            onClick={onToggleLiveDbForm}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-[#ffffff10] bg-[#1a1a24]/50 hover:border-[#f97316]/40 hover:bg-[#f97316]/5 transition-all text-xs text-gray-400"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[#f97316] shrink-0" />
            <span>Connect Live DB</span>
            <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${showLiveDbForm ? "rotate-180" : ""}`} />
          </button>

          {/* Live DB form */}
          {showLiveDbForm && (
            <div className="space-y-2 pt-1">
              {/* Dialect selector */}
              <select
                value={liveDbForm.dialect}
                onChange={(e) => {
                  const dialect = e.target.value
                  onLiveDbFormChange({
                    ...liveDbForm,
                    dialect,
                    port: dialect === "postgresql" ? "5432"
                        : dialect === "mysql"       ? "3306"
                        : dialect === "azure_sql"   ? "1433"
                        : "0",
                    user: "",
                    password: "",
                  })
                  onEntraReset()
                }}
                className="w-full px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs focus:outline-none focus:border-[#f97316]/40"
              >
                <option value="postgresql">PostgreSQL</option>
                <option value="mysql">MySQL</option>
                <option value="azure_sql">Azure SQL (Microsoft Entra)</option>
              </select>

              <input
                type="text"
                placeholder="Host"
                value={liveDbForm.host}
                onChange={(e) => onLiveDbFormChange({ ...liveDbForm, host: e.target.value })}
                className="w-full px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-500 focus:outline-none focus:border-[#f97316]/40"
              />

              <div className="flex gap-2">
                <input
                  type="number"
                  placeholder="Port"
                  value={liveDbForm.port}
                  onChange={(e) => onLiveDbFormChange({ ...liveDbForm, port: e.target.value })}
                  className="w-20 px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-500 focus:outline-none focus:border-[#f97316]/40"
                />
                <input
                  type="text"
                  placeholder="Database name"
                  value={liveDbForm.dbname}
                  onChange={(e) => onLiveDbFormChange({ ...liveDbForm, dbname: e.target.value })}
                  className="flex-1 px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-500 focus:outline-none focus:border-[#f97316]/40"
                />
              </div>

              {liveDbForm.dialect === "azure_sql" ? (
                <div className="space-y-2">
                  {entraSignInState === "signed_in" && entraEmail ? (
                    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/20">
                      <div className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0" />
                      <span className="text-xs text-green-400 truncate">{entraEmail}</span>
                      <button
                        type="button"
                        onClick={onEntraReset}
                        className="ml-auto text-gray-500 hover:text-gray-300 text-[10px]"
                      >
                        Change
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={onMicrosoftSignIn}
                      disabled={entraSignInState === "signing_in" || !process.env.NEXT_PUBLIC_AZURE_CLIENT_ID}
                      className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[#0078d4]/20 hover:bg-[#0078d4]/30 border border-[#0078d4]/30 text-[#4fc3f7] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {entraSignInState === "signing_in"
                        ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Signing in...</>
                        : "Sign in with Microsoft"
                      }
                    </button>
                  )}
                  {entraSignInState === "error" && (
                    <p className="text-[10px] text-red-400 px-1">Sign-in failed. Check your Azure AD app configuration.</p>
                  )}
                  {!process.env.NEXT_PUBLIC_AZURE_CLIENT_ID && (
                    <p className="text-[10px] text-yellow-500/70 px-1">Set NEXT_PUBLIC_AZURE_CLIENT_ID to enable Microsoft sign-in.</p>
                  )}
                </div>
              ) : (
                <>
                  <input
                    type="text"
                    placeholder="Username"
                    value={liveDbForm.user}
                    onChange={(e) => onLiveDbFormChange({ ...liveDbForm, user: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-500 focus:outline-none focus:border-[#f97316]/40"
                  />
                  <div className="relative">
                    <input
                      type={showDbPassword ? "text" : "password"}
                      placeholder="Password"
                      value={liveDbForm.password}
                      onChange={(e) => onLiveDbFormChange({ ...liveDbForm, password: e.target.value })}
                      className="w-full px-3 py-2 pr-8 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-500 focus:outline-none focus:border-[#f97316]/40"
                    />
                    <button
                      type="button"
                      onClick={() => onShowDbPasswordChange(!showDbPassword)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                    >
                      {showDbPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </>
              )}

              {/* PII Masking toggle — DOCBOT-604 */}
              <button
                type="button"
                onClick={() => onLiveDbFormChange({ ...liveDbForm, pii_masking_enabled: !liveDbForm.pii_masking_enabled })}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] hover:border-[#f97316]/20 transition-all"
              >
                <span className="text-xs text-gray-400">PII masking</span>
                <div className={`relative w-8 h-4 rounded-full transition-colors ${liveDbForm.pii_masking_enabled ? "bg-[#f97316]" : "bg-[#ffffff15]"}`}>
                  <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${liveDbForm.pii_masking_enabled ? "translate-x-4" : "translate-x-0.5"}`} />
                </div>
              </button>

              {liveDbError && (
                <p className="text-[10px] text-red-400 px-1">{liveDbError}</p>
              )}

              <button
                onClick={onLiveDbConnect}
                disabled={
                  liveDbConnectState === "connecting" ||
                  !liveDbForm.host ||
                  !liveDbForm.dbname ||
                  (liveDbForm.dialect === "azure_sql"
                    ? !entraToken
                    : !liveDbForm.user)
                }
                className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[#f97316]/20 hover:bg-[#f97316]/30 border border-[#f97316]/30 text-[#f97316] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {liveDbConnectState === "connecting"
                  ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Connecting...</>
                  : <><Database className="w-3.5 h-3.5" /> Connect</>
                }
              </button>
            </div>
          )}
        </div>
      )}

      {/* DOCBOT-405: Autopilot toggle — visible when DB connected OR docs uploaded */}
      {(isDbConnected || hasDocSession) && (
        <button
          onClick={onAutopilotToggle}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-xl text-xs border transition-all mt-2 ${
            autopilotMode
              ? "border-[#667eea]/50 bg-[#667eea]/10 text-[#a5b4fc]"
              : "border-[#ffffff10] text-gray-400 hover:text-white hover:bg-[#ffffff08]"
          }`}
        >
          <Wand2 className={`w-3.5 h-3.5 shrink-0 ${autopilotMode ? "text-[#a5b4fc]" : "text-[#667eea]"}`} />
          <span className="flex-1 text-left font-medium">Autopilot</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
            autopilotMode ? "bg-[#667eea]/30 text-[#a5b4fc]" : "bg-[#ffffff10] text-gray-500"
          }`}>
            {autopilotMode ? "ON" : "OFF"}
          </span>
        </button>
      )}
    </div>
  )
}
