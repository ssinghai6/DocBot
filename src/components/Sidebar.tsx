"use client"

import React, { useState } from "react"
import {
  Brain, X, Menu, Trash2, Keyboard,
  Database, LogOut, ShieldCheck, Shield,
  ChevronDown, Sparkles, Wrench, Settings,
} from "lucide-react"
import FileUploadZone from "@/components/FileUploadZone"
import ConnectionPanel from "@/components/ConnectionPanel"
import MarketplacePanel from "@/components/MarketplacePanel"
import EdgarPanel from "@/components/EdgarPanel"
import PersonaSelector from "@/components/PersonaSelector"
import type {
  AuthUser,
  FileUploadState,
  WorkspaceConnection,
  QueryHistoryItem,
  LiveDbForm,
  ConnectorInfo,
  ConnectorSyncResponse,
} from "@/components/types"

export interface SidebarProps {
  sidebarOpen: boolean
  setSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>

  authChecked: boolean
  authUser: AuthUser | null
  handleLogout: () => void
  setAdminPanelOpen: React.Dispatch<React.SetStateAction<boolean>>
  setAuthModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  setAuthError: React.Dispatch<React.SetStateAction<string | null>>

  workspaceConnections: WorkspaceConnection[]
  setShowLiveDbForm: React.Dispatch<React.SetStateAction<boolean>>
  setLiveDbForm: React.Dispatch<React.SetStateAction<LiveDbForm>>
  showToast: (type: "success" | "error" | "warning" | "info", message: string) => void

  fileUploadState: FileUploadState
  uploadProgress: number | null
  uploadedFiles: File[]
  deepVisualMode: boolean
  onDeepVisualModeChange: React.Dispatch<React.SetStateAction<boolean>>
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  fileInputRef: React.RefObject<HTMLInputElement | null>

  isDbConnected: boolean
  dbFileName: string | null
  dbUploadState: "idle" | "uploading" | "connected" | "error"
  showLiveDbForm: boolean
  liveDbForm: LiveDbForm
  showDbPassword: boolean
  liveDbConnectState: "idle" | "connecting" | "error"
  liveDbError: string | null
  entraToken: string | null
  entraEmail: string | null
  entraSignInState: "idle" | "signing_in" | "signed_in" | "error"
  autopilotMode: boolean
  hasDocSession: boolean
  queryHistory: QueryHistoryItem[]
  historyOpen: boolean
  expandedHistoryId: string | null
  onDbUpload: (file: File, type: "csv" | "sqlite") => void
  onDbDisconnect: () => void
  onToggleLiveDbForm: () => void
  onLiveDbFormChange: React.Dispatch<React.SetStateAction<LiveDbForm>>
  onShowDbPasswordChange: React.Dispatch<React.SetStateAction<boolean>>
  onLiveDbConnect: () => void
  onAutopilotToggle: () => void
  onHistoryToggle: () => void
  onExpandedHistoryChange: React.Dispatch<React.SetStateAction<string | null>>
  onSetInput: React.Dispatch<React.SetStateAction<string>>
  onMicrosoftSignIn: () => void
  onEntraReset: () => void

  selectedPersona: string
  suggestedPersona: string | null
  isAutoMode: boolean
  deepResearch: boolean
  onSelectPersona: React.Dispatch<React.SetStateAction<string>>
  onSetAutoMode: React.Dispatch<React.SetStateAction<boolean>>
  onDeepResearchChange: React.Dispatch<React.SetStateAction<boolean>>

  clearChat: () => void
  messagesLength: number

  // Marketplace connectors
  connectors: ConnectorInfo[]
  onConnectorRegister: (connectorType: string, credentials: Record<string, string>) => Promise<void>
  onConnectorSync: (connectorId: string, startDate: string, endDate: string) => Promise<ConnectorSyncResponse>
  onConnectorDisconnect: (connectorId: string) => void

  // EDGAR SEC filings
  onEdgarFilingIngested: (sessionId: string, label: string) => void
}

export default function Sidebar(props: SidebarProps) {
  const {
    sidebarOpen,
    setSidebarOpen,
    authChecked,
    authUser,
    handleLogout,
    setAdminPanelOpen,
    setAuthModalOpen,
    setAuthError,
    workspaceConnections,
    setShowLiveDbForm,
    setLiveDbForm,
    showToast,
    fileUploadState,
    uploadProgress,
    uploadedFiles,
    deepVisualMode,
    onDeepVisualModeChange,
    onDragOver,
    onDragLeave,
    onDrop,
    onFileChange,
    fileInputRef,
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
    selectedPersona,
    suggestedPersona,
    isAutoMode,
    deepResearch,
    onSelectPersona,
    onSetAutoMode,
    onDeepResearchChange,
    clearChat,
    messagesLength,
    connectors,
    onConnectorRegister,
    onConnectorSync,
    onConnectorDisconnect,
    onEdgarFilingIngested,
  } = props;

  const [marketplaceOpen, setMarketplaceOpen] = useState(false);
  const [personaOpen, setPersonaOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"sources" | "tools" | "settings">("sources");

  return (
    <>
      {/* Mobile Menu Toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed top-3 left-3 z-50 p-2 bg-[var(--color-bg-elevated)] rounded-[5px] border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)]"
      >
        {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
      </button>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-[var(--bg-scrim)] z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        w-[260px] bg-[var(--color-bg-surface)] border-r border-[var(--color-border-subtle)]
        flex flex-col p-4 z-40 shrink-0 overflow-y-auto
        transition-transform duration-[var(--duration-base)] ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        fixed lg:relative h-full
      `}>
        {/* Logo */}
        <div className="flex items-center gap-2.5 mb-4 h-11">
          <div className="w-8 h-8 rounded-[5px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/30 flex items-center justify-center">
            <Brain className="w-4 h-4 text-[var(--color-cyan-500)]" />
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-[13px] font-semibold text-[var(--color-text-primary)] flex items-center gap-1.5 leading-none">
              DocBot
              <span className="px-1 py-px bg-[var(--color-cyan-500)]/15 text-[var(--color-cyan-500)] text-[9px] font-bold rounded-[3px] uppercase tracking-wider">AI</span>
            </h2>
            <p className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5">Analytical workspace</p>
          </div>
        </div>

        {/* Auth widget */}
        {authChecked && (
          <div className="mb-4">
            {authUser ? (
              <div className="flex items-center gap-2 px-2.5 py-2 bg-[var(--color-bg-elevated)] rounded-[5px] border border-[var(--color-border-subtle)]">
                <div className="w-6 h-6 rounded-[3px] bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/20 flex items-center justify-center shrink-0">
                  <ShieldCheck className="w-3 h-3 text-[var(--color-cyan-500)]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-medium text-[var(--color-text-primary)] truncate">{authUser.name || authUser.email}</p>
                  <p className="text-[10px] text-[var(--color-text-tertiary)] truncate">{authUser.email}</p>
                </div>
                {authUser.role === "admin" && (
                  <button
                    onClick={() => setAdminPanelOpen(true)}
                    className="p-1 rounded-[3px] hover:bg-[var(--color-bg-overlay)] text-[var(--color-text-tertiary)] hover:text-[var(--color-amber-500)] transition-colors"
                    title="Admin panel"
                  >
                    <Shield className="w-3 h-3" />
                  </button>
                )}
                <button
                  onClick={handleLogout}
                  className="p-1 rounded-[3px] hover:bg-[var(--color-danger-500)]/10 text-[var(--color-text-tertiary)] hover:text-[var(--color-danger-500)] transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-3 h-3" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => { setAuthModalOpen(true); setAuthError(null); }}
                className="flex items-center justify-center gap-1.5 w-full px-3 py-2 rounded-[5px] bg-[var(--color-cyan-500)]/10 hover:bg-[var(--color-cyan-500)]/15 border border-[var(--color-cyan-500)]/30 text-[var(--color-cyan-500)] text-[11px] font-medium transition-colors"
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                Sign in / Create account
              </button>
            )}
          </div>
        )}

        <FileUploadZone
          fileUploadState={fileUploadState}
          uploadProgress={uploadProgress}
          uploadedFiles={uploadedFiles}
          deepVisualMode={deepVisualMode}
          onDeepVisualModeChange={onDeepVisualModeChange}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onFileChange={onFileChange}
          fileInputRef={fileInputRef}
          onDbUpload={onDbUpload}
          dbUploadState={dbUploadState}
        />

        {/* ── Tab Bar ── */}
        <div className="flex items-center border-b border-[var(--color-border-subtle)] mb-4">
          {([
            { key: "sources" as const, label: "Sources", icon: <Database className="w-3 h-3" />, badge: isDbConnected },
            { key: "tools" as const, label: "Tools", icon: <Wrench className="w-3 h-3" />, badge: false },
            { key: "settings" as const, label: "Settings", icon: <Settings className="w-3 h-3" />, badge: false },
          ]).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`relative flex-1 flex items-center justify-center gap-1.5 h-8 text-[11px] font-medium transition-colors ${
                activeTab === tab.key
                  ? "text-[var(--color-text-primary)] after:content-[''] after:absolute after:left-0 after:right-0 after:-bottom-px after:h-[2px] after:bg-[var(--color-cyan-500)]"
                  : "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"
              }`}
            >
              {tab.icon}
              {tab.label}
              {tab.badge && <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-success-500)]" />}
            </button>
          ))}
        </div>

        {/* ── Tab Content ── */}
        <div className="flex-1 min-h-0">
          {/* SOURCES TAB */}
          {activeTab === "sources" && (
            <div>
              {/* Saved connections */}
              {authUser && workspaceConnections.filter(wc => wc.host !== "__local_file__").length > 0 && (
                <div className="mb-4">
                  <h3 className="text-[10px] font-semibold uppercase tracking-wider mb-2 text-[var(--color-text-tertiary)] flex items-center gap-1.5">
                    <Database className="w-3 h-3" />
                    Saved connections
                  </h3>
                  <ul className="space-y-0.5">
                    {workspaceConnections.filter(wc => wc.host !== "__local_file__").map((wc) => (
                      <li key={wc.id}>
                        <button
                          onClick={() => {
                            setShowLiveDbForm(true);
                            setLiveDbForm(prev => ({
                              ...prev,
                              dialect: wc.dialect,
                              host: wc.host,
                              dbname: wc.db_name,
                            }));
                            showToast("info", "Connection details loaded — enter credentials and connect");
                          }}
                          className="w-full text-left px-2 py-1.5 rounded-[3px] hover:bg-[var(--color-bg-overlay)] transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            <Database className="w-3 h-3 text-[var(--color-success-500)] shrink-0" />
                            <span className="text-[11px] text-[var(--color-text-secondary)] truncate flex-1">
                              {wc.dialect} · {wc.db_name}
                            </span>
                          </div>
                          <p className="text-[10px] text-[var(--color-text-quaternary)] mt-0.5 pl-5 truncate">{wc.host}</p>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Live DB / Connection Panel */}
              <ConnectionPanel
                isDbConnected={isDbConnected}
                dbFileName={dbFileName}
                dbUploadState={dbUploadState}
                showLiveDbForm={showLiveDbForm}
                liveDbForm={liveDbForm}
                showDbPassword={showDbPassword}
                liveDbConnectState={liveDbConnectState}
                liveDbError={liveDbError}
                entraToken={entraToken}
                entraEmail={entraEmail}
                entraSignInState={entraSignInState}
                autopilotMode={autopilotMode}
                hasDocSession={hasDocSession}
                queryHistory={queryHistory}
                historyOpen={historyOpen}
                expandedHistoryId={expandedHistoryId}
                onDbUpload={onDbUpload}
                onDbDisconnect={onDbDisconnect}
                onToggleLiveDbForm={onToggleLiveDbForm}
                onLiveDbFormChange={onLiveDbFormChange}
                onShowDbPasswordChange={onShowDbPasswordChange}
                onLiveDbConnect={onLiveDbConnect}
                onAutopilotToggle={onAutopilotToggle}
                onHistoryToggle={onHistoryToggle}
                onExpandedHistoryChange={onExpandedHistoryChange}
                onSetInput={onSetInput}
                onMicrosoftSignIn={onMicrosoftSignIn}
                onEntraReset={onEntraReset}
              />
            </div>
          )}

          {/* TOOLS TAB */}
          {activeTab === "tools" && (
            <div>
              {/* EDGAR */}
              <EdgarPanel
                onFilingIngested={onEdgarFilingIngested}
                showToast={showToast}
              />

              {/* Marketplace */}
              <div className="mb-4">
                <button
                  onClick={() => setMarketplaceOpen(!marketplaceOpen)}
                  className="w-full flex items-center gap-2 mb-2"
                >
                  <span className="text-[10px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider flex-1 text-left">Marketplace</span>
                  {connectors.length > 0 && (
                    <span className="text-[10px] text-[var(--color-cyan-500)] font-medium bg-[var(--color-cyan-500)]/10 px-1.5 py-0.5 rounded-[3px]">
                      {connectors.length}
                    </span>
                  )}
                  <ChevronDown className={`w-3 h-3 text-[var(--color-text-tertiary)] transition-transform ${marketplaceOpen ? "rotate-180" : ""}`} />
                </button>
                {marketplaceOpen && (
                  <MarketplacePanel
                    connectors={connectors}
                    onRegister={onConnectorRegister}
                    onSync={onConnectorSync}
                    onDisconnect={onConnectorDisconnect}
                  />
                )}
              </div>

              {/* Persona Selector */}
              <div className="mb-4">
                <button
                  onClick={() => setPersonaOpen(!personaOpen)}
                  className="w-full flex items-center gap-2 mb-2"
                >
                  <Sparkles className="w-3 h-3 text-[var(--color-amber-500)]" />
                  <span className="text-[10px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider flex-1 text-left">
                    {isAutoMode ? "Auto-routing" : selectedPersona}
                  </span>
                  <ChevronDown className={`w-3 h-3 text-[var(--color-text-tertiary)] transition-transform ${personaOpen ? "rotate-180" : ""}`} />
                </button>
                {personaOpen && (
                  <PersonaSelector
                    selectedPersona={selectedPersona}
                    suggestedPersona={suggestedPersona}
                    isAutoMode={isAutoMode}
                    deepResearch={deepResearch}
                    onSelectPersona={onSelectPersona}
                    onSetAutoMode={onSetAutoMode}
                    onDeepResearchChange={onDeepResearchChange}
                  />
                )}
              </div>
            </div>
          )}

          {/* SETTINGS TAB */}
          {activeTab === "settings" && (
            <div>
              {/* Clear Chat */}
              <button
                onClick={clearChat}
                disabled={messagesLength === 0}
                className="w-full flex items-center justify-center h-8 rounded-[5px] bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] hover:bg-[var(--color-danger-500)]/10 hover:border-[var(--color-danger-500)]/30 hover:text-[var(--color-danger-500)] transition-colors text-[11px] font-medium disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-[var(--color-bg-elevated)] disabled:hover:border-[var(--color-border-subtle)] disabled:hover:text-[var(--color-text-secondary)] mb-4"
              >
                <Trash2 className="w-3.5 h-3.5 mr-1.5" />
                Clear chat
              </button>

              {/* Keyboard Shortcuts */}
              <div className="p-2.5 bg-[var(--color-bg-inset)] rounded-[5px] border border-[var(--color-border-subtle)] mb-4">
                <div className="flex items-center gap-1.5 text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-2">
                  <Keyboard className="w-3 h-3" />
                  <span className="font-semibold">Keyboard shortcuts</span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 text-[10px] text-[var(--color-text-tertiary)]">
                  <div className="flex items-center gap-1">
                    <kbd className="px-1 py-px bg-[var(--color-bg-overlay)] border border-[var(--color-border-default)] rounded-[3px] text-[var(--color-text-secondary)] font-mono">⌘↵</kbd>
                    <span className="ml-1">Send</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <kbd className="px-1 py-px bg-[var(--color-bg-overlay)] border border-[var(--color-border-default)] rounded-[3px] text-[var(--color-text-secondary)] font-mono">Esc</kbd>
                    <span className="ml-1">Clear</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <kbd className="px-1 py-px bg-[var(--color-bg-overlay)] border border-[var(--color-border-default)] rounded-[3px] text-[var(--color-text-secondary)] font-mono">⌘K</kbd>
                    <span className="ml-1">Commands</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <kbd className="px-1 py-px bg-[var(--color-bg-overlay)] border border-[var(--color-border-default)] rounded-[3px] text-[var(--color-text-secondary)] font-mono">⌘I</kbd>
                    <span className="ml-1">Inspector</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="pt-3 mt-auto text-center text-[10px] text-[var(--color-text-quaternary)]">
          <p>Built by <a href="https://sanshrit-singhai.vercel.app" className="text-[var(--color-cyan-500)] hover:underline" target="_blank" rel="noopener noreferrer">Sanshrit Singhai</a></p>
        </div>
      </aside>
    </>
  );
}
