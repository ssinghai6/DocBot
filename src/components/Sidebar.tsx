"use client"

import React from "react"
import {
  Brain, X, Menu, Trash2, Keyboard,
  Database, LogOut, ShieldCheck, Shield,
} from "lucide-react"
import FileUploadZone from "@/components/FileUploadZone"
import ConnectionPanel from "@/components/ConnectionPanel"
import MarketplacePanel from "@/components/MarketplacePanel"
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
  } = props;

  return (
    <>
      {/* Mobile Menu Toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-[#12121a]/90 backdrop-blur-xl rounded-lg border border-[#ffffff08]"
      >
        {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Settings / Options Sidebar */}
      <aside className={`
        w-80 backdrop-blur-2xl bg-[#12121a]/95 border-r border-[#ffffff08] flex flex-col p-5 z-40 shrink-0 shadow-2xl overflow-y-auto
        transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        fixed lg:relative h-full
      `}>
        {/* Logo */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#667eea] to-[#764ba2] flex items-center justify-center shadow-lg shadow-[#667eea]/20">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-bold text-white tracking-wide flex items-center gap-2">
              DocBot
              <span className="px-1.5 py-0.5 bg-[#667eea]/20 text-[#667eea] text-[10px] font-bold rounded">AI</span>
            </h2>
            <p className="text-xs text-gray-500">Document Intelligence</p>
          </div>
        </div>

        {/* Auth widget */}
        {authChecked && (
          <div className="mb-5">
            {authUser ? (
              <div className="flex items-center gap-2 px-3 py-2.5 bg-[#1a1a24]/70 rounded-xl border border-[#ffffff08]">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#667eea]/30 to-[#764ba2]/30 flex items-center justify-center shrink-0">
                  <ShieldCheck className="w-3.5 h-3.5 text-[#a5b4fc]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-200 truncate">{authUser.name || authUser.email}</p>
                  <p className="text-[10px] text-gray-500 truncate">{authUser.email}</p>
                </div>
                {authUser.role === "admin" && (
                  <button
                    onClick={() => setAdminPanelOpen(true)}
                    className="p-1.5 rounded-lg hover:bg-[#ffffff10] text-gray-500 hover:text-[#f59e0b] transition-colors"
                    title="Admin panel"
                  >
                    <Shield className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  onClick={handleLogout}
                  className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-500 hover:text-red-400 transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => { setAuthModalOpen(true); setAuthError(null); }}
                className="flex items-center justify-center gap-2 w-full px-3 py-2.5 rounded-xl bg-[#667eea]/15 hover:bg-[#667eea]/25 border border-[#667eea]/30 text-[#a5b4fc] text-xs font-medium transition-all"
              >
                <ShieldCheck className="w-4 h-4" />
                Sign in / Create account
              </button>
            )}
          </div>
        )}

        {/* Persistent Workspace: Saved DB connections */}
        {authUser && workspaceConnections.filter(wc => wc.host !== "__local_file__").length > 0 && (
          <div className="mb-5">
            <h3 className="text-xs font-semibold mb-2 text-gray-400 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" />
              Saved connections
            </h3>
            <ul className="space-y-1">
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
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-[#ffffff08] transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Database className="w-3 h-3 text-[#10b981] shrink-0" />
                      <span className="text-[11px] text-gray-300 truncate flex-1">
                        {wc.dialect} · {wc.db_name}
                      </span>
                    </div>
                    <p className="text-[10px] text-gray-600 mt-0.5 pl-5 truncate">{wc.host}</p>
                  </button>
                </li>
              ))}
            </ul>
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
        />

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

        <MarketplacePanel
          connectors={connectors}
          onRegister={onConnectorRegister}
          onSync={onConnectorSync}
          onDisconnect={onConnectorDisconnect}
        />

        <PersonaSelector
          selectedPersona={selectedPersona}
          suggestedPersona={suggestedPersona}
          isAutoMode={isAutoMode}
          deepResearch={deepResearch}
          onSelectPersona={onSelectPersona}
          onSetAutoMode={onSetAutoMode}
          onDeepResearchChange={onDeepResearchChange}
        />

        {/* Footer */}
        <div className="pt-4 border-t border-[#ffffff08]">
          <button
            onClick={clearChat}
            disabled={messagesLength === 0}
            className="w-full flex items-center justify-center py-2.5 px-4 rounded-xl bg-[#1a1a24]/80 border border-[#ffffff06] hover:bg-red-500/15 hover:border-red-500/30 hover:text-red-400 transition-all text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-[#1a1a24]/80 disabled:hover:border-[#ffffff06] disabled:hover:text-inherit"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Clear Chat
          </button>

          {/* Keyboard Shortcuts Help */}
          <div className="mt-3 p-3 bg-[#1a1a24]/30 rounded-xl border border-[#ffffff06]">
            <div className="flex items-center gap-2 text-xs text-gray-500 mb-2">
              <Keyboard className="w-3 h-3" />
              <span className="font-medium">Keyboard Shortcuts</span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-500">
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-[#ffffff10] rounded text-gray-400">Ctrl</kbd>
                <span>+</span>
                <kbd className="px-1.5 py-0.5 bg-[#ffffff10] rounded text-gray-400">↵</kbd>
                <span className="ml-1">Send</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-[#ffffff10] rounded text-gray-400">Esc</kbd>
                <span className="ml-1">Clear</span>
              </div>
            </div>
          </div>

          <div className="mt-4 text-center text-xs text-gray-600">
            <p>Built by <a href="https://sanshrit-singhai.vercel.app" className="text-[#667eea] hover:underline" target="_blank" rel="noopener noreferrer">Sanshrit Singhai</a></p>
          </div>
        </div>
      </aside>
    </>
  );
}
