"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import {
  X, CheckCircle2, AlertCircle,
  AlertTriangle, Info,
} from "lucide-react"

import Sidebar from "@/components/Sidebar"
import ChatArea from "@/components/ChatArea"
import AuthModal from "@/components/AuthModal"
import AdminPanel from "@/components/AdminPanel"
import { useChatHandlers } from "@/hooks/useChatHandlers"
import { useChatSubmit } from "@/hooks/useChatSubmit"

import {
  AuthMeSchema,
} from "@/components/types"
import type {
  AuthUser,
  AdminUser,
  AuditEvent,
  WorkspaceConnection,
  Toast,
  FileUploadState,
  QueryHistoryItem,
  AutopilotStep,
  Message,
  LiveDbForm,
  ConnectorInfo,
} from "@/components/types"

// Toast Component
function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`flex items-start gap-3 p-4 rounded-xl border backdrop-blur-xl shadow-lg animate-in slide-in-from-right duration-300 ${toast.type === "success" ? "bg-[#10b981]/10 border-[#10b981]/30 text-[#10b981]" :
            toast.type === "error" ? "bg-[#ef4444]/10 border-[#ef4444]/30 text-[#ef4444]" :
              toast.type === "warning" ? "bg-[#f59e0b]/10 border-[#f59e0b]/30 text-[#f59e0b]" :
                "bg-[#3b82f6]/10 border-[#3b82f6]/30 text-[#3b82f6]"
            }`}
        >
          {toast.type === "success" && <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "error" && <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "warning" && <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "info" && <Info className="w-5 h-5 shrink-0 mt-0.5" />}
          <p className="text-sm font-medium flex-1 text-gray-200">{toast.message}</p>
          <button onClick={() => onDismiss(toast.id)} className="shrink-0 hover:opacity-70 transition-opacity">
            <X className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  // Sidebar State
  const [selectedPersona, setSelectedPersona] = useState("Generalist");
  const [suggestedPersona, setSuggestedPersona] = useState<string | null>(null);
  const [deepVisualMode, setDeepVisualMode] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [drProgress, setDrProgress] = useState<{ step: string; message: string } | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isAutoMode, setIsAutoMode] = useState(true);

  // Database connection state
  const [isDbConnected, setIsDbConnected] = useState(false);
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [dbUploadState, setDbUploadState] = useState<"idle" | "uploading" | "connected" | "error">("idle");
  const [dbFileName, setDbFileName] = useState<string | null>(null);
  const [isCsvConnection, setIsCsvConnection] = useState(false);

  // Live DB connection form state
  const [showLiveDbForm, setShowLiveDbForm] = useState(false);
  const [liveDbForm, setLiveDbForm] = useState<LiveDbForm>({
    dialect: "postgresql",
    host: "",
    port: "5432",
    dbname: "",
    user: "",
    password: "",
    pii_masking_enabled: false,
  });
  const [showDbPassword, setShowDbPassword] = useState(false);
  const [liveDbConnectState, setLiveDbConnectState] = useState<"idle" | "connecting" | "error">("idle");

  // Entra interactive auth state
  const [entraToken, setEntraToken] = useState<string | null>(null);
  const [entraEmail, setEntraEmail] = useState<string | null>(null);
  const [entraSignInState, setEntraSignInState] = useState<"idle" | "signing_in" | "signed_in" | "error">("idle");
  const [liveDbError, setLiveDbError] = useState<string | null>(null);

  // Query history panel
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null);

  // Advanced charts
  const [chartType, setChartType] = useState<string>("auto");
  const [zoomedChart, setZoomedChart] = useState<string | null>(null);

  // Analytical Autopilot
  const [autopilotMode, setAutopilotMode] = useState(false);
  const [autopilotRunning, setAutopilotRunning] = useState(false);
  const [autopilotSteps, setAutopilotSteps] = useState<AutopilotStep[]>([]);
  const [autopilotPlan, setAutopilotPlan] = useState<string[]>([]);

  // Chat mode
  const [chatMode, setChatMode] = useState<"docs" | "database" | "hybrid">("docs");

  // Auth state
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [samlConfigured, setSamlConfigured] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [authConfig, setAuthConfig] = useState<{ email: boolean; github: boolean; google: boolean; saml: boolean } | null>(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalTab, setAuthModalTab] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authName, setAuthName] = useState("");
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  // Persistent Workspace state
  const [workspaceConnections, setWorkspaceConnections] = useState<WorkspaceConnection[]>([]);

  // Marketplace connector state
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);

  // Admin panel state
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const [adminTab, setAdminTab] = useState<"users" | "audit">("users");
  const [adminUsers, setAdminUsers] = useState<AdminUser[]>([]);
  const [adminUsersLoading, setAdminUsersLoading] = useState(false);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditEventTypeFilter, setAuditEventTypeFilter] = useState<string>("all");

  // Stable anonymous session ID used when no PDF session exists yet
  const anonymousSessionIdRef = useRef<string>(
    typeof crypto !== "undefined" ? crypto.randomUUID() : Math.random().toString(36).substring(2)
  );

  // File Upload State
  const [fileUploadState, setFileUploadState] = useState<FileUploadState>("idle");

  // Toast State
  const [toasts, setToasts] = useState<Toast[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const lastMessageRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Toast helper
  const showToast = useCallback((type: Toast['type'], message: string) => {
    const id = Math.random().toString(36).substring(7);
    setToasts(prev => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 5000);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // ── Hooks ──────────────────────────────────────────────────────────────────

  const handlers = useChatHandlers({
    sessionId,
    anonymousSessionIdRef,
    deepVisualMode,
    authModalTab,
    authEmail,
    authPassword,
    authName,
    auditEventTypeFilter,
    liveDbForm,
    entraToken,
    fileInputRef,

    setMessages,
    setInput,
    setIsLoading,
    setUploadedFiles,
    setSessionId,
    setUploadProgress,
    setSelectedPersona,
    setSuggestedPersona,
    setIsDbConnected,
    setConnectionId,
    setDbUploadState,
    setDbFileName,
    setIsCsvConnection,
    setShowLiveDbForm,
    setLiveDbConnectState,
    setLiveDbError,
    setChatMode,
    setQueryHistory,
    setHistoryOpen,
    setAutopilotMode,
    setAutopilotSteps,
    setAutopilotPlan,
    setFileUploadState,
    setShowDbPassword,
    setEntraToken,
    setEntraEmail,
    setEntraSignInState,
    setAuthUser,
    setAuthSubmitting,
    setAuthError,
    setAuthModalOpen,
    setAuthEmail,
    setAuthPassword,
    setAuthName,
    setAdminUsersLoading,
    setAdminUsers,
    setAuditLoading,
    setAuditEvents,
    setWorkspaceConnections,
    setConnectors,

    showToast,
  });

  const { handleSendMessage } = useChatSubmit({
    input,
    sessionId,
    isDbConnected,
    connectionId,
    selectedPersona,
    isAutoMode,
    chatMode,
    autopilotMode,
    isCsvConnection,
    chartType,
    deepResearch,
    messages,

    setMessages,
    setInput,
    setIsLoading,
    setChatMode,
    setAutopilotRunning,
    setAutopilotSteps,
    setAutopilotPlan,
    setDrProgress,

    showToast,
    loadQueryHistory: handlers.loadQueryHistory,
  });

  // ── Effects ────────────────────────────────────────────────────────────────

  // Check auth on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const configRes = await fetch("/api/auth/config");
        if (configRes.ok) {
          const cfg = await configRes.json();
          setAuthConfig(cfg);
          setSamlConfigured(cfg.saml === true);
        }

        const res = await fetch("/api/auth/me", { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          const parsed = AuthMeSchema.safeParse(data);
          if (parsed.success) {
            setAuthUser(parsed.data);
            handlers.fetchWorkspace();
            handlers.loadConnectors();
          }
        }

        if (typeof window !== "undefined") {
          const params = new URLSearchParams(window.location.search);
          if (params.get("auth_success")) {
            const meRes = await fetch("/api/auth/me", { credentials: "include" });
            if (meRes.ok) {
              const data = await meRes.json();
              const parsed = AuthMeSchema.safeParse(data);
              if (parsed.success) {
                setAuthUser(parsed.data);
                showToast("success", `Welcome, ${parsed.data.name || parsed.data.email}!`);
                handlers.fetchWorkspace();
              }
            }
            window.history.replaceState({}, "", window.location.pathname);
          } else if (params.get("auth_error")) {
            const errorMap: Record<string, string> = {
              invalid_state: "Authentication session expired. Please try again.",
              github_failed: "GitHub sign-in failed. Please try again.",
              google_failed: "Google sign-in failed. Please try again.",
              provision_failed: "Account setup failed. Please try again.",
            };
            const msg = errorMap[params.get("auth_error")!] ?? "Sign-in failed. Please try again.";
            showToast("error", msg);
            window.history.replaceState({}, "", window.location.pathname);
          }
        }
      } catch {
        // network error — stay in open mode
      } finally {
        setAuthChecked(true);
      }
    };
    checkAuth();
  }, [handlers.fetchWorkspace]); // eslint-disable-line react-hooks/exhaustive-deps

  // When admin panel opens, load data for the active tab
  useEffect(() => {
    if (!adminPanelOpen) return;
    if (adminTab === "users") handlers.loadAdminUsers();
    else handlers.loadAuditLog(auditEventTypeFilter);
  }, [adminPanelOpen, adminTab]); // eslint-disable-line react-hooks/exhaustive-deps

  const scrollToBottom = () => {
    if (lastMessageRef.current && chatContainerRef.current) {
      const container = chatContainerRef.current;
      const msgRect = lastMessageRef.current.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const scrollOffset = msgRect.top - containerRect.top + container.scrollTop - 16;
      container.scrollTo({ top: scrollOffset, behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [input]);

  // Fall back to "docs" mode when the active mode's required source becomes unavailable
  useEffect(() => {
    if (chatMode === "database" && !connectionId) {
      setChatMode("docs");
    } else if (chatMode === "hybrid" && (!connectionId || !sessionId)) {
      setChatMode("docs");
    }
  }, [connectionId, sessionId, chatMode]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (input.trim() && (sessionId || isDbConnected) && !isLoading) {
          handleSendMessage();
        }
      }
      if (e.key === 'Escape') {
        setInput("");
        textareaRef.current?.blur();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [input, sessionId, isDbConnected, isLoading]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen relative z-10 text-[#e0e0e0] overflow-hidden bg-[#0a0a0f]">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        authChecked={authChecked}
        authUser={authUser}
        handleLogout={handlers.handleLogout}
        setAdminPanelOpen={setAdminPanelOpen}
        setAuthModalOpen={setAuthModalOpen}
        setAuthError={setAuthError}
        workspaceConnections={workspaceConnections}
        setShowLiveDbForm={setShowLiveDbForm}
        setLiveDbForm={setLiveDbForm}
        showToast={showToast}
        fileUploadState={fileUploadState}
        uploadProgress={uploadProgress}
        uploadedFiles={uploadedFiles}
        deepVisualMode={deepVisualMode}
        onDeepVisualModeChange={setDeepVisualMode}
        onDragOver={handlers.handleDragOver}
        onDragLeave={handlers.handleDragLeave}
        onDrop={handlers.handleDrop}
        onFileChange={handlers.handleFileUpload}
        fileInputRef={fileInputRef}
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
        hasDocSession={!!sessionId}
        queryHistory={queryHistory}
        historyOpen={historyOpen}
        expandedHistoryId={expandedHistoryId}
        onDbUpload={handlers.handleDbUpload}
        onDbDisconnect={handlers.handleDbDisconnect}
        onToggleLiveDbForm={() => { setShowLiveDbForm(v => !v); setLiveDbError(null); }}
        onLiveDbFormChange={setLiveDbForm}
        onShowDbPasswordChange={setShowDbPassword}
        onLiveDbConnect={handlers.handleLiveDbConnect}
        onAutopilotToggle={() => setAutopilotMode(v => !v)}
        onHistoryToggle={() => setHistoryOpen(v => !v)}
        onExpandedHistoryChange={setExpandedHistoryId}
        onSetInput={setInput}
        onMicrosoftSignIn={handlers.handleMicrosoftSignIn}
        onEntraReset={() => { setEntraToken(null); setEntraEmail(null); setEntraSignInState("idle"); }}
        selectedPersona={selectedPersona}
        suggestedPersona={suggestedPersona}
        isAutoMode={isAutoMode}
        deepResearch={deepResearch}
        onSelectPersona={setSelectedPersona}
        onSetAutoMode={setIsAutoMode}
        onDeepResearchChange={setDeepResearch}
        clearChat={handlers.clearChat}
        messagesLength={messages.length}
        connectors={connectors}
        onConnectorRegister={handlers.handleConnectorRegister}
        onConnectorSync={handlers.handleConnectorSync}
        onConnectorDisconnect={handlers.handleConnectorDisconnect}
      />

      <ChatArea
        sessionId={sessionId}
        uploadedFiles={uploadedFiles}
        selectedPersona={selectedPersona}
        isDbConnected={isDbConnected}
        connectionId={connectionId}
        chatMode={chatMode}
        setChatMode={setChatMode}
        messages={messages}
        isLoading={isLoading}
        input={input}
        setInput={setInput}
        chartType={chartType}
        setChartType={setChartType}
        autopilotMode={autopilotMode}
        setAutopilotMode={setAutopilotMode}
        autopilotRunning={autopilotRunning}
        autopilotSteps={autopilotSteps}
        autopilotPlan={autopilotPlan}
        deepResearch={deepResearch}
        drProgress={drProgress}
        chatContainerRef={chatContainerRef}
        lastMessageRef={lastMessageRef}
        messagesEndRef={messagesEndRef}
        textareaRef={textareaRef}
        handleSendMessage={handleSendMessage}
        clearSession={handlers.clearSession}
        exportChat={handlers.exportChat}
        showToast={showToast}
      />

      {/* Auth Modal */}
      {authModalOpen && !authUser && (
        <AuthModal
          authConfig={authConfig}
          authModalTab={authModalTab}
          setAuthModalTab={setAuthModalTab}
          authEmail={authEmail}
          setAuthEmail={setAuthEmail}
          authPassword={authPassword}
          setAuthPassword={setAuthPassword}
          authName={authName}
          setAuthName={setAuthName}
          authSubmitting={authSubmitting}
          authError={authError}
          setAuthError={setAuthError}
          handleEmailAuth={handlers.handleEmailAuth}
          onClose={() => setAuthModalOpen(false)}
        />
      )}

      {/* Admin Panel Modal */}
      {adminPanelOpen && (
        <AdminPanel
          adminTab={adminTab}
          setAdminTab={setAdminTab}
          adminUsers={adminUsers}
          adminUsersLoading={adminUsersLoading}
          loadAdminUsers={handlers.loadAdminUsers}
          auditEvents={auditEvents}
          auditLoading={auditLoading}
          auditEventTypeFilter={auditEventTypeFilter}
          setAuditEventTypeFilter={setAuditEventTypeFilter}
          loadAuditLog={handlers.loadAuditLog}
          exportAuditLogCsv={handlers.exportAuditLogCsv}
          updateUserRole={handlers.updateUserRole}
          authUser={authUser}
          showToast={showToast}
          onClose={() => setAdminPanelOpen(false)}
        />
      )}
    </div>
  );
}
