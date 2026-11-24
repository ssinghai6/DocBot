"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import {
  X, CircleCheck, CircleAlert,
  TriangleAlert, Info, PanelRight,
} from "lucide-react"

import Sidebar from "@/components/Sidebar"
import ChatArea from "@/components/ChatArea"
import InspectorPanel from "@/components/InspectorPanel"
import AuthModal from "@/components/AuthModal"
import OutOfCreditsModal from "@/components/OutOfCreditsModal"
import AdminPanel from "@/components/AdminPanel"
import CommandPalette, { useCommandPalette, buildCommands } from "@/components/CommandPalette"
import ToolPicker, { useToolPicker } from "@/components/ToolPicker"
import { useChatHandlers } from "@/hooks/useChatHandlers"
import { useChatSubmit } from "@/hooks/useChatSubmit"
import { PanelGroup, Panel, ResizeHandle, type PanelImperativeHandle } from "@/components/ui"
import { useUIStore } from "@/store/uiStore"
import { useBreakpoint } from "@/lib/useBreakpoint"
import { resolveChatMode, createAnonymousSessionId, type ChatMode } from "@/lib/chatSession"
import { VerifyRequestSchema, VerifyResponseSchema } from "@/lib/schemas/emailVerify"
import { OwnerUnlockResponseSchema } from "@/lib/schemas/ownerUnlock"
import { gateResponse, onHttpGate } from "@/lib/httpGate"
import { z } from "zod"
import type { Paywall } from "@/components/OutOfCreditsModal"

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
  AdminMetrics,
  ApiKey,
  ApiKeyCreateResponse,
} from "@/components/types"

// Toast Component
function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`flex items-start gap-2.5 p-3 rounded-[5px] border bg-[var(--color-bg-elevated)] shadow-[var(--elev-3)] ${
            toast.type === "success" ? "border-[var(--color-success-500)]/40 text-[var(--color-success-500)]" :
            toast.type === "error" ? "border-[var(--color-danger-500)]/40 text-[var(--color-danger-500)]" :
            toast.type === "warning" ? "border-[var(--color-warning-500)]/40 text-[var(--color-warning-500)]" :
            "border-[var(--color-cyan-500)]/40 text-[var(--color-cyan-500)]"
          }`}
        >
          {toast.type === "success" && <CircleCheck className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "error" && <CircleAlert className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "warning" && <TriangleAlert className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "info" && <Info className="w-5 h-5 shrink-0 mt-0.5" />}
          <p className="text-[12px] font-medium flex-1 text-[var(--color-text-primary)]">{toast.message}</p>
          <button onClick={() => onDismiss(toast.id)} className="shrink-0 hover:opacity-70 transition-opacity">
            <X className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}

type DemoDataset = "quickbite" | "fuel" | "tieout";
const DEMO_DATASETS: DemoDataset[] = ["quickbite", "fuel", "tieout"];
const DemoInitSchema = z.object({ session_id: z.string(), connection_id: z.string() });

const DEMO_UI: Record<DemoDataset, { db: string; doc: string; persona: string; toast: string }> = {
  quickbite: {
    db: "QuickBite-Financials.db",
    doc: "QuickBite-10K-2025.pdf",
    persona: "Finance Expert",
    toast: "Demo loaded — QuickBite 10-K + financial database ready",
  },
  fuel: {
    db: "FuelPrices-1970-2026.db",
    doc: "fuelprice.pdf",
    persona: "Data Analyst",
    toast: "Demo loaded — fuel-price report + 1970–2026 crude oil data ready",
  },
  tieout: {
    db: "Tidewell-TrialBalance-FY2025.db",
    doc: "Tidewell-Draft-10K-FY2025.pdf",
    persona: "Finance Expert",
    toast: "Demo loaded — draft 10-K + trial balance ready to tie out",
  },
};

export default function Home() {
  // UI store (cross-cutting UI state)
  const inspectorOpen = useUIStore((s) => s.inspectorOpen);
  const setInspectorOpen = useUIStore((s) => s.setInspectorOpen);
  const selectedLineage = useUIStore((s) => s.selectedLineage);
  const bp = useBreakpoint();

  // Auto-collapse inspector on narrower screens
  useEffect(() => {
    if (bp === "xs" || bp === "sm" || bp === "md") {
      setInspectorOpen(false);
    }
  }, [bp, setInspectorOpen]);

  // DOCBOT-1520: badge the collapsed Inspector rail/toggle when new lineage
  // (citations/SQL/discrepancies) arrives while the Inspector is collapsed.
  // Derived during render: remember the last lineage seen with the Inspector open
  // (state adjusted during render, not in an effect, to avoid a cascading re-render).
  const [seenLineage, setSeenLineage] = useState(selectedLineage);
  if (inspectorOpen && seenLineage !== selectedLineage) {
    setSeenLineage(selectedLineage);
  }
  const hasUnviewedLineage =
    !inspectorOpen && selectedLineage !== null && selectedLineage !== seenLineage;

  // Imperative handle for the inspector panel — drives smooth collapse/expand
  const inspectorPanelRef = useRef<PanelImperativeHandle>(null);
  useEffect(() => {
    const handle = inspectorPanelRef.current;
    if (!handle) return;
    if (inspectorOpen && handle.isCollapsed()) handle.expand();
    if (!inspectorOpen && !handle.isCollapsed()) handle.collapse();
  }, [inspectorOpen]);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  // Sidebar State
  const [selectedPersona, setSelectedPersona] = useState("Generalist");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarTab, setSidebarTab] = useState<"sources" | "tools" | "settings">("sources");
  const [isAutoMode, setIsAutoMode] = useState(true);

  // Demo mode
  const [demoLoading, setDemoLoading] = useState(false);

  // Command palette
  const cmdPalette = useCommandPalette();

  // DOCBOT-1514: explicit tool picker (additive power-user feature)
  const toolPicker = useToolPicker();

  // Database connection state
  const [isDbConnected, setIsDbConnected] = useState(false);
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [dbUploadState, setDbUploadState] = useState<"idle" | "uploading" | "connected" | "error">("idle");
  const [dbFileName, setDbFileName] = useState<string | null>(null);
  const [isCsvConnection, setIsCsvConnection] = useState(false);

  // DOCBOT-1528: schema sync freshness — "Schema last synced" + refresh button
  const [schemaCachedAt, setSchemaCachedAt] = useState<string | null>(null);
  const [schemaRefreshing, setSchemaRefreshing] = useState(false);

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

  // Autopilot
  const [autopilotMode, setAutopilotMode] = useState(false);
  const [autopilotRunning, setAutopilotRunning] = useState(false);
  const [autopilotSteps, setAutopilotSteps] = useState<AutopilotStep[]>([]);
  const [autopilotPlan, setAutopilotPlan] = useState<string[]>([]);

  // Chat mode
  const [chatMode, setChatMode] = useState<ChatMode>("docs");

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
  // DOCBOT-1713: Turnstile bot-check on signup (widget only when the
  // NEXT_PUBLIC_TURNSTILE_SITE_KEY is set; token verified server-side).
  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY ?? "";
  const [turnstileToken, setTurnstileToken] = useState("");
  const [turnstileResetSignal, setTurnstileResetSignal] = useState(0);
  const bumpTurnstileReset = useCallback(() => setTurnstileResetSignal((n) => n + 1), []);

  // Persistent Workspace state
  const [workspaceConnections, setWorkspaceConnections] = useState<WorkspaceConnection[]>([]);

  // Marketplace connector state
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);

  // Admin panel state
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const [adminTab, setAdminTab] = useState<"users" | "audit" | "usage" | "keys">("users");
  const [adminUsers, setAdminUsers] = useState<AdminUser[]>([]);
  const [adminUsersLoading, setAdminUsersLoading] = useState(false);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditEventTypeFilter, setAuditEventTypeFilter] = useState<string>("all");
  // DOCBOT-1528: audit log filters — user + date range
  const [auditUserIdFilter, setAuditUserIdFilter] = useState<string>("all");
  const [auditStartDate, setAuditStartDate] = useState<string>("");
  const [auditEndDate, setAuditEndDate] = useState<string>("");
  // DOCBOT-1528: Usage tab — GET /admin/metrics
  const [adminMetrics, setAdminMetrics] = useState<AdminMetrics | null>(null);
  const [adminMetricsLoading, setAdminMetricsLoading] = useState(false);
  // DOCBOT-1530: Keys tab — developer API key management
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [apiKeysLoading, setApiKeysLoading] = useState(false);
  const [newlyCreatedApiKey, setNewlyCreatedApiKey] = useState<ApiKeyCreateResponse | null>(null);

  // Stable anonymous session ID used when no PDF session exists yet.
  // Lazy initializer runs once on mount, so the impure ID generator never runs during re-renders.
  const [anonymousSessionId] = useState(createAnonymousSessionId);

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
  // `sticky` toasts stay until closed (DOCBOT-2305: platform-capacity notice).
  const showToast = useCallback((type: Toast['type'], message: string, sticky = false) => {
    const id = Math.random().toString(36).substring(7);
    setToasts(prev => [...prev, { id, type, message }]);
    if (sticky) return;
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 5000);
  }, []);

  // DOCBOT-2305: one place mapping refused API calls to UI, for every fetch site.
  const [gatePaywall, setGatePaywall] = useState<Paywall | null>(null);
  useEffect(() => onHttpGate((gate) => {
    if (gate.kind === "signin") {
      setAuthModalTab("login");
      setAuthError(gate.message);
      setAuthModalOpen(true);
    } else if (gate.kind === "rate") {
      showToast("warning", gate.message);
    } else if (gate.kind === "credits") {
      setGatePaywall({ kind: "credits", payload: gate.payload });
    } else if (gate.kind === "capacity") {
      showToast("error", gate.message, true);
    }
  }), [showToast]);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // Owner unlock (DOCBOT-1605): if the page is opened with ?owner_key=<key>,
  // read it once, strip it from the address bar immediately, then POST it in
  // the X-Owner-Key header. The backend answers with an HttpOnly cookie holding
  // an HMAC of the key, which bypasses usage guards for this browser.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const ownerKey = params.get("owner_key");
    if (!ownerKey) return;
    params.delete("owner_key");
    const clean =
      window.location.pathname + (params.toString() ? `?${params}` : "");
    window.history.replaceState({}, "", clean);
    (async () => {
      try {
        const res = await fetch("/api/auth/owner-unlock", {
          method: "POST",
          headers: { "X-Owner-Key": ownerKey },
        });
        const parsed = OwnerUnlockResponseSchema.safeParse(
          await res.json().catch(() => null)
        );
        if (res.ok && parsed.success && parsed.data.owner) {
          showToast("success", "Owner mode unlocked — no usage limits");
        } else {
          showToast("error", "Invalid owner key");
        }
      } catch {
        showToast("error", "Owner unlock failed");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Hooks ──────────────────────────────────────────────────────────────────

  const handlers = useChatHandlers({
    sessionId,
    anonymousSessionId,
    authModalTab,
    authEmail,
    authPassword,
    authName,
    liveDbForm,
    entraToken,
    fileInputRef,
    turnstileToken,
    bumpTurnstileReset,

    setMessages,
    setInput,
    setIsLoading,
    setUploadedFiles,
    setSessionId,
    setUploadProgress,
    setSelectedPersona,
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
    setAdminMetrics,
    setAdminMetricsLoading,
    setApiKeys,
    setApiKeysLoading,
    setNewlyCreatedApiKey,
    setWorkspaceConnections,
    setConnectors,
    setSchemaCachedAt,
    setSchemaRefreshing,

    showToast,
  });

  // Demo mode handler — dataset: "quickbite" (discrepancy), "fuel" (forecasting),
  // or "tieout" (draft 10-K vs trial balance, DOCBOT-2212). DOCBOT-2632:
  // `keepMessages` restarts an expired demo without wiping the chat.
  const handleTryDemo = useCallback(async (
    dataset: DemoDataset = "quickbite",
    opts?: { keepMessages?: boolean },
  ): Promise<{ sessionId: string; connectionId: string } | null> => {
    setDemoLoading(true);
    try {
      const res = await fetch(`/api/demo/init?dataset=${dataset}`, { method: "POST" });
      // DOCBOT-1711 / 2305: sign-in, rate, credits or capacity gate → shown by the gate listener.
      if (await gateResponse(res)) return null;
      if (!res.ok) throw new Error("Demo initialization failed");
      const data = DemoInitSchema.parse(await res.json());
      const demo = DEMO_UI[dataset];
      setSessionId(data.session_id);
      setConnectionId(data.connection_id);
      setIsDbConnected(true);
      setDbFileName(demo.db);
      setChatMode("hybrid");
      setSelectedPersona(demo.persona);
      setIsAutoMode(false);
      setUploadedFiles([new File([], demo.doc)]);
      if (!opts?.keepMessages) {
        setMessages([]);
        showToast("success", demo.toast);
      }
      return { sessionId: data.session_id, connectionId: data.connection_id };
    } catch {
      showToast("error", "Failed to load demo. Please try again.");
      return null;
    } finally {
      setDemoLoading(false);
    }
  }, [showToast]);

  // DOCBOT-2632: an expired demo restarts on the dataset it was showing.
  const restartDemo = () => handleTryDemo(
    DEMO_DATASETS.find(d => DEMO_UI[d].db === dbFileName) ?? "quickbite",
    { keepMessages: true },
  );

  const { handleSendMessage, paywall, dismissPaywall } = useChatSubmit({
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
    messages,
    pickedTool: toolPicker.pickedTool,
    clearPickedTool: toolPicker.clearPickedTool,

    setMessages,
    setInput,
    setIsLoading,
    setChatMode,
    setAutopilotRunning,
    setAutopilotSteps,
    setAutopilotPlan,

    showToast,
    loadQueryHistory: handlers.loadQueryHistory,
    restartDemo,
  });

  const shownPaywall = paywall ?? gatePaywall;

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
          // DOCBOT-1713: emailed verification link lands here with
          // ?verify_token=... — POST it once, toast the result, then clean
          // the URL so a refresh/revisit can't replay the link.
          const verifyToken = params.get("verify_token");
          if (verifyToken) {
            try {
              const parse = VerifyRequestSchema.safeParse({ token: verifyToken });
              const vRes = await fetch("/api/auth/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(parse.success ? parse.data : { token: verifyToken }),
              });
              const vData = await vRes.json();
              const vParsed = VerifyResponseSchema.safeParse(vData);
              if (vRes.ok && vParsed.success) {
                showToast("success", `Email verified for ${vParsed.data.email} — sign in to continue.`);
              } else {
                showToast(
                  "error",
                  (vData && typeof vData.detail === "string" ? vData.detail : "")
                    || "That verification link is invalid or has expired. Sign in to get a new one.",
                );
              }
            } catch {
              showToast("error", "Verification failed — please sign in and request a new link.");
            }
            window.history.replaceState({}, "", window.location.pathname);
            setAuthChecked(true);
            return;
          }
          // DOCBOT-1805: back from Stripe (Checkout success/cancel URL or the
          // Payment Link's after-payment redirect). No balance endpoint yet.
          const checkout = params.get("checkout");
          if (checkout) {
            if (checkout === "success") showToast("success", "Payment received — your credits are being added.");
            else showToast("info", "Checkout cancelled — you weren't charged.");
            window.history.replaceState({}, "", window.location.pathname);
          }
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
          } else if (params.has("login")) {
            // DOCBOT-1712: /signup, /verify link here for existing users.
            if (!res.ok) {
              setAuthModalTab("login");
              setAuthModalOpen(true);
            }
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
    else if (adminTab === "audit") {
      handlers.loadAuditLog({
        eventType: auditEventTypeFilter,
        userId: auditUserIdFilter,
        startDate: auditStartDate,
        endDate: auditEndDate,
      });
    } else if (adminTab === "usage") {
      handlers.loadAdminMetrics();
    } else {
      handlers.loadApiKeys();
    }
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

  // Promote to hybrid when both sources exist; fall back when the active mode's
  // required source becomes unavailable. Adjusted during render (guarded, and
  // resolveChatMode is idempotent) instead of in an effect, so there is no
  // extra commit with a stale mode.
  const resolvedChatMode = resolveChatMode(chatMode, connectionId, sessionId);
  if (resolvedChatMode !== chatMode) {
    setChatMode(resolvedChatMode);
  }

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
      // Cmd/Ctrl+I toggles inspector
      if ((e.ctrlKey || e.metaKey) && (e.key === 'i' || e.key === 'I')) {
        e.preventDefault();
        setInspectorOpen(!inspectorOpen);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [input, sessionId, isDbConnected, isLoading, inspectorOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen relative z-10 text-[var(--color-text-secondary)] overflow-hidden bg-[var(--color-bg-base)]">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        activeTab={sidebarTab}
        setActiveTab={setSidebarTab}
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
        connectionId={connectionId}
        schemaCachedAt={schemaCachedAt}
        schemaRefreshing={schemaRefreshing}
        onSchemaRefresh={handlers.refreshSchema}
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
        isAutoMode={isAutoMode}
        onSelectPersona={setSelectedPersona}
        onSetAutoMode={setIsAutoMode}
        clearChat={handlers.clearChat}
        messagesLength={messages.length}
        connectors={connectors}
        onConnectorRegister={handlers.handleConnectorRegister}
        onConnectorSync={handlers.handleConnectorSync}
        onConnectorDisconnect={handlers.handleConnectorDisconnect}
        onEdgarFilingIngested={(sid: string, label: string) => {
          setSessionId(sid)
          setSelectedPersona("Finance Expert")
          setMessages([])
          showToast("success", `Ready to chat with ${label}`)
        }}
      />

      <PanelGroup
        orientation="horizontal"
        id="docbot-main-panels"
        className="flex-1 min-w-0"
      >
      <Panel id="main" defaultSize={72} minSize={40}>
      <ChatArea
        sessionId={sessionId}
        uploadedFiles={uploadedFiles}
        selectedPersona={selectedPersona}
        setSelectedPersona={setSelectedPersona}
        isAutoMode={isAutoMode}
        setAutoMode={setIsAutoMode}
        isDbConnected={isDbConnected}
        connectionId={connectionId}
        dbFileName={dbFileName}
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
        chatContainerRef={chatContainerRef}
        lastMessageRef={lastMessageRef}
        messagesEndRef={messagesEndRef}
        textareaRef={textareaRef}
        handleSendMessage={handleSendMessage}
        clearSession={handlers.clearSession}
        exportChat={handlers.exportChat}
        showToast={showToast}
        onUploadClick={() => { setSidebarOpen(true); setSidebarTab("sources"); fileInputRef.current?.click(); }}
        onConnectDatabase={() => { setSidebarOpen(true); setSidebarTab("sources"); setShowLiveDbForm(true); }}
        onBrowseEdgar={() => { setSidebarOpen(true); setSidebarTab("tools"); }}
        onTryDemo={handleTryDemo}
        demoLoading={demoLoading}
        pickedTool={toolPicker.pickedTool}
        onOpenToolPicker={() => toolPicker.setIsOpen(true)}
        onClearPickedTool={toolPicker.clearPickedTool}
        inspectorOpen={inspectorOpen}
        onToggleInspector={() => setInspectorOpen(!inspectorOpen)}
        hasUnviewedLineage={hasUnviewedLineage}
      />
      </Panel>

      <ResizeHandle />
      <Panel
        id="inspector"
        panelRef={inspectorPanelRef}
        defaultSize="380px"
        minSize="280px"
        maxSize="640px"
        collapsible
        collapsedSize="36px"
        onResize={(size) => {
          const pct = typeof size === "number" ? size : size.asPercentage
          const isCollapsed = pct < 1
          if (isCollapsed && inspectorOpen) setInspectorOpen(false)
          if (!isCollapsed && !inspectorOpen) setInspectorOpen(true)
        }}
        className="min-w-0"
      >
        {inspectorOpen ? (
          <InspectorPanel onClose={() => setInspectorOpen(false)} />
        ) : (
          // DOCBOT-1520: thin, always-visible collapse rail (not 0px/invisible) —
          // a single click re-expands the Inspector. Narrow viewports still get
          // the chat width back (36px vs. the old 0px is a negligible trade).
          <button
            onClick={() => setInspectorOpen(true)}
            aria-label="Open inspector"
            title="Inspector (⌘I)"
            className="relative h-full w-full flex flex-col items-center justify-start pt-3 bg-[var(--color-bg-surface)] border-l border-[var(--color-border-subtle)] hover:bg-[var(--color-bg-elevated)] transition-colors"
          >
            <PanelRight className="w-4 h-4 text-[var(--color-text-tertiary)]" />
            {hasUnviewedLineage && (
              <span
                className="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-[var(--color-cyan-500)]"
                aria-hidden="true"
              />
            )}
          </button>
        )}
      </Panel>
      </PanelGroup>

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
          turnstileSiteKey={turnstileSiteKey}
          turnstileToken={turnstileToken}
          setTurnstileToken={setTurnstileToken}
          turnstileResetSignal={turnstileResetSignal}
          onClose={() => { setAuthModalOpen(false); setAuthError(null); }}
        />
      )}

      {/* DOCBOT-1805: out-of-credits / rate-limit paywall */}
      {shownPaywall && (
        <OutOfCreditsModal
          paywall={shownPaywall}
          isSignedIn={authUser !== null}
          email={authUser?.email}
          onSignIn={() => { dismissPaywall(); setGatePaywall(null); setAuthModalTab("register"); setAuthModalOpen(true); }}
          onClose={() => { dismissPaywall(); setGatePaywall(null); }}
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
          auditUserIdFilter={auditUserIdFilter}
          setAuditUserIdFilter={setAuditUserIdFilter}
          auditStartDate={auditStartDate}
          setAuditStartDate={setAuditStartDate}
          auditEndDate={auditEndDate}
          setAuditEndDate={setAuditEndDate}
          loadAuditLog={handlers.loadAuditLog}
          exportAuditLogCsv={handlers.exportAuditLogCsv}
          updateUserRole={handlers.updateUserRole}
          adminMetrics={adminMetrics}
          adminMetricsLoading={adminMetricsLoading}
          loadAdminMetrics={handlers.loadAdminMetrics}
          apiKeys={apiKeys}
          apiKeysLoading={apiKeysLoading}
          newlyCreatedApiKey={newlyCreatedApiKey}
          loadApiKeys={handlers.loadApiKeys}
          createApiKey={handlers.createApiKey}
          revokeApiKey={handlers.revokeApiKey}
          dismissNewlyCreatedApiKey={() => setNewlyCreatedApiKey(null)}
          authUser={authUser}
          showToast={showToast}
          onClose={() => setAdminPanelOpen(false)}
        />
      )}

      {/* Command Palette (Cmd+K) */}
      <CommandPalette
        isOpen={cmdPalette.isOpen}
        onClose={cmdPalette.onClose}
        commands={buildCommands({
          onConnectDatabase: () => { setSidebarOpen(true); setSidebarTab("sources"); setShowLiveDbForm(true); },
          onSearchEdgar: () => { setSidebarOpen(true); setSidebarTab("tools"); },
          onAddConnector: () => { setSidebarOpen(true); setSidebarTab("tools"); },
          onToggleInspector: () => setInspectorOpen(!inspectorOpen),
          onClearChat: messages.length > 0 ? handlers.clearChat : undefined,
          onExportChat: sessionId && messages.length > 0 ? () => handlers.exportChat("markdown") : undefined,
          onSwitchPersona: setSelectedPersona,
          onSetAutoMode: setIsAutoMode,
        })}
      />

      {/* Tool Picker (DOCBOT-1514) — explicit, additive power-user tool pick */}
      <ToolPicker
        isOpen={toolPicker.isOpen}
        onClose={toolPicker.onClose}
        onPick={toolPicker.setPickedTool}
      />
    </div>
  );
}
