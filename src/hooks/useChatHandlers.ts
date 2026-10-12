"use client"

import { useCallback } from "react"
import {
  AuthMeSchema,
  AdminUsersResponseSchema,
  AuditLogResponseSchema,
  WorkspaceSchema,
  DbSchemaResponseSchema,
  AdminMetricsSchema,
} from "@/components/types"
import type {
  AuthUser,
  AdminUser,
  AuditEvent,
  FileUploadState,
  WorkspaceConnection,
  Toast,
  LiveDbForm,
  QueryHistoryItem,
  ConnectorInfo,
  ConnectorSyncResponse,
  AdminMetrics,
} from "@/components/types"

// DOCBOT-1528: shared shape for GET /admin/audit-log's filter query params.
export interface AuditLogFilters {
  eventType?: string
  userId?: string
  startDate?: string
  endDate?: string
}

interface UseChatHandlersParams {
  sessionId: string | null
  anonymousSessionIdRef: React.RefObject<string>
  authModalTab: "login" | "register"
  authEmail: string
  authPassword: string
  authName: string
  liveDbForm: LiveDbForm
  entraToken: string | null
  fileInputRef: React.RefObject<HTMLInputElement | null>

  setMessages: React.Dispatch<React.SetStateAction<import("@/components/types").Message[]>>
  setInput: React.Dispatch<React.SetStateAction<string>>
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>
  setUploadedFiles: React.Dispatch<React.SetStateAction<File[]>>
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>
  setUploadProgress: React.Dispatch<React.SetStateAction<number | null>>
  setSelectedPersona: React.Dispatch<React.SetStateAction<string>>
  setIsDbConnected: React.Dispatch<React.SetStateAction<boolean>>
  setConnectionId: React.Dispatch<React.SetStateAction<string | null>>
  setDbUploadState: React.Dispatch<React.SetStateAction<"idle" | "uploading" | "connected" | "error">>
  setDbFileName: React.Dispatch<React.SetStateAction<string | null>>
  setIsCsvConnection: React.Dispatch<React.SetStateAction<boolean>>
  setShowLiveDbForm: React.Dispatch<React.SetStateAction<boolean>>
  setLiveDbConnectState: React.Dispatch<React.SetStateAction<"idle" | "connecting" | "error">>
  setLiveDbError: React.Dispatch<React.SetStateAction<string | null>>
  setChatMode: React.Dispatch<React.SetStateAction<"docs" | "database" | "hybrid">>
  setQueryHistory: React.Dispatch<React.SetStateAction<QueryHistoryItem[]>>
  setHistoryOpen: React.Dispatch<React.SetStateAction<boolean>>
  setAutopilotMode: React.Dispatch<React.SetStateAction<boolean>>
  setAutopilotSteps: React.Dispatch<React.SetStateAction<import("@/components/types").AutopilotStep[]>>
  setAutopilotPlan: React.Dispatch<React.SetStateAction<string[]>>
  setFileUploadState: React.Dispatch<React.SetStateAction<FileUploadState>>
  setShowDbPassword: React.Dispatch<React.SetStateAction<boolean>>
  setEntraToken: React.Dispatch<React.SetStateAction<string | null>>
  setEntraEmail: React.Dispatch<React.SetStateAction<string | null>>
  setEntraSignInState: React.Dispatch<React.SetStateAction<"idle" | "signing_in" | "signed_in" | "error">>
  setAuthUser: React.Dispatch<React.SetStateAction<AuthUser | null>>
  setAuthSubmitting: React.Dispatch<React.SetStateAction<boolean>>
  setAuthError: React.Dispatch<React.SetStateAction<string | null>>
  setAuthModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  setAuthEmail: React.Dispatch<React.SetStateAction<string>>
  setAuthPassword: React.Dispatch<React.SetStateAction<string>>
  setAuthName: React.Dispatch<React.SetStateAction<string>>
  setAdminUsersLoading: React.Dispatch<React.SetStateAction<boolean>>
  setAdminUsers: React.Dispatch<React.SetStateAction<AdminUser[]>>
  setAuditLoading: React.Dispatch<React.SetStateAction<boolean>>
  setAuditEvents: React.Dispatch<React.SetStateAction<AuditEvent[]>>
  setAdminMetrics: React.Dispatch<React.SetStateAction<AdminMetrics | null>>
  setAdminMetricsLoading: React.Dispatch<React.SetStateAction<boolean>>
  setWorkspaceConnections: React.Dispatch<React.SetStateAction<WorkspaceConnection[]>>
  setConnectors: React.Dispatch<React.SetStateAction<ConnectorInfo[]>>
  setSchemaCachedAt: React.Dispatch<React.SetStateAction<string | null>>
  setSchemaRefreshing: React.Dispatch<React.SetStateAction<boolean>>

  showToast: (type: Toast['type'], message: string) => void
}

export function useChatHandlers(params: UseChatHandlersParams) {
  const {
    sessionId,
    anonymousSessionIdRef,
    authModalTab,
    authEmail,
    authPassword,
    authName,
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
    setWorkspaceConnections,
    setConnectors,
    setSchemaCachedAt,
    setSchemaRefreshing,

    showToast,
  } = params;

  const loadQueryHistory = useCallback(async (connId: string) => {
    try {
      const res = await fetch(`/api/db/history/${connId}?limit=20`);
      if (!res.ok) return;
      const data = await res.json();
      setQueryHistory(data.history ?? []);
    } catch {
      // non-fatal — history panel just stays empty
    }
  }, [setQueryHistory]);

  // DOCBOT-1528: schema sync freshness — wires the existing (previously
  // frontend-unused) GET /api/db/schema/{id} schema_cached_at field and
  // POST /api/db/refresh-schema/{id} endpoint into the Sources tab.
  const loadSchemaCacheInfo = useCallback(async (connId: string) => {
    try {
      const res = await fetch(`/api/db/schema/${connId}`);
      if (!res.ok) return;
      const data = await res.json();
      const parsed = DbSchemaResponseSchema.safeParse(data);
      if (parsed.success) setSchemaCachedAt(parsed.data.schema_cached_at ?? null);
    } catch {
      // non-fatal — "Schema last synced" line just stays hidden
    }
  }, [setSchemaCachedAt]);

  const refreshSchema = useCallback(async (connId: string) => {
    setSchemaRefreshing(true);
    try {
      const res = await fetch(`/api/db/refresh-schema/${connId}`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      await loadSchemaCacheInfo(connId);
      showToast("success", "Schema refreshed");
    } catch (err) {
      showToast("error", `Schema refresh failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSchemaRefreshing(false);
    }
  }, [loadSchemaCacheInfo, showToast, setSchemaRefreshing]);

  const handleDbConnected = useCallback((connId: string) => {
    setIsDbConnected(true);
    setConnectionId(connId);
    setChatMode(sessionId ? "hybrid" : "database");
    setSelectedPersona(prev => (prev === "Generalist" ? "Data Analyst" : prev));
    loadQueryHistory(connId);
    loadSchemaCacheInfo(connId);
  }, [loadQueryHistory, loadSchemaCacheInfo, sessionId, setIsDbConnected, setConnectionId, setChatMode, setSelectedPersona]);

  const handleDbDisconnect = useCallback(() => {
    setIsDbConnected(false);
    setConnectionId(null);
    setDbFileName(null);
    setIsCsvConnection(false);
    setDbUploadState("idle");
    setChatMode("docs");
    setSelectedPersona(prev => (prev === "Data Analyst" ? "Generalist" : prev));
    setShowLiveDbForm(false);
    setLiveDbConnectState("idle");
    setLiveDbError(null);
    setQueryHistory([]);
    setHistoryOpen(false);
    setAutopilotMode(false);
    setAutopilotSteps([]);
    setAutopilotPlan([]);
  }, [setIsDbConnected, setConnectionId, setDbFileName, setDbUploadState, setChatMode, setSelectedPersona, setShowLiveDbForm, setLiveDbConnectState, setLiveDbError, setQueryHistory, setHistoryOpen, setAutopilotMode, setAutopilotSteps, setAutopilotPlan]);

  // Sign-in gate: anonymous upload/DB-connect is refused by the backend — 401
  // when GATE_UPLOADS is on, or 403 when PUBLIC_DEMO_MODE hard-blocks it. Either
  // way, pop the login modal and prompt the visitor to sign in for full access
  // instead of showing a raw error.
  const promptLoginIfGated = (status: number): boolean => {
    if (status === 401 || status === 403) {
      setAuthModalOpen(true);
      showToast("info", "Sign in (free) to upload your own data");
      return true;
    }
    return false;
  };

  const handleDbUpload = async (file: File, type: "csv" | "sqlite") => {
    setDbUploadState("uploading");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId ?? anonymousSessionIdRef.current);
    try {
      const endpoint = type === "csv" ? "/api/db/upload/csv" : "/api/db/upload";
      const response = await fetch(endpoint, { method: "POST", body: formData });
      if (promptLoginIfGated(response.status)) {
        setDbUploadState("idle");
        return;
      }
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await response.json();
      setDbFileName(file.name);
      setIsCsvConnection(type === "csv");
      setDbUploadState("connected");
      handleDbConnected(data.connection_id);
      showToast("success", `Connected: ${file.name}`);
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Upload failed";
      setDbUploadState("error");
      showToast("error", `DB upload failed: ${msg}`);
      setTimeout(() => setDbUploadState("idle"), 3000);
    }
  };

  const handleLiveDbConnect = async () => {
    setLiveDbConnectState("connecting");
    setLiveDbError(null);
    try {
      const isEntra = liveDbForm.dialect === "azure_sql";
      const body = isEntra
        ? {
            session_id: sessionId ?? anonymousSessionIdRef.current,
            dialect: "azure_sql",
            host: liveDbForm.host,
            port: parseInt(liveDbForm.port, 10),
            dbname: liveDbForm.dbname,
            auth_type: "entra_interactive",
            access_token: entraToken,
            pii_masking_enabled: liveDbForm.pii_masking_enabled,
          }
        : {
            session_id: sessionId ?? anonymousSessionIdRef.current,
            dialect: liveDbForm.dialect,
            host: liveDbForm.host,
            port: parseInt(liveDbForm.port, 10),
            dbname: liveDbForm.dbname,
            user: liveDbForm.user,
            password: liveDbForm.password,
            pii_masking_enabled: liveDbForm.pii_masking_enabled,
          };
      const response = await fetch("/api/db/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (promptLoginIfGated(response.status)) {
        setLiveDbConnectState("idle");
        return;
      }
      if (!response.ok) {
        const err = await response.json();
        const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail) || "Connection failed";
        throw new Error(detail);
      }
      const data = await response.json();
      setDbFileName(`${liveDbForm.dialect}://${liveDbForm.host}/${liveDbForm.dbname}`);
      setIsCsvConnection(false);
      setDbUploadState("connected");
      setShowLiveDbForm(false);
      setLiveDbConnectState("idle");
      handleDbConnected(data.connection_id);
      showToast("success", `Connected to ${liveDbForm.dbname}`);
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Connection failed";
      setLiveDbConnectState("error");
      setLiveDbError(msg);
    }
  };

  const msalConfig = {
    auth: {
      clientId: process.env.NEXT_PUBLIC_AZURE_CLIENT_ID ?? "",
      authority: "https://login.microsoftonline.com/common",
      redirectUri: typeof window !== "undefined" ? window.location.origin : "",
    },
  };

  async function handleMicrosoftSignIn() {
    setEntraSignInState("signing_in");
    try {
      const { PublicClientApplication } = await import("@azure/msal-browser");
      const msalInstance = new PublicClientApplication(msalConfig);
      await msalInstance.initialize();
      const result = await msalInstance.acquireTokenPopup({
        scopes: ["https://database.windows.net/.default"],
      });
      setEntraToken(result.accessToken);
      setEntraEmail(result.account?.username ?? "Microsoft Account");
      setEntraSignInState("signed_in");
    } catch (err) {
      console.error("MSAL sign-in error:", err);
      setEntraSignInState("error");
    }
  }

  const fetchWorkspace = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/workspace", { credentials: "include" });
      if (!res.ok) return;
      const data = await res.json();
      const parsed = WorkspaceSchema.safeParse(data);
      if (parsed.success) {
        setWorkspaceConnections(parsed.data.db_connections);
      }
    } catch {
      // non-fatal — workspace panel stays empty
    }
  }, [setWorkspaceConnections]);

  const handleLogout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
    } catch {
      // ignore
    }
    setAuthUser(null);
    setWorkspaceConnections([]);
    showToast("info", "Signed out");
  }, [showToast, setAuthUser, setWorkspaceConnections]);

  const handleEmailAuth = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthSubmitting(true);
    setAuthError(null);
    try {
      const endpoint = authModalTab === "register" ? "/api/auth/register" : "/api/auth/login";
      const body: Record<string, string> = { email: authEmail, password: authPassword };
      if (authModalTab === "register" && authName) body.name = authName;

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        setAuthError(data.detail ?? "Something went wrong.");
        return;
      }
      const meRes = await fetch("/api/auth/me", { credentials: "include" });
      if (meRes.ok) {
        const me = await meRes.json();
        const parsed = AuthMeSchema.safeParse(me);
        if (parsed.success) setAuthUser(parsed.data);
      }
      setAuthModalOpen(false);
      setAuthEmail("");
      setAuthPassword("");
      setAuthName("");
      showToast("success", authModalTab === "register" ? "Account created!" : "Welcome back!");
      fetchWorkspace();
    } catch {
      setAuthError("Network error. Please try again.");
    } finally {
      setAuthSubmitting(false);
    }
  }, [authModalTab, authEmail, authPassword, authName, showToast, fetchWorkspace, setAuthSubmitting, setAuthError, setAuthUser, setAuthModalOpen, setAuthEmail, setAuthPassword, setAuthName]);

  const loadAdminUsers = useCallback(async () => {
    setAdminUsersLoading(true);
    try {
      const res = await fetch("/admin/users", { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const parsed = AdminUsersResponseSchema.safeParse(data);
      if (parsed.success) setAdminUsers(parsed.data.users);
    } catch (err) {
      showToast("error", `Failed to load users: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAdminUsersLoading(false);
    }
  }, [showToast, setAdminUsersLoading, setAdminUsers]);

  // DOCBOT-1528: Usage tab — surfaces the existing admin-only GET
  // /admin/metrics endpoint (api/metrics_service.py), previously unused by
  // any frontend UI. Read-only; no new backend aggregation.
  const loadAdminMetrics = useCallback(async () => {
    setAdminMetricsLoading(true);
    try {
      const res = await fetch("/admin/metrics", { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const parsed = AdminMetricsSchema.safeParse(data);
      if (parsed.success) setAdminMetrics(parsed.data);
      else console.warn("AdminMetricsSchema parse failed", parsed.error);
    } catch (err) {
      showToast("error", `Failed to load usage metrics: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAdminMetricsLoading(false);
    }
  }, [showToast, setAdminMetricsLoading, setAdminMetrics]);

  // DOCBOT-1528: accepts a full filters object (event type + user + date
  // range) rather than a single eventType string, so AdminPanel's audit
  // filters compose instead of clobbering each other. All fields optional —
  // {} / undefined reproduces the old "all events, no range" behavior.
  const loadAuditLog = useCallback(async (filters?: AuditLogFilters) => {
    setAuditLoading(true);
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (filters?.eventType && filters.eventType !== "all") params.set("event_type", filters.eventType);
      if (filters?.userId && filters.userId !== "all") params.set("user_id", filters.userId);
      if (filters?.startDate) params.set("start_date", filters.startDate);
      if (filters?.endDate) params.set("end_date", filters.endDate);
      const res = await fetch(`/admin/audit-log?${params.toString()}`, { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const parsed = AuditLogResponseSchema.safeParse(data);
      if (parsed.success) setAuditEvents(parsed.data.events);
    } catch (err) {
      showToast("error", `Failed to load audit log: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAuditLoading(false);
    }
  }, [showToast, setAuditLoading, setAuditEvents]);

  const updateUserRole = useCallback(async (userId: string, role: string) => {
    try {
      const res = await fetch(`/admin/users/${userId}/role`, {
        method: "PATCH",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setAdminUsers(prev =>
        prev.map(u => u.id === userId ? { ...u, role: role as AdminUser["role"] } : u)
      );
      showToast("success", "Role updated");
    } catch (err) {
      showToast("error", `Role update failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [showToast, setAdminUsers]);

  const exportAuditLogCsv = useCallback(async (filters?: AuditLogFilters) => {
    try {
      const params = new URLSearchParams({ format: "csv", limit: "5000" });
      if (filters?.eventType && filters.eventType !== "all") params.set("event_type", filters.eventType);
      if (filters?.userId && filters.userId !== "all") params.set("user_id", filters.userId);
      if (filters?.startDate) params.set("start_date", filters.startDate);
      if (filters?.endDate) params.set("end_date", filters.endDate);
      const res = await fetch(`/admin/audit-log?${params.toString()}`, { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "audit_log.csv";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      showToast("success", "Audit log exported");
    } catch (err) {
      showToast("error", `Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [showToast]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const files = Array.from(e.target.files);

    const MAX_FILE_SIZE = 4.5 * 1024 * 1024; // 4.5 MB max for Vercel
    const totalSize = files.reduce((acc, file) => acc + file.size, 0);

    if (totalSize > MAX_FILE_SIZE) {
      showToast("error", "Total file size exceeds 4.5MB limit. Please upload smaller documents.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setUploadedFiles(files);
    setIsLoading(true);
    setUploadProgress(0);
    setFileUploadState("uploading");

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min((prev || 0) + 10, 90));
      }, 200);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (promptLoginIfGated(response.status)) {
        setFileUploadState("idle");
        setUploadedFiles([]);
        setUploadProgress(0);
        return;
      }
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to upload");
      }

      const data = await response.json();
      setSessionId(data.session_id);

      if (data.suggested_persona && data.suggested_persona !== "Generalist") {
        setSelectedPersona(data.suggested_persona);
        showToast("info", `Switched to ${data.suggested_persona} mode for your document`);
      } else {
        showToast("success", `Successfully processed ${files.length} document${files.length > 1 ? 's' : ''}`);
      }

      setFileUploadState("success");
      setTimeout(() => setFileUploadState("idle"), 2000);

    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Upload failed";
      console.error("Upload error:", error);
      setFileUploadState("error");
      showToast("error", `Upload failed: ${msg}`);
      setUploadedFiles([]);
      setTimeout(() => setFileUploadState("idle"), 3000);
    } finally {
      setIsLoading(false);
      setUploadProgress(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setFileUploadState("dragover");
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setFileUploadState("idle");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const files = Array.from(e.dataTransfer.files).filter(f => f.type === "application/pdf");

    if (files.length > 0) {
      const dataTransfer = new DataTransfer();
      files.forEach(f => dataTransfer.items.add(f));
      if (fileInputRef.current) {
        fileInputRef.current.files = dataTransfer.files;
        const event = { target: { files: dataTransfer.files } } as unknown as React.ChangeEvent<HTMLInputElement>;
        handleFileUpload(event);
      }
    } else {
      showToast("warning", "Please drop PDF files only");
      setFileUploadState("idle");
    }
  };

  const clearChat = () => {
    setMessages([]);
    showToast("info", "Chat cleared");
  };

  const clearSession = () => {
    setSessionId(null);
    setUploadedFiles([]);
    setMessages([]);
    showToast("info", "Session cleared");
  };

  const exportChat = async (format: 'txt' | 'markdown' | 'json') => {
    if (!sessionId) {
      showToast("error", "No active session to export");
      return;
    }

    try {
      const response = await fetch(`/api/export/${sessionId}?format=${format}`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error("Export failed");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `docbot-chat.${format === 'markdown' ? 'md' : format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      showToast("success", `Chat exported as ${format.toUpperCase()}`);
    } catch (error) {
      console.error("Export error:", error);
      showToast("error", "Failed to export chat");
    }
  };

  // ── Marketplace Connector handlers ──────────────────────────────────────

  const loadConnectors = useCallback(async () => {
    try {
      const res = await fetch("/api/connectors");
      if (res.ok) {
        const data = await res.json();
        setConnectors(data.connectors ?? []);
      }
    } catch {
      // non-fatal — connector list stays empty
    }
  }, [setConnectors]);

  const handleConnectorRegister = useCallback(async (connectorType: string, credentials: Record<string, string>) => {
    const res = await fetch("/api/connectors/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connector_type: connectorType, credentials }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Registration failed");
    }
    const data = await res.json();
    setConnectors(prev => [...prev, { connector_id: data.connector_id, connector_type: data.connector_type }]);
    showToast("success", `${connectorType} marketplace connected`);
  }, [setConnectors, showToast]);

  const handleConnectorSync = useCallback(async (connectorId: string, startDate: string, endDate: string): Promise<ConnectorSyncResponse> => {
    const res = await fetch(`/api/connectors/${connectorId}/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start_date: startDate, end_date: endDate }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Sync failed");
    }
    const data = await res.json();
    showToast("success", "Data synced successfully");
    // DOCBOT-1528: refresh the connector list so the persisted
    // "Last synced" badge (sourced from last_sync_at / sync_status) reflects
    // this sync immediately instead of only showing on next manual refresh.
    loadConnectors();
    return data as ConnectorSyncResponse;
  }, [showToast, loadConnectors]);

  const handleConnectorDisconnect = useCallback(async (connectorId: string) => {
    try {
      const res = await fetch(`/api/connectors/${connectorId}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Disconnect failed");
      }
    } catch (e) {
      console.error("Disconnect API error:", e);
    }
    setConnectors(prev => prev.filter(c => c.connector_id !== connectorId));
    showToast("info", "Marketplace disconnected");
  }, [setConnectors, showToast]);

  return {
    loadQueryHistory,
    loadSchemaCacheInfo,
    refreshSchema,
    handleDbConnected,
    handleDbDisconnect,
    handleDbUpload,
    handleLiveDbConnect,
    handleMicrosoftSignIn,
    fetchWorkspace,
    handleLogout,
    handleEmailAuth,
    loadAdminUsers,
    loadAuditLog,
    loadAdminMetrics,
    updateUserRole,
    exportAuditLogCsv,
    handleFileUpload,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    clearChat,
    clearSession,
    exportChat,
    loadConnectors,
    handleConnectorRegister,
    handleConnectorSync,
    handleConnectorDisconnect,
  };
}
