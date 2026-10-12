"use client"

import React, { useMemo, useState } from "react"
import {
  X, Loader2, Users, ClipboardList, Shield,
  Filter, RefreshCw, Download, ChevronDown, ChevronRight,
  BarChart3, KeyRound, Copy, Check, Trash2, Plus,
} from "lucide-react"
import type { AdminUser, AuditEvent, AuthUser, AdminMetrics, ApiKey, ApiKeyCreateResponse } from "@/components/types"
import type { AuditLogFilters } from "@/hooks/useChatHandlers"

interface AdminPanelProps {
  adminTab: "users" | "audit" | "usage" | "keys"
  setAdminTab: (tab: "users" | "audit" | "usage" | "keys") => void
  adminUsers: AdminUser[]
  adminUsersLoading: boolean
  loadAdminUsers: () => void
  auditEvents: AuditEvent[]
  auditLoading: boolean
  auditEventTypeFilter: string
  setAuditEventTypeFilter: (v: string) => void
  // DOCBOT-1528: audit log filters — user + date range
  auditUserIdFilter: string
  setAuditUserIdFilter: (v: string) => void
  auditStartDate: string
  setAuditStartDate: (v: string) => void
  auditEndDate: string
  setAuditEndDate: (v: string) => void
  loadAuditLog: (filters?: AuditLogFilters) => void
  exportAuditLogCsv: (filters?: AuditLogFilters) => void
  updateUserRole: (userId: string, role: string) => void
  // DOCBOT-1528: Usage tab (GET /admin/metrics)
  adminMetrics: AdminMetrics | null
  adminMetricsLoading: boolean
  loadAdminMetrics: () => void
  // DOCBOT-1530: Keys tab — developer API key management
  apiKeys: ApiKey[]
  apiKeysLoading: boolean
  newlyCreatedApiKey: ApiKeyCreateResponse | null
  loadApiKeys: () => void
  createApiKey: (name: string, scopes: string[]) => void
  revokeApiKey: (keyId: string) => void
  dismissNewlyCreatedApiKey: () => void
  authUser: AuthUser | null
  showToast: (type: "success" | "error" | "warning" | "info", message: string) => void
  onClose: () => void
}

export default function AdminPanel({
  adminTab,
  setAdminTab,
  adminUsers,
  adminUsersLoading,
  loadAdminUsers,
  auditEvents,
  auditLoading,
  auditEventTypeFilter,
  setAuditEventTypeFilter,
  auditUserIdFilter,
  setAuditUserIdFilter,
  auditStartDate,
  setAuditStartDate,
  auditEndDate,
  setAuditEndDate,
  loadAuditLog,
  exportAuditLogCsv,
  updateUserRole,
  adminMetrics,
  adminMetricsLoading,
  loadAdminMetrics,
  apiKeys,
  apiKeysLoading,
  newlyCreatedApiKey,
  loadApiKeys,
  createApiKey,
  revokeApiKey,
  dismissNewlyCreatedApiKey,
  authUser,
  showToast,
  onClose,
}: AdminPanelProps) {
  const eventColors: Record<string, string> = {
    login: "bg-[var(--color-success-500)]/20 text-[#34d399]",
    logout: "bg-[#6b7280]/20 text-[#9ca3af]",
    upload: "bg-[var(--color-cyan-500)]/20 text-[#60a5fa]",
    query: "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)]",
    db_connect: "bg-[var(--color-amber-500)]/20 text-[#fbbf24]",
    db_disconnect: "bg-[var(--color-danger-500)]/20 text-[var(--color-danger-500)]",
  }

  // DOCBOT-1528: click-to-expand full row detail (metadata_json is often
  // truncated by the Detail column's `truncate` class — this surfaces it in full).
  const [expandedAuditId, setExpandedAuditId] = useState<string | null>(null);

  // DOCBOT-1530: Keys tab — create-key form state (local to this panel; the
  // issued key itself lives in `newlyCreatedApiKey`, owned by the parent so
  // it survives a tab switch until explicitly dismissed).
  const ALL_SCOPES = ["hybrid:read", "lineage:read", "autopilot:run", "connectors:manage"] as const;
  const [showCreateKeyForm, setShowCreateKeyForm] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyScopes, setNewKeyScopes] = useState<string[]>([]);
  const [copiedKey, setCopiedKey] = useState(false);

  const toggleScope = (scope: string) => {
    setNewKeyScopes(prev => prev.includes(scope) ? prev.filter(s => s !== scope) : [...prev, scope]);
  };

  const submitCreateKey = () => {
    if (!newKeyName.trim() || newKeyScopes.length === 0) {
      showToast("warning", "Name and at least one scope are required.");
      return;
    }
    createApiKey(newKeyName.trim(), newKeyScopes);
    setShowCreateKeyForm(false);
    setNewKeyName("");
    setNewKeyScopes([]);
  };

  const currentAuditFilters = (): AuditLogFilters => ({
    eventType: auditEventTypeFilter,
    userId: auditUserIdFilter,
    startDate: auditStartDate,
    endDate: auditEndDate,
  });

  // DOCBOT-1528: user-email filter dropdown, sourced from the admin users
  // list already loaded for the Users tab — no new backend call needed.
  const userOptions = useMemo(
    () => adminUsers.map(u => ({ id: u.id, email: u.email })).sort((a, b) => a.email.localeCompare(b.email)),
    [adminUsers]
  );

  const prettyMetadata = (metadataJson: string | null): string => {
    if (!metadataJson) return "";
    try {
      return JSON.stringify(JSON.parse(metadataJson), null, 2);
    } catch {
      return metadataJson;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative z-10 w-full max-w-4xl max-h-[85vh] flex flex-col bg-[var(--color-bg-surface)] border border-[var(--color-border-default)] rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[var(--color-border-subtle)] shrink-0">
          <div className="w-8 h-8 rounded-lg bg-[var(--color-amber-500)]/20 flex items-center justify-center">
            <Shield className="w-4 h-4 text-[var(--color-amber-500)]" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Admin Panel</h2>
            <p className="text-xs text-[var(--color-text-tertiary)]">User management, audit log &amp; usage</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {/* Tab switcher */}
            <div className="flex gap-1 p-1 bg-[var(--color-bg-elevated)] rounded-lg border border-[var(--color-border-subtle)]">
              <button
                onClick={() => setAdminTab("users")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  adminTab === "users"
                    ? "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)] border border-[var(--color-cyan-500)]/30"
                    : "text-[var(--color-text-secondary)] hover:text-gray-200 hover:bg-[var(--color-bg-overlay)]"
                }`}
              >
                <Users className="w-3 h-3" />
                Users
              </button>
              <button
                onClick={() => { setAdminTab("audit"); loadAuditLog(currentAuditFilters()); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  adminTab === "audit"
                    ? "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)] border border-[var(--color-cyan-500)]/30"
                    : "text-[var(--color-text-secondary)] hover:text-gray-200 hover:bg-[var(--color-bg-overlay)]"
                }`}
              >
                <ClipboardList className="w-3 h-3" />
                Audit Log
              </button>
              <button
                onClick={() => { setAdminTab("usage"); loadAdminMetrics(); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  adminTab === "usage"
                    ? "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)] border border-[var(--color-cyan-500)]/30"
                    : "text-[var(--color-text-secondary)] hover:text-gray-200 hover:bg-[var(--color-bg-overlay)]"
                }`}
              >
                <BarChart3 className="w-3 h-3" />
                Usage
              </button>
              <button
                onClick={() => { setAdminTab("keys"); loadApiKeys(); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  adminTab === "keys"
                    ? "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)] border border-[var(--color-cyan-500)]/30"
                    : "text-[var(--color-text-secondary)] hover:text-gray-200 hover:bg-[var(--color-bg-overlay)]"
                }`}
              >
                <KeyRound className="w-3 h-3" />
                Keys
              </button>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-[var(--color-bg-overlay)] text-[var(--color-text-tertiary)] hover:text-gray-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto">
          {adminTab === "users" && (
            <div className="p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs text-[var(--color-text-secondary)]">
                  {adminUsersLoading ? "Loading…" : `${adminUsers.length} user${adminUsers.length !== 1 ? "s" : ""}`}
                </p>
                <button
                  onClick={loadAdminUsers}
                  disabled={adminUsersLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-secondary)] hover:text-gray-200 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] hover:border-[var(--color-border-default)] transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${adminUsersLoading ? "animate-spin" : ""}`} />
                  Refresh
                </button>
              </div>

              {adminUsersLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-[var(--color-cyan-500)] animate-spin" />
                </div>
              ) : adminUsers.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-tertiary)]">
                  <Users className="w-8 h-8 mb-3 opacity-40" />
                  <p className="text-sm">No users found</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[var(--color-border-subtle)]">
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">User</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">Provider</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">Last Login</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2">Role</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#ffffff06]">
                      {adminUsers.map((user) => (
                        <tr key={user.id} className="hover:bg-[#ffffff03] transition-colors">
                          <td className="py-3 pr-4">
                            <div>
                              <p className="text-xs font-medium text-gray-200 truncate max-w-[200px]">{user.email}</p>
                              <p className="text-[10px] text-[var(--color-text-tertiary)] truncate max-w-[200px]">{user.name}</p>
                            </div>
                          </td>
                          <td className="py-3 pr-4">
                            <span className="text-xs text-[var(--color-text-secondary)] capitalize">{user.provider}</span>
                          </td>
                          <td className="py-3 pr-4">
                            <span className="text-xs text-[var(--color-text-tertiary)]">
                              {user.last_login_at
                                ? new Date(user.last_login_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })
                                : "Never"}
                            </span>
                          </td>
                          <td className="py-3">
                            <select
                              value={user.role}
                              onChange={(e) => {
                                if (authUser?.id === user.id && e.target.value !== "admin") {
                                  showToast("warning", "You cannot change your own role");
                                  return;
                                }
                                updateUserRole(user.id, e.target.value);
                              }}
                              className="text-xs px-2 py-1 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 focus:outline-none focus:border-[var(--color-cyan-500)]/40 cursor-pointer"
                            >
                              <option value="viewer">viewer</option>
                              <option value="analyst">analyst</option>
                              <option value="admin">admin</option>
                            </select>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {adminTab === "audit" && (
            <div className="p-5">
              <div className="flex items-center gap-3 mb-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <Filter className="w-3.5 h-3.5 text-[var(--color-text-tertiary)] shrink-0" />
                  <select
                    value={auditEventTypeFilter}
                    onChange={(e) => {
                      setAuditEventTypeFilter(e.target.value);
                      loadAuditLog({ ...currentAuditFilters(), eventType: e.target.value });
                    }}
                    className="text-xs px-2.5 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                  >
                    <option value="all">All events</option>
                    <option value="login">login</option>
                    <option value="logout">logout</option>
                    <option value="upload">upload</option>
                    <option value="query">query</option>
                    <option value="db_connect">db_connect</option>
                    <option value="db_disconnect">db_disconnect</option>
                  </select>
                </div>

                {/* DOCBOT-1528: user-email filter, sourced from the Users tab's list */}
                <select
                  value={auditUserIdFilter}
                  onChange={(e) => {
                    setAuditUserIdFilter(e.target.value);
                    loadAuditLog({ ...currentAuditFilters(), userId: e.target.value });
                  }}
                  className="text-xs px-2.5 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                  title="Filter by user"
                >
                  <option value="all">All users</option>
                  {userOptions.map(u => (
                    <option key={u.id} value={u.id}>{u.email}</option>
                  ))}
                </select>

                {/* DOCBOT-1528: date range filter */}
                <div className="flex items-center gap-1.5">
                  <input
                    type="date"
                    value={auditStartDate}
                    onChange={(e) => {
                      setAuditStartDate(e.target.value);
                      loadAuditLog({ ...currentAuditFilters(), startDate: e.target.value });
                    }}
                    className="text-xs px-2 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                    title="From date"
                  />
                  <span className="text-[10px] text-[var(--color-text-tertiary)]">to</span>
                  <input
                    type="date"
                    value={auditEndDate}
                    onChange={(e) => {
                      setAuditEndDate(e.target.value);
                      loadAuditLog({ ...currentAuditFilters(), endDate: e.target.value });
                    }}
                    className="text-xs px-2 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                    title="To date"
                  />
                  {(auditStartDate || auditEndDate) && (
                    <button
                      onClick={() => {
                        setAuditStartDate("");
                        setAuditEndDate("");
                        loadAuditLog({ ...currentAuditFilters(), startDate: "", endDate: "" });
                      }}
                      className="text-[10px] text-[var(--color-text-tertiary)] hover:text-gray-200 underline"
                    >
                      clear
                    </button>
                  )}
                </div>

                <p className="text-xs text-[var(--color-text-tertiary)] flex-1 text-right">
                  {auditLoading ? "Loading…" : `${auditEvents.length} event${auditEvents.length !== 1 ? "s" : ""}`}
                </p>
                <button
                  onClick={() => exportAuditLogCsv(currentAuditFilters())}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-secondary)] hover:text-gray-200 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] hover:border-[var(--color-border-default)] transition-all"
                >
                  <Download className="w-3 h-3" />
                  Export CSV
                </button>
                <button
                  onClick={() => loadAuditLog(currentAuditFilters())}
                  disabled={auditLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-secondary)] hover:text-gray-200 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] hover:border-[var(--color-border-default)] transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${auditLoading ? "animate-spin" : ""}`} />
                </button>
              </div>

              {auditLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-[var(--color-cyan-500)] animate-spin" />
                </div>
              ) : auditEvents.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-tertiary)]">
                  <ClipboardList className="w-8 h-8 mb-3 opacity-40" />
                  <p className="text-sm">No events found</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[var(--color-border-subtle)]">
                        <th className="w-5 pb-2" />
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-3">Time</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-3">Event</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-3">Detail</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2">Session</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#ffffff06]">
                      {auditEvents.map((ev) => {
                        const colorClass = eventColors[ev.event_type] ?? "bg-[#ffffff10] text-[var(--color-text-secondary)]";
                        const isExpanded = expandedAuditId === ev.id;
                        return (
                          <React.Fragment key={ev.id}>
                            <tr
                              className="hover:bg-[#ffffff03] transition-colors cursor-pointer"
                              onClick={() => setExpandedAuditId(isExpanded ? null : ev.id)}
                            >
                              <td className="py-2.5 pl-1 text-[var(--color-text-tertiary)]">
                                {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                              </td>
                              <td className="py-2.5 pr-3 whitespace-nowrap">
                                <span className="text-xs text-[var(--color-text-tertiary)]">
                                  {ev.occurred_at
                                    ? new Date(ev.occurred_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
                                    : "—"}
                                </span>
                              </td>
                              <td className="py-2.5 pr-3">
                                <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wide ${colorClass}`}>
                                  {ev.event_type}
                                </span>
                              </td>
                              <td className="py-2.5 pr-3 max-w-[220px]">
                                <p className="text-xs text-[var(--color-text-secondary)] truncate" title={ev.detail ?? ""}>
                                  {ev.detail || "—"}
                                </p>
                              </td>
                              <td className="py-2.5">
                                <span className="text-[10px] text-[var(--color-text-quaternary)] font-mono">
                                  {ev.session_id ? ev.session_id.slice(0, 8) + "…" : "—"}
                                </span>
                              </td>
                            </tr>
                            {/* DOCBOT-1528: click-to-expand full row detail */}
                            {isExpanded && (
                              <tr>
                                <td colSpan={5} className="pb-3 pl-6 pr-3">
                                  <div className="rounded-lg bg-[var(--color-bg-inset)] border border-[var(--color-border-subtle)] p-3 space-y-2">
                                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                                      <div>
                                        <span className="text-[var(--color-text-tertiary)]">Event ID: </span>
                                        <span className="text-gray-300 font-mono">{ev.id}</span>
                                      </div>
                                      <div>
                                        <span className="text-[var(--color-text-tertiary)]">User ID: </span>
                                        <span className="text-gray-300 font-mono">{ev.user_id || "—"}</span>
                                      </div>
                                      <div>
                                        <span className="text-[var(--color-text-tertiary)]">Session: </span>
                                        <span className="text-gray-300 font-mono">{ev.session_id || "—"}</span>
                                      </div>
                                      <div>
                                        <span className="text-[var(--color-text-tertiary)]">Occurred: </span>
                                        <span className="text-gray-300">
                                          {ev.occurred_at ? new Date(ev.occurred_at).toLocaleString() : "—"}
                                        </span>
                                      </div>
                                    </div>
                                    <div>
                                      <p className="text-[var(--color-text-tertiary)] text-[11px] mb-1">Detail</p>
                                      <p className="text-xs text-gray-300 whitespace-pre-wrap break-all">{ev.detail || "—"}</p>
                                    </div>
                                    {ev.metadata_json && (
                                      <div>
                                        <p className="text-[var(--color-text-tertiary)] text-[11px] mb-1">Metadata</p>
                                        <pre className="text-[10px] text-[var(--color-text-secondary)] bg-[var(--color-bg-base)] rounded-lg p-2 overflow-x-auto whitespace-pre-wrap break-all leading-relaxed">
                                          {prettyMetadata(ev.metadata_json)}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {adminTab === "usage" && (
            <div className="p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs text-[var(--color-text-secondary)]">
                  Platform usage &amp; LLM cost/latency (GET /admin/metrics)
                </p>
                <button
                  onClick={loadAdminMetrics}
                  disabled={adminMetricsLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-secondary)] hover:text-gray-200 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] hover:border-[var(--color-border-default)] transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${adminMetricsLoading ? "animate-spin" : ""}`} />
                  Refresh
                </button>
              </div>

              {adminMetricsLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-[var(--color-cyan-500)] animate-spin" />
                </div>
              ) : !adminMetrics ? (
                <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-tertiary)]">
                  <BarChart3 className="w-8 h-8 mb-3 opacity-40" />
                  <p className="text-sm">No metrics loaded</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Top-line stat tiles */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {[
                      { label: "Total sessions", value: adminMetrics.total_sessions },
                      { label: "Total queries", value: adminMetrics.total_queries },
                      { label: "Documents uploaded", value: adminMetrics.total_documents_uploaded },
                      { label: "Active DB connections", value: adminMetrics.active_db_connections },
                      { label: "Avg response time", value: adminMetrics.avg_response_time_ms != null ? `${Math.round(adminMetrics.avg_response_time_ms)}ms` : "—" },
                      { label: "Uptime", value: `${Math.floor(adminMetrics.uptime_seconds / 3600)}h` },
                    ].map((tile) => (
                      <div key={tile.label} className="rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] p-3">
                        <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1">{tile.label}</p>
                        <p className="text-lg font-semibold text-gray-200">{tile.value}</p>
                      </div>
                    ))}
                  </div>

                  {/* Queries by type */}
                  <div>
                    <p className="text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider mb-2">Queries by type</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(adminMetrics.queries_by_type).map(([type, count]) => (
                        <span key={type} className="text-xs px-2.5 py-1 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] text-gray-300">
                          {type}: <span className="font-semibold">{count}</span>
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* LLM cost/latency */}
                  {adminMetrics.llm_metrics && (
                    <div>
                      <p className="text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider mb-2">
                        LLM spend ({adminMetrics.llm_metrics.window_days}-day window)
                      </p>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
                        <div className="rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] p-3">
                          <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1">Total calls</p>
                          <p className="text-base font-semibold text-gray-200">{adminMetrics.llm_metrics.total_calls}</p>
                        </div>
                        <div className="rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] p-3">
                          <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1">Total cost</p>
                          <p className="text-base font-semibold text-gray-200">${adminMetrics.llm_metrics.total_cost_usd.toFixed(4)}</p>
                        </div>
                        <div className="rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] p-3">
                          <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1">Fallback rate</p>
                          <p className="text-base font-semibold text-gray-200">
                            {adminMetrics.llm_metrics.fallback_rate != null ? `${(adminMetrics.llm_metrics.fallback_rate * 100).toFixed(1)}%` : "—"}
                          </p>
                        </div>
                        <div className="rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] p-3">
                          <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1">p95 latency</p>
                          <p className="text-base font-semibold text-gray-200">
                            {adminMetrics.llm_metrics.p95_latency_ms != null ? `${Math.round(adminMetrics.llm_metrics.p95_latency_ms)}ms` : "—"}
                          </p>
                        </div>
                      </div>

                      {adminMetrics.llm_metrics.by_caller.length > 0 && (
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b border-[var(--color-border-subtle)]">
                                <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-3">Caller</th>
                                <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-3">Calls</th>
                                <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-3">Cost</th>
                                <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2">p50 / p95</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-[#ffffff06]">
                              {adminMetrics.llm_metrics.by_caller.map((c) => (
                                <tr key={c.caller}>
                                  <td className="py-2 pr-3 text-xs text-gray-300 font-mono">{c.caller}</td>
                                  <td className="py-2 pr-3 text-xs text-gray-300">{c.calls}</td>
                                  <td className="py-2 pr-3 text-xs text-gray-300">${c.cost_usd.toFixed(4)}</td>
                                  <td className="py-2 text-xs text-gray-300">
                                    {c.p50_latency_ms != null ? Math.round(c.p50_latency_ms) : "—"}ms / {c.p95_latency_ms != null ? Math.round(c.p95_latency_ms) : "—"}ms
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}

                  {/* LLM response cache */}
                  {adminMetrics.llm_cache_metrics && (
                    <div>
                      <p className="text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider mb-2">LLM response cache</p>
                      <div className="flex flex-wrap gap-2">
                        <span className="text-xs px-2.5 py-1 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] text-gray-300">
                          Hits: <span className="font-semibold">{adminMetrics.llm_cache_metrics.hits}</span>
                        </span>
                        <span className="text-xs px-2.5 py-1 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] text-gray-300">
                          Misses: <span className="font-semibold">{adminMetrics.llm_cache_metrics.misses}</span>
                        </span>
                        <span className="text-xs px-2.5 py-1 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] text-gray-300">
                          Hit rate: <span className="font-semibold">
                            {adminMetrics.llm_cache_metrics.hit_rate != null ? `${(adminMetrics.llm_cache_metrics.hit_rate * 100).toFixed(1)}%` : "—"}
                          </span>
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {adminTab === "keys" && (
            <div className="p-5">
              {/* DOCBOT-1530: raw key is shown exactly once, right after creation */}
              {newlyCreatedApiKey && (
                <div className="mb-4 rounded-lg border border-[var(--color-success-500)]/30 bg-[var(--color-success-500)]/10 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-[#34d399] mb-1">
                        Key created — copy it now, it won&apos;t be shown again
                      </p>
                      <code className="block text-xs text-gray-200 bg-black/30 rounded px-2 py-1.5 break-all">
                        {newlyCreatedApiKey.raw_key}
                      </code>
                    </div>
                    <button
                      onClick={dismissNewlyCreatedApiKey}
                      className="shrink-0 p-1 rounded text-[var(--color-text-tertiary)] hover:text-gray-200"
                      aria-label="Dismiss"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <button
                    onClick={async () => {
                      try {
                        await navigator.clipboard.writeText(newlyCreatedApiKey.raw_key);
                        setCopiedKey(true);
                        setTimeout(() => setCopiedKey(false), 2000);
                      } catch {
                        showToast("error", "Copy failed — select and copy manually.");
                      }
                    }}
                    className="mt-2 flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs text-[#34d399] bg-[var(--color-success-500)]/10 border border-[var(--color-success-500)]/30 hover:bg-[var(--color-success-500)]/20 transition-all"
                  >
                    {copiedKey ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                    {copiedKey ? "Copied" : "Copy key"}
                  </button>
                </div>
              )}

              <div className="flex items-center justify-between mb-4">
                <p className="text-xs text-[var(--color-text-secondary)]">
                  {apiKeysLoading ? "Loading…" : `${apiKeys.length} key${apiKeys.length !== 1 ? "s" : ""}`}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowCreateKeyForm(v => !v)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-cyan-500)] hover:text-gray-200 bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/30 hover:border-[var(--color-cyan-500)]/50 transition-all"
                  >
                    <Plus className="w-3 h-3" />
                    New key
                  </button>
                  <button
                    onClick={loadApiKeys}
                    disabled={apiKeysLoading}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-secondary)] hover:text-gray-200 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] hover:border-[var(--color-border-default)] transition-all disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3 h-3 ${apiKeysLoading ? "animate-spin" : ""}`} />
                    Refresh
                  </button>
                </div>
              </div>

              {showCreateKeyForm && (
                <div className="mb-4 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] p-3 space-y-3">
                  <div>
                    <label className="block text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1">Name</label>
                    <input
                      type="text"
                      value={newKeyName}
                      onChange={(e) => setNewKeyName(e.target.value)}
                      placeholder="e.g. Production backend"
                      className="w-full text-xs px-2.5 py-1.5 rounded-lg bg-[var(--color-bg-surface)] border border-[var(--color-border-default)] text-gray-200 placeholder-gray-500 focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1.5">Scopes</label>
                    <div className="flex flex-wrap gap-2">
                      {ALL_SCOPES.map((scope) => (
                        <label
                          key={scope}
                          className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border cursor-pointer transition-all ${
                            newKeyScopes.includes(scope)
                              ? "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)] border-[var(--color-cyan-500)]/30"
                              : "bg-[var(--color-bg-surface)] text-[var(--color-text-secondary)] border-[var(--color-border-subtle)]"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={newKeyScopes.includes(scope)}
                            onChange={() => toggleScope(scope)}
                            className="sr-only"
                          />
                          {scope}
                        </label>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={submitCreateKey}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-[var(--color-cyan-500)] hover:opacity-90 transition-opacity"
                    >
                      Create key
                    </button>
                    <button
                      onClick={() => { setShowCreateKeyForm(false); setNewKeyName(""); setNewKeyScopes([]); }}
                      className="px-3 py-1.5 rounded-lg text-xs text-[var(--color-text-secondary)] hover:text-gray-200"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {apiKeysLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-[var(--color-cyan-500)] animate-spin" />
                </div>
              ) : apiKeys.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-tertiary)]">
                  <KeyRound className="w-8 h-8 mb-3 opacity-40" />
                  <p className="text-sm">No API keys issued yet</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[var(--color-border-subtle)]">
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">Key</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">Scopes</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">Created</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2 pr-4">Last used</th>
                        <th className="text-left text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider pb-2">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#ffffff06]">
                      {apiKeys.map((key) => (
                        <tr key={key.id} className="hover:bg-[#ffffff03] transition-colors">
                          <td className="py-3 pr-4">
                            <p className="text-xs font-medium text-gray-200">{key.name}</p>
                            <code className="text-[10px] text-[var(--color-text-tertiary)]">{key.key_prefix}…</code>
                          </td>
                          <td className="py-3 pr-4">
                            <div className="flex flex-wrap gap-1">
                              {key.scopes.map((s) => (
                                <span key={s} className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)]">
                                  {s}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="py-3 pr-4 text-xs text-[var(--color-text-tertiary)]">
                            {key.created_at ? new Date(key.created_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) : "—"}
                          </td>
                          <td className="py-3 pr-4 text-xs text-[var(--color-text-tertiary)]">
                            {key.last_used_at ? new Date(key.last_used_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) : "Never"}
                          </td>
                          <td className="py-3">
                            {key.revoked_at ? (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-danger-500)]/20 text-[var(--color-danger-500)]">Revoked</span>
                            ) : (
                              <button
                                onClick={() => revokeApiKey(key.id)}
                                className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-danger-500)]/10 text-[var(--color-danger-500)] border border-[var(--color-danger-500)]/30 hover:bg-[var(--color-danger-500)]/20 transition-all"
                              >
                                <Trash2 className="w-2.5 h-2.5" />
                                Revoke
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
