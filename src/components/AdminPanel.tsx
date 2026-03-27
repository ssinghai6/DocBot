"use client"

import React from "react"
import {
  X, Loader2, Users, ClipboardList, Shield,
  Filter, RefreshCw, Download,
} from "lucide-react"
import type { AdminUser, AuditEvent, AuthUser } from "@/components/types"

interface AdminPanelProps {
  adminTab: "users" | "audit"
  setAdminTab: (tab: "users" | "audit") => void
  adminUsers: AdminUser[]
  adminUsersLoading: boolean
  loadAdminUsers: () => void
  auditEvents: AuditEvent[]
  auditLoading: boolean
  auditEventTypeFilter: string
  setAuditEventTypeFilter: (v: string) => void
  loadAuditLog: (eventType: string) => void
  exportAuditLogCsv: () => void
  updateUserRole: (userId: string, role: string) => void
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
  loadAuditLog,
  exportAuditLogCsv,
  updateUserRole,
  authUser,
  showToast,
  onClose,
}: AdminPanelProps) {
  const eventColors: Record<string, string> = {
    login: "bg-[#10b981]/20 text-[#34d399]",
    logout: "bg-[#6b7280]/20 text-[#9ca3af]",
    upload: "bg-[#3b82f6]/20 text-[#60a5fa]",
    query: "bg-[#8b5cf6]/20 text-[#a78bfa]",
    db_connect: "bg-[#f59e0b]/20 text-[#fbbf24]",
    db_disconnect: "bg-[#ef4444]/20 text-[#f87171]",
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative z-10 w-full max-w-4xl max-h-[85vh] flex flex-col bg-[#12121a] border border-[#ffffff10] rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[#ffffff08] shrink-0">
          <div className="w-8 h-8 rounded-lg bg-[#f59e0b]/20 flex items-center justify-center">
            <Shield className="w-4 h-4 text-[#f59e0b]" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-white">Admin Panel</h2>
            <p className="text-xs text-gray-500">User management &amp; audit log</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {/* Tab switcher */}
            <div className="flex gap-1 p-1 bg-[#1a1a24] rounded-lg border border-[#ffffff08]">
              <button
                onClick={() => setAdminTab("users")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  adminTab === "users"
                    ? "bg-[#667eea]/20 text-[#a5b4fc] border border-[#667eea]/30"
                    : "text-gray-400 hover:text-gray-200 hover:bg-[#ffffff08]"
                }`}
              >
                <Users className="w-3 h-3" />
                Users
              </button>
              <button
                onClick={() => { setAdminTab("audit"); loadAuditLog(auditEventTypeFilter); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  adminTab === "audit"
                    ? "bg-[#667eea]/20 text-[#a5b4fc] border border-[#667eea]/30"
                    : "text-gray-400 hover:text-gray-200 hover:bg-[#ffffff08]"
                }`}
              >
                <ClipboardList className="w-3 h-3" />
                Audit Log
              </button>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-[#ffffff10] text-gray-500 hover:text-gray-200 transition-colors"
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
                <p className="text-xs text-gray-400">
                  {adminUsersLoading ? "Loading\u2026" : `${adminUsers.length} user${adminUsers.length !== 1 ? "s" : ""}`}
                </p>
                <button
                  onClick={loadAdminUsers}
                  disabled={adminUsersLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-200 bg-[#1a1a24] border border-[#ffffff08] hover:border-[#ffffff15] transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${adminUsersLoading ? "animate-spin" : ""}`} />
                  Refresh
                </button>
              </div>

              {adminUsersLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-[#667eea] animate-spin" />
                </div>
              ) : adminUsers.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                  <Users className="w-8 h-8 mb-3 opacity-40" />
                  <p className="text-sm">No users found</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[#ffffff08]">
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-4">User</th>
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-4">Provider</th>
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-4">Last Login</th>
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2">Role</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#ffffff06]">
                      {adminUsers.map((user) => (
                        <tr key={user.id} className="hover:bg-[#ffffff03] transition-colors">
                          <td className="py-3 pr-4">
                            <div>
                              <p className="text-xs font-medium text-gray-200 truncate max-w-[200px]">{user.email}</p>
                              <p className="text-[10px] text-gray-500 truncate max-w-[200px]">{user.name}</p>
                            </div>
                          </td>
                          <td className="py-3 pr-4">
                            <span className="text-xs text-gray-400 capitalize">{user.provider}</span>
                          </td>
                          <td className="py-3 pr-4">
                            <span className="text-xs text-gray-500">
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
                              className="text-xs px-2 py-1 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 focus:outline-none focus:border-[#667eea]/40 cursor-pointer"
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
                  <Filter className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                  <select
                    value={auditEventTypeFilter}
                    onChange={(e) => {
                      setAuditEventTypeFilter(e.target.value);
                      loadAuditLog(e.target.value);
                    }}
                    className="text-xs px-2.5 py-1.5 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 focus:outline-none focus:border-[#667eea]/40"
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
                <p className="text-xs text-gray-500 flex-1">
                  {auditLoading ? "Loading\u2026" : `${auditEvents.length} event${auditEvents.length !== 1 ? "s" : ""}`}
                </p>
                <button
                  onClick={exportAuditLogCsv}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-200 bg-[#1a1a24] border border-[#ffffff08] hover:border-[#ffffff15] transition-all"
                >
                  <Download className="w-3 h-3" />
                  Export CSV
                </button>
                <button
                  onClick={() => loadAuditLog(auditEventTypeFilter)}
                  disabled={auditLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-200 bg-[#1a1a24] border border-[#ffffff08] hover:border-[#ffffff15] transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-3 h-3 ${auditLoading ? "animate-spin" : ""}`} />
                </button>
              </div>

              {auditLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-[#667eea] animate-spin" />
                </div>
              ) : auditEvents.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                  <ClipboardList className="w-8 h-8 mb-3 opacity-40" />
                  <p className="text-sm">No events found</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[#ffffff08]">
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-3">Time</th>
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-3">Event</th>
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-3">Detail</th>
                        <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-2">Session</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#ffffff06]">
                      {auditEvents.map((ev) => {
                        const colorClass = eventColors[ev.event_type] ?? "bg-[#ffffff10] text-gray-400";
                        return (
                          <tr key={ev.id} className="hover:bg-[#ffffff03] transition-colors">
                            <td className="py-2.5 pr-3 whitespace-nowrap">
                              <span className="text-xs text-gray-500">
                                {ev.occurred_at
                                  ? new Date(ev.occurred_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
                                  : "\u2014"}
                              </span>
                            </td>
                            <td className="py-2.5 pr-3">
                              <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wide ${colorClass}`}>
                                {ev.event_type}
                              </span>
                            </td>
                            <td className="py-2.5 pr-3 max-w-[220px]">
                              <p className="text-xs text-gray-400 truncate" title={ev.detail ?? ""}>
                                {ev.detail || "\u2014"}
                              </p>
                            </td>
                            <td className="py-2.5">
                              <span className="text-[10px] text-gray-600 font-mono">
                                {ev.session_id ? ev.session_id.slice(0, 8) + "\u2026" : "\u2014"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
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
