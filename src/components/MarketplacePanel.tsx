"use client"

import React, { useState } from "react"
import {
  ShoppingCart, Plus, ChevronDown, XCircle,
  RefreshCw, Loader2, Eye, EyeOff, AlertCircle,
} from "lucide-react"
import type { ConnectorInfo, ConnectorSyncResponse } from "./types"

interface MarketplacePanelProps {
  connectors: ConnectorInfo[]
  onRegister: (connectorType: string, credentials: Record<string, string>) => Promise<void>
  onSync: (connectorId: string, startDate: string, endDate: string) => Promise<ConnectorSyncResponse>
  onDisconnect: (connectorId: string) => void
}

type CredentialField = {
  key: string
  label: string
  type: "text" | "password"
  placeholder: string
  optional?: boolean
}

const CREDENTIAL_FIELDS: Record<string, CredentialField[]> = {
  amazon: [
    { key: "client_id", label: "Client ID", type: "text", placeholder: "LWA Client ID" },
    { key: "client_secret", label: "Client Secret", type: "password", placeholder: "LWA Client Secret" },
    { key: "refresh_token", label: "Refresh Token", type: "password", placeholder: "LWA Refresh Token" },
    { key: "marketplace_id", label: "Marketplace ID", type: "text", placeholder: "e.g. ATVPDKIKX0DER" },
  ],
  shopify: [
    { key: "shop_domain", label: "Shop Domain", type: "text", placeholder: "mystore.myshopify.com" },
    { key: "access_token", label: "Access Token", type: "password", placeholder: "Admin API access token" },
    { key: "webhook_secret", label: "Webhook Secret", type: "password", placeholder: "Optional — for webhook verification", optional: true },
  ],
}

const CONNECTOR_TYPES = Object.keys(CREDENTIAL_FIELDS)

function ConnectorTypeLabel({ type }: { type: string }) {
  const labels: Record<string, string> = {
    amazon: "Amazon SP-API",
    shopify: "Shopify",
  }
  return <>{labels[type] ?? type}</>
}

export default function MarketplacePanel({
  connectors,
  onRegister,
  onSync,
  onDisconnect,
}: MarketplacePanelProps) {
  // Registration form state
  const [showForm, setShowForm] = useState(false)
  const [selectedType, setSelectedType] = useState(CONNECTOR_TYPES[0])
  const [credentials, setCredentials] = useState<Record<string, string>>({})
  const [registerState, setRegisterState] = useState<"idle" | "submitting" | "error">("idle")
  const [registerError, setRegisterError] = useState<string | null>(null)
  const [visibleFields, setVisibleFields] = useState<Set<string>>(new Set())

  // Per-connector sync state
  const [syncingId, setSyncingId] = useState<string | null>(null)
  const [syncDates, setSyncDates] = useState<Record<string, { start: string; end: string }>>({})
  const [expandedSyncId, setExpandedSyncId] = useState<string | null>(null)
  const [syncResult, setSyncResult] = useState<{ id: string; result: ConnectorSyncResponse } | null>(null)
  const [syncError, setSyncError] = useState<string | null>(null)

  const toggleFieldVisibility = (key: string) => {
    setVisibleFields(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const handleRegister = async () => {
    setRegisterState("submitting")
    setRegisterError(null)
    try {
      await onRegister(selectedType, credentials)
      // Reset form on success
      setShowForm(false)
      setCredentials({})
      setVisibleFields(new Set())
      setRegisterState("idle")
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Registration failed"
      setRegisterError(msg)
      setRegisterState("error")
    }
  }

  const handleSync = async (connectorId: string) => {
    const dates = syncDates[connectorId]
    if (!dates?.start || !dates?.end) return

    setSyncingId(connectorId)
    setSyncError(null)
    setSyncResult(null)
    try {
      const result = await onSync(connectorId, dates.start, dates.end)
      setSyncResult({ id: connectorId, result })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Sync failed"
      setSyncError(msg)
    } finally {
      setSyncingId(null)
    }
  }

  const updateSyncDate = (connectorId: string, field: "start" | "end", value: string) => {
    setSyncDates(prev => ({
      ...prev,
      [connectorId]: {
        start: prev[connectorId]?.start ?? "",
        end: prev[connectorId]?.end ?? "",
        [field]: value,
      },
    }))
  }

  const fields = CREDENTIAL_FIELDS[selectedType] ?? []
  const allFieldsFilled = fields.filter(f => !f.optional).every(f => credentials[f.key]?.trim())

  return (
    <div className="mb-5">
      <h3 className="text-sm font-semibold mb-3 text-[var(--color-text-primary)] flex items-center gap-2">
        <ShoppingCart className="w-4 h-4 text-[var(--color-cyan-500)]" />
        Marketplace
        {connectors.length > 0 && (
          <span className="ml-auto text-[10px] text-[var(--color-cyan-500)] font-medium bg-[var(--color-cyan-500)]/15 px-1.5 py-0.5 rounded-full">
            {connectors.length}
          </span>
        )}
      </h3>

      {/* Connected connectors */}
      {connectors.length > 0 && (
        <div className="space-y-2 mb-2">
          {connectors.map(conn => (
            <div key={conn.connector_id} className="rounded-xl border border-[var(--color-border-default)] overflow-hidden">
              {/* Connector header */}
              <div className="flex items-center gap-2 px-3 py-2.5 bg-[var(--color-cyan-500)]/10 text-xs">
                <ShoppingCart className="w-3.5 h-3.5 text-[var(--color-cyan-500)] shrink-0" />
                <div className="flex-1 min-w-0">
                  <span className="text-gray-300 font-medium">
                    <ConnectorTypeLabel type={conn.connector_type} />
                  </span>
                  <p className="text-[10px] text-[var(--color-text-tertiary)] truncate mt-0.5">
                    {conn.connector_id.slice(0, 12)}...
                  </p>
                </div>
                <button
                  onClick={() => setExpandedSyncId(expandedSyncId === conn.connector_id ? null : conn.connector_id)}
                  className="text-[var(--color-text-tertiary)] hover:text-[var(--color-cyan-500)] transition-colors shrink-0"
                  title="Sync data"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => onDisconnect(conn.connector_id)}
                  className="text-[var(--color-text-tertiary)] hover:text-red-400 transition-colors shrink-0"
                  title="Disconnect"
                >
                  <XCircle className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Sync panel (expandable) */}
              {expandedSyncId === conn.connector_id && (
                <div className="border-t border-[var(--color-border-default)] px-3 py-2.5 space-y-2">
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <label className="text-[10px] text-[var(--color-text-tertiary)] block mb-0.5">Start date</label>
                      <input
                        type="date"
                        value={syncDates[conn.connector_id]?.start ?? ""}
                        onChange={e => updateSyncDate(conn.connector_id, "start", e.target.value)}
                        className="w-full px-2 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-[10px] text-[var(--color-text-tertiary)] block mb-0.5">End date</label>
                      <input
                        type="date"
                        value={syncDates[conn.connector_id]?.end ?? ""}
                        onChange={e => updateSyncDate(conn.connector_id, "end", e.target.value)}
                        className="w-full px-2 py-1.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs focus:outline-none focus:border-[var(--color-cyan-500)]/40"
                      />
                    </div>
                  </div>

                  <button
                    onClick={() => handleSync(conn.connector_id)}
                    disabled={
                      syncingId === conn.connector_id ||
                      !syncDates[conn.connector_id]?.start ||
                      !syncDates[conn.connector_id]?.end
                    }
                    className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--color-cyan-500)]/20 hover:bg-[var(--color-cyan-500)]/30 border border-[var(--color-cyan-500)]/30 text-[var(--color-cyan-500)] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {syncingId === conn.connector_id ? (
                      <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Syncing...</>
                    ) : (
                      <><RefreshCw className="w-3.5 h-3.5" /> Sync Data</>
                    )}
                  </button>

                  {/* Sync result */}
                  {syncResult && syncResult.id === conn.connector_id && (
                    <div className="px-2 py-1.5 rounded-lg bg-[var(--color-success-500)]/10 border border-[var(--color-success-500)]/20 text-[10px] text-[var(--color-success-500)]">
                      {syncResult.result.orders_persisted != null && (
                        <span>Orders: {syncResult.result.orders_persisted}</span>
                      )}
                      {syncResult.result.financials_persisted != null && (
                        <span className="ml-2">Financials: {syncResult.result.financials_persisted}</span>
                      )}
                    </div>
                  )}

                  {/* Sync error */}
                  {syncError && syncingId === null && expandedSyncId === conn.connector_id && (
                    <p className="text-[10px] text-red-400 px-1 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3 shrink-0" />
                      {syncError}
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Connect Marketplace button / form */}
      <button
        onClick={() => { setShowForm(v => !v); setRegisterError(null); setRegisterState("idle"); }}
        className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)]/50 hover:border-[var(--color-cyan-500)]/40 hover:bg-[var(--color-cyan-500)]/5 transition-all text-xs text-[var(--color-text-secondary)]"
      >
        <Plus className="w-3.5 h-3.5 text-[var(--color-cyan-500)] shrink-0" />
        <span>Connect Marketplace</span>
        <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${showForm ? "rotate-180" : ""}`} />
      </button>

      {showForm && (
        <div className="space-y-2 pt-2">
          {/* Type selector */}
          <select
            value={selectedType}
            onChange={e => {
              setSelectedType(e.target.value)
              setCredentials({})
              setVisibleFields(new Set())
              setRegisterError(null)
            }}
            className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs focus:outline-none focus:border-[var(--color-cyan-500)]/40"
          >
            {CONNECTOR_TYPES.map(t => (
              <option key={t} value={t}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </option>
            ))}
          </select>

          {/* Dynamic credential fields */}
          {fields.map(field => (
            <div key={field.key} className="relative">
              <label className="text-[10px] text-[var(--color-text-tertiary)] block mb-0.5 px-0.5">{field.label}</label>
              <input
                type={field.type === "password" && !visibleFields.has(field.key) ? "password" : "text"}
                placeholder={field.placeholder}
                value={credentials[field.key] ?? ""}
                onChange={e => setCredentials(prev => ({ ...prev, [field.key]: e.target.value }))}
                className="w-full px-3 py-2 pr-8 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-gray-300 text-xs placeholder-[var(--color-text-quaternary)] focus:outline-none focus:border-[var(--color-cyan-500)]/40"
              />
              {field.type === "password" && (
                <button
                  type="button"
                  onClick={() => toggleFieldVisibility(field.key)}
                  className="absolute right-2 bottom-[7px] text-[var(--color-text-tertiary)] hover:text-gray-300"
                >
                  {visibleFields.has(field.key)
                    ? <EyeOff className="w-3.5 h-3.5" />
                    : <Eye className="w-3.5 h-3.5" />
                  }
                </button>
              )}
            </div>
          ))}

          {registerError && (
            <p className="text-[10px] text-red-400 px-1 flex items-center gap-1">
              <AlertCircle className="w-3 h-3 shrink-0" />
              {registerError}
            </p>
          )}

          <button
            onClick={handleRegister}
            disabled={registerState === "submitting" || !allFieldsFilled}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-cyan-500)]/20 hover:bg-[var(--color-cyan-500)]/30 border border-[var(--color-cyan-500)]/30 text-[var(--color-cyan-500)] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {registerState === "submitting" ? (
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Connecting...</>
            ) : (
              <><ShoppingCart className="w-3.5 h-3.5" /> Test &amp; Connect</>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
