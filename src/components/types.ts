// Shared TypeScript types extracted from page.tsx
// Used by multiple components to avoid duplication

import { z } from "zod"

// ── Zod schemas ────────────────────────────────────────────────────────────

export const AuthMeSchema = z.object({
  id: z.string(),
  email: z.string(),
  name: z.string(),
  role: z.enum(["viewer", "analyst", "admin"]),
  provider: z.string(),
})
export type AuthUser = z.infer<typeof AuthMeSchema>

export const AdminUserSchema = z.object({
  id: z.string(),
  email: z.string(),
  name: z.string(),
  role: z.enum(["viewer", "analyst", "admin"]),
  provider: z.string(),
  last_login_at: z.string().nullable(),
  created_at: z.string().nullable(),
})
export type AdminUser = z.infer<typeof AdminUserSchema>

export const AdminUsersResponseSchema = z.object({
  count: z.number(),
  users: z.array(AdminUserSchema),
})

export const AuditEventSchema = z.object({
  id: z.string(),
  event_type: z.string(),
  session_id: z.string().nullable(),
  user_id: z.string().nullable(),
  detail: z.string().nullable(),
  metadata_json: z.string().nullable(),
  occurred_at: z.string().nullable(),
})
export type AuditEvent = z.infer<typeof AuditEventSchema>

export const AuditLogResponseSchema = z.object({
  count: z.number(),
  events: z.array(AuditEventSchema),
})

// ── Workspace schemas ─────────────────────────────────────────────────────

export const WorkspaceSessionSchema = z.object({
  session_id: z.string(),
  created_at: z.string().nullable(),
  file_count: z.number(),
  persona: z.string(),
})
export type WorkspaceSession = z.infer<typeof WorkspaceSessionSchema>

export const WorkspaceConnectionSchema = z.object({
  id: z.string(),
  dialect: z.string(),
  host: z.string(),
  db_name: z.string(),
  created_at: z.string().nullable(),
})
export type WorkspaceConnection = z.infer<typeof WorkspaceConnectionSchema>

export const WorkspaceSchema = z.object({
  sessions: z.array(WorkspaceSessionSchema),
  db_connections: z.array(WorkspaceConnectionSchema),
})

// ── Domain types ──────────────────────────────────────────────────────────

export type Citation = {
  source: string
  page: number
  text: string
}

export type ChartMeta = {
  type: string
  title: string
  x_label: string
  y_label: string
  series_count: number
}

export type Toast = {
  id: string
  type: "success" | "error" | "info" | "warning"
  message: string
}

export type FileUploadState = "idle" | "dragover" | "uploading" | "success" | "error"

// DOCBOT-504: Query History
export type QueryHistoryItem = {
  id: string
  question: string
  sql: string
  executed_at: string | null
  row_count: number | null
}

// DOCBOT-405: Autopilot step result from SSE stream
export type AutopilotStep = {
  step_num: number
  tool: string
  step_label: string
  content: string
  artifact_id?: string | null
  chart_b64?: string | null
  error?: string | null
}

export type Message = {
  role: "user" | "assistant"
  content: string
  timestamp?: Date
  citations?: Citation[]
  charts?: string[]           // base64 PNG strings from E2B analysis
  chartMetas?: ChartMeta[]    // DOCBOT-305: metadata per chart
  analysisCode?: string       // Python code block, collapsible
  sql?: string                // SQL query from metadata chunk
  explanation?: string        // SQL explanation from metadata chunk
  autopilotSteps?: AutopilotStep[]  // DOCBOT-405: persisted investigation steps
  agentPersona?: string       // DOCBOT-802: which persona handled this message
  agentPersonas?: string[]    // DOCBOT-802: for hybrid messages with multiple personas
}

// ── Connector schemas ────────────────────────────────────────────────────

export const ConnectorInfoSchema = z.object({
  connector_id: z.string(),
  connector_type: z.string(),
})
export type ConnectorInfo = z.infer<typeof ConnectorInfoSchema>

export const ConnectorListResponseSchema = z.object({
  connectors: z.array(ConnectorInfoSchema),
})

export const ConnectorSyncResponseSchema = z.object({
  orders_persisted: z.number().optional(),
  financials_persisted: z.number().optional(),
})
export type ConnectorSyncResponse = z.infer<typeof ConnectorSyncResponseSchema>

// Live DB connection form state shape
export type LiveDbForm = {
  dialect: string
  host: string
  port: string
  dbname: string
  user: string
  password: string
  pii_masking_enabled: boolean
}
