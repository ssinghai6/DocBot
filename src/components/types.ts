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

// DOCBOT-1510: per-answer lineage trace (mirrors api/utils/lineage.py::Lineage)
export const LineageStepSchema = z.object({
  name: z.string(),
  tool: z.string().nullish(),
  status: z.enum(["ok", "error", "skipped", "retried"]).default("ok"),
  latency_ms: z.number().nullish(),
  detail: z.string().nullish(),
  lane: z.enum(["docs", "db"]).nullish(),
})
export type LineageStep = z.infer<typeof LineageStepSchema>

export const LineageSourceSchema = z.object({
  id: z.string(),
  kind: z.enum(["pdf", "sql_table", "csv", "edgar", "other"]).default("other"),
  label: z.string(),
  page: z.number().nullish(),
  snippet: z.string().nullish(),
  score: z.number().nullish(),
  rerank_score: z.number().nullish(),
  cited: z.boolean().default(true),
  step_num: z.number().nullish(),
  sub_question: z.string().nullish(),
})
export type LineageSource = z.infer<typeof LineageSourceSchema>

export const LineageSqlSchema = z.object({
  sql: z.string().nullish(),
  tables_selected: z.array(z.string()).default([]),
  tables_considered: z.array(z.string()).default([]),
  row_count: z.number().nullish(),
  execution_time_ms: z.number().nullish(),
  drift_retry: z.boolean().default(false),
  result_preview: z.array(z.record(z.string(), z.unknown())).default([]),
})
export type LineageSql = z.infer<typeof LineageSqlSchema>

export const LineageDiscrepancySchema = z.object({
  label: z.string(),
  doc_value: z.number().nullish(),
  db_value: z.number().nullish(),
  delta: z.number().nullish(),
  pct: z.number().nullish(),
})
export type LineageDiscrepancy = z.infer<typeof LineageDiscrepancySchema>

export const LineageModelCallSchema = z.object({
  caller: z.string().nullish(),
  provider: z.string(),
  model: z.string(),
  latency_ms: z.number().nullish(),
  input_tokens: z.number().nullish(),
  output_tokens: z.number().nullish(),
  estimated_cost_usd: z.number().nullish(),
  fallback_triggered: z.boolean().default(false),
  success: z.boolean().default(true),
})
export type LineageModelCall = z.infer<typeof LineageModelCallSchema>

export const LineageSchema = z.object({
  run_id: z.string(),
  mode: z.enum(["docs", "db", "csv", "hybrid", "autopilot"]),
  intent: z.string().nullish(),
  question: z.string().nullish(),
  standalone_query: z.string().nullish(),
  expanded_queries: z.array(z.string()).default([]),
  sub_questions: z.array(z.string()).default([]),
  steps: z.array(LineageStepSchema).default([]),
  sources: z.array(LineageSourceSchema).default([]),
  sql: LineageSqlSchema.nullish(),
  discrepancies: z.array(LineageDiscrepancySchema).default([]),
  model_calls: z.array(LineageModelCallSchema).default([]),
  pii: z.object({ masked: z.boolean().default(false), counts: z.record(z.string(), z.number()).default({}) }).default({ masked: false, counts: {} }),
  cache_hit: z.boolean().default(false),
  persona: z.string().nullish(),
  total_latency_ms: z.number().nullish(),
})
export type Lineage = z.infer<typeof LineageSchema>

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
  sql?: string | null
  explanation?: string | null
  code?: string | null        // generated Python/pandas code executed in E2B
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
  lineage?: Lineage           // DOCBOT-1510: provenance for this answer
  traceId?: string            // DOCBOT-1512: agent trace id for feedback submission
}

// DOCBOT-1512: feedback POST response
export const TraceFeedbackResponseSchema = z.object({
  status: z.string(),
  trace_id: z.string(),
  feedback: z.enum(["up", "down"]),
})
export type TraceFeedbackResponse = z.infer<typeof TraceFeedbackResponseSchema>

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
