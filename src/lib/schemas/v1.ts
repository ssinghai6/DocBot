// Public /v1 API contracts (Zod) — DOCBOT-3901, EPIC-39.
//
// Mirrors api/v1_models.py field-for-field (same snake_case names).
// tests/unit/test_v1_models.py fails if a Pydantic field is missing here.
//
// Response schemas use z.object (unknown keys stripped) so additive backend
// changes don't break old clients. Loosely-typed pipeline payloads
// (citations, plan steps, connector rows, sync summaries) are records.
// Lineage (GET /v1/lineage/{run_id}) reuses the shared Lineage model and is
// typed loosely here; the OpenAPI spec (GET /v1/openapi.json) has the full shape.

import { z } from "zod"

const looseRecord = z.record(z.string(), z.unknown())

export const V1ChatMessageSchema = z.object({
  role: z.string(),
  content: z.string(),
})

export const V1HybridChatRequestSchema = z.object({
  question: z.string(),
  session_id: z.string(),
  connection_id: z.string().nullable().optional(),
  persona: z.string().optional(),
  has_docs: z.boolean().optional(),
  history: z.array(V1ChatMessageSchema).optional(),
  chart_type: z.string().optional(),
})

export const V1HybridChatResponseSchema = z.object({
  answer: z.string(),
  citations: z.array(looseRecord).nullable().optional(),
  intent: z.string().nullable().optional(),
  has_sql: z.boolean().nullable().optional(),
  has_docs: z.boolean().nullable().optional(),
  run_id: z.string().nullable().optional(),
  trace_id: z.string().nullable().optional(),
})

export const V1AutopilotRequestSchema = z.object({
  session_id: z.string(),
  connection_id: z.string().nullable().optional(),
  question: z.string(),
  persona: z.string().optional(),
  has_docs: z.boolean().optional(),
  has_db: z.boolean().optional(),
  has_csv: z.boolean().optional(),
  history: z.array(V1ChatMessageSchema).optional(),
})

export const V1AutopilotResponseSchema = z.object({
  plan: z.array(looseRecord).optional(),
  steps: z.array(looseRecord).optional(),
  answer: z.string().optional(),
  citations: z.array(looseRecord).nullable().optional(),
  warnings: z.array(z.string()).optional(),
})

export const V1ConnectorRegisterRequestSchema = z.object({
  connector_type: z.string(),
  credentials: z.record(z.string(), z.string()),
})

export const V1ConnectorRegisterResponseSchema = z.object({
  connector_id: z.string(),
  connector_type: z.string(),
  status: z.string(),
})

export const V1ConnectorListResponseSchema = z.object({
  connectors: z.array(looseRecord),
})

export const V1ConnectorSyncResponseSchema = looseRecord

export const V1LineageSchema = z.looseObject({
  run_id: z.string(),
})

export const ErrorTypeSchema = z.enum([
  "invalid_request",
  "authentication",
  "permission",
  "not_found",
  "rate_limit",
  "server",
])

export const V1ErrorSchema = z.object({
  type: ErrorTypeSchema,
  code: z.string(),
  message: z.string(),
  request_id: z.string(),
})

export const V1ErrorResponseSchema = z.object({
  error: V1ErrorSchema,
})

export type V1HybridChatRequest = z.infer<typeof V1HybridChatRequestSchema>
export type V1HybridChatResponse = z.infer<typeof V1HybridChatResponseSchema>
export type V1AutopilotRequest = z.infer<typeof V1AutopilotRequestSchema>
export type V1AutopilotResponse = z.infer<typeof V1AutopilotResponseSchema>
export type V1ConnectorRegisterRequest = z.infer<typeof V1ConnectorRegisterRequestSchema>
export type V1ConnectorRegisterResponse = z.infer<typeof V1ConnectorRegisterResponseSchema>
export type V1ConnectorListResponse = z.infer<typeof V1ConnectorListResponseSchema>
export type V1ErrorResponse = z.infer<typeof V1ErrorResponseSchema>
