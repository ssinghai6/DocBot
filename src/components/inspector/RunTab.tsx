"use client"

import { useEffect, useMemo, useState } from "react"
import { LineageSchema } from "@/components/types"
import type { Lineage, LineageModelCall } from "@/components/types"
import { Chip, CopyButton, EmptyTab, SectionLabel, fmtMs, fmtNum, fmtUsd } from "./shared"

/** Fetch the persisted lineage, which includes llm_calls joined by run_id. */
function useModelCalls(lineage: Lineage | null): { calls: LineageModelCall[]; loading: boolean } {
  const runId = lineage?.run_id ?? null
  const inline = lineage?.model_calls
  const [fetched, setFetched] = useState<{ runId: string; calls: LineageModelCall[] } | null>(null)

  useEffect(() => {
    if (!runId || (inline && inline.length > 0)) return
    const controller = new AbortController()
    // llm_calls rows are written by a background queue; give it a moment.
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/lineage/${encodeURIComponent(runId)}`, { signal: controller.signal })
        if (!res.ok) return
        const parsed = LineageSchema.safeParse(await res.json())
        if (parsed.success) setFetched({ runId, calls: parsed.data.model_calls })
      } catch {
        // best-effort; the tab renders without model rows
      }
    }, 1200)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [runId, inline])

  if (inline && inline.length > 0) return { calls: inline, loading: false }
  if (fetched && fetched.runId === runId) return { calls: fetched.calls, loading: false }
  return { calls: [], loading: !!runId }
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] px-3 py-2">
      <div className="text-[9px] uppercase tracking-wider text-[var(--color-text-quaternary)]">{label}</div>
      <div className="text-[13px] font-semibold text-[var(--color-text-primary)] tabular-nums mt-0.5">{value}</div>
    </div>
  )
}

export default function RunTab({ lineage }: { lineage: Lineage | null }) {
  const { calls, loading } = useModelCalls(lineage)

  const totals = useMemo(() => {
    const input = calls.reduce((a, c) => a + (c.input_tokens ?? 0), 0)
    const output = calls.reduce((a, c) => a + (c.output_tokens ?? 0), 0)
    const cost = calls.reduce((a, c) => a + (c.estimated_cost_usd ?? 0), 0)
    const fallbacks = calls.filter((c) => c.fallback_triggered).length
    return { input, output, cost, fallbacks }
  }, [calls])

  if (!lineage) {
    return <div className="p-4"><EmptyTab label="run" hint="Model, tokens, cost and timing for an answer appear here." /></div>
  }

  const maxStep = Math.max(1, ...lineage.steps.map((s) => s.latency_ms ?? 0))
  const exportJson = () => JSON.stringify({ ...lineage, model_calls: calls }, null, 2)

  return (
    <div className="p-3 space-y-4">
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Total time" value={fmtMs(lineage.total_latency_ms)} />
        <Stat label="LLM calls" value={calls.length ? String(calls.length) : loading ? "…" : "–"} />
        <Stat label="Tokens in / out" value={calls.length ? `${fmtNum(totals.input, 0)} / ${fmtNum(totals.output, 0)}` : "–"} />
        <Stat label="Est. cost" value={calls.length ? fmtUsd(totals.cost) : "–"} />
      </div>

      <div className="flex flex-wrap gap-1.5">
        {lineage.pii.masked && <Chip tone="green" title="PII patterns masked in output">PII masked</Chip>}
        {Object.entries(lineage.pii.counts).map(([k, v]) => <Chip key={k}>{k}: {v}</Chip>)}
        {lineage.cache_hit && <Chip tone="green">cache hit</Chip>}
        {totals.fallbacks > 0 && <Chip tone="amber" title="Primary provider failed; fallback used">{totals.fallbacks} fallback</Chip>}
      </div>

      <div className="space-y-2">
        <SectionLabel>Models</SectionLabel>
        {calls.length === 0 ? (
          <p className="text-[11px] text-[var(--color-text-tertiary)]">
            {loading ? "Loading model records…" : "No LLM call records for this answer."}
          </p>
        ) : (
          <div className="space-y-1.5">
            {calls.map((c, i) => (
              <div key={i} className="rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[12px] font-medium text-[var(--color-text-primary)] truncate">{c.model}</span>
                  <span className="text-[10px] tabular-nums text-[var(--color-text-tertiary)] flex-none">{fmtMs(c.latency_ms)}</span>
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <Chip>{c.provider}</Chip>
                  {c.caller && <Chip>{c.caller}</Chip>}
                  {!c.success && <Chip tone="red">failed</Chip>}
                  {c.fallback_triggered && <Chip tone="amber">fallback</Chip>}
                  {(c.input_tokens != null || c.output_tokens != null) && (
                    <Chip>{fmtNum(c.input_tokens, 0)} in / {fmtNum(c.output_tokens, 0)} out</Chip>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {lineage.steps.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Timing</SectionLabel>
          <div className="space-y-1.5">
            {lineage.steps.map((s, i) => (
              <div key={i} className="grid grid-cols-[110px_1fr_44px] items-center gap-2 text-[11px]">
                <span className="truncate capitalize text-[var(--color-text-secondary)]">{s.name.replace(/_/g, " ")}</span>
                <div className="h-1.5 rounded-full bg-[var(--color-border-subtle)] overflow-hidden">
                  <div
                    className={s.status === "error" ? "h-full bg-red-400" : "h-full bg-[var(--color-cyan-500)]"}
                    style={{ width: `${s.latency_ms ? Math.max(3, (s.latency_ms / maxStep) * 100) : 0}%` }}
                  />
                </div>
                <span className="text-right tabular-nums text-[var(--color-text-tertiary)]">{fmtMs(s.latency_ms)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <CopyButton text={lineage.run_id} label="Copy run ID" />
        <CopyButton text={exportJson()} label="Copy JSON" />
      </div>
      <p className="text-[10px] text-[var(--color-text-quaternary)] break-all">run_id {lineage.run_id}</p>
    </div>
  )
}
