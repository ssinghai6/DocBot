"use client"

import { AlertTriangle, Database, FileText, GitBranch, Layers, MessageSquare, Search, Sparkles } from "lucide-react"
import type { Lineage, LineageSource, LineageStep } from "@/components/types"
import { useUIStore } from "@/store/uiStore"
import { Chip, EmptyTab, SectionLabel, fmtMs, fmtNum } from "./shared"

const STATUS_DOT: Record<LineageStep["status"], string> = {
  ok: "bg-emerald-400",
  retried: "bg-[var(--color-amber-500)]",
  error: "bg-red-400",
  skipped: "bg-[var(--color-text-quaternary)]",
}

function prettyName(name: string): string {
  return name.replace(/_/g, " ")
}

function StepRow({ step, maxMs }: { step: LineageStep; maxMs: number }) {
  const pct = step.latency_ms && maxMs > 0 ? Math.max(4, (step.latency_ms / maxMs) * 100) : 0
  return (
    <li className="relative pl-5 pb-3 last:pb-0">
      <span className="absolute left-[3px] top-0 bottom-0 w-px bg-[var(--color-border-subtle)]" aria-hidden />
      <span className={`absolute left-0 top-[5px] w-[7px] h-[7px] rounded-full ${STATUS_DOT[step.status]}`} aria-hidden />
      <div className="flex items-center justify-between gap-2">
        <span className="text-[12px] font-medium text-[var(--color-text-primary)] capitalize truncate">{prettyName(step.name)}</span>
        <span className="text-[10px] tabular-nums text-[var(--color-text-tertiary)] flex-none">{fmtMs(step.latency_ms)}</span>
      </div>
      <div className="flex flex-wrap items-center gap-1 mt-0.5">
        {step.tool && <Chip>{step.tool}</Chip>}
        {step.status !== "ok" && <Chip tone={step.status === "error" ? "red" : "amber"}>{step.status}</Chip>}
      </div>
      {step.detail && <p className="text-[11px] text-[var(--color-text-tertiary)] mt-1 leading-snug break-words">{step.detail}</p>}
      {pct > 0 && (
        <div className="mt-1.5 h-[3px] rounded-full bg-[var(--color-border-subtle)] overflow-hidden">
          <div className="h-full bg-[var(--color-cyan-500)]/70" style={{ width: `${pct}%` }} />
        </div>
      )}
    </li>
  )
}

function SourceLink({ source, lineage }: { source: LineageSource; lineage: Lineage }) {
  const selectLineage = useUIStore((s) => s.selectLineage)
  const Icon = source.kind === "sql_table" || source.kind === "csv" ? Database : FileText
  return (
    <button
      type="button"
      onClick={() => selectLineage(lineage, { tab: "sources", focusSourceId: source.id })}
      className="flex items-center gap-1.5 w-full text-left text-[11px] px-2 h-6 rounded-[3px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] hover:border-[var(--color-cyan-500)]/50 transition-colors"
      title={source.snippet ?? source.label}
    >
      <Icon className="w-3 h-3 flex-none text-[var(--color-cyan-500)]" />
      <span className="truncate text-[var(--color-text-secondary)]">{source.label}</span>
      {source.page != null && <span className="text-[var(--color-text-quaternary)] tabular-nums flex-none">p{source.page}</span>}
      {!source.cited && <span className="ml-auto text-[9px] uppercase text-[var(--color-text-quaternary)] flex-none">unused</span>}
    </button>
  )
}

function Lane({
  title, tone, icon, steps, sources, lineage, maxMs,
}: {
  title: string
  tone: "cyan" | "amber"
  icon: React.ReactNode
  steps: LineageStep[]
  sources: LineageSource[]
  lineage: Lineage
  maxMs: number
}) {
  const color = tone === "cyan" ? "var(--color-cyan-500)" : "var(--color-amber-500)"
  return (
    <div className="rounded-[5px] border bg-[var(--color-bg-inset)] p-3" style={{ borderColor: `color-mix(in srgb, ${color} 35%, transparent)` }}>
      <div className="flex items-center gap-1.5 mb-2" style={{ color }}>
        {icon}
        <span className="text-[10px] font-semibold uppercase tracking-wider">{title}</span>
      </div>
      <ul>{steps.map((s, i) => <StepRow key={`${s.name}-${i}`} step={s} maxMs={maxMs} />)}</ul>
      {sources.length > 0 && (
        <div className="mt-2 space-y-1">
          {sources.slice(0, 6).map((s) => <SourceLink key={s.id} source={s} lineage={lineage} />)}
          {sources.length > 6 && <div className="text-[10px] text-[var(--color-text-quaternary)] pl-1">+{sources.length - 6} more in Sources</div>}
        </div>
      )}
    </div>
  )
}

function Node({ icon, title, children }: { icon: React.ReactNode; title: string; children?: React.ReactNode }) {
  return (
    <div className="rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] p-3">
      <div className="flex items-center gap-1.5 mb-1 text-[var(--color-text-tertiary)]">
        {icon}
        <span className="text-[10px] font-semibold uppercase tracking-wider">{title}</span>
      </div>
      {children}
    </div>
  )
}

function Connector({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 pl-4 h-5" aria-hidden>
      <span className="w-px h-full bg-[var(--color-border-default)]" />
      {label && <span className="text-[9px] uppercase tracking-wider text-[var(--color-text-quaternary)]">{label}</span>}
    </div>
  )
}

export default function LineageTab({ lineage }: { lineage: Lineage | null }) {
  if (!lineage) {
    return (
      <div className="p-4">
        <EmptyTab label="lineage" hint="Ask a question. Each answer shows its sources, pipeline steps and checks here." />
      </div>
    )
  }

  const maxMs = Math.max(0, ...lineage.steps.map((s) => s.latency_ms ?? 0))
  const docSteps = lineage.steps.filter((s) => s.lane === "docs")
  const dbSteps = lineage.steps.filter((s) => s.lane === "db")
  const otherSteps = lineage.steps.filter((s) => !s.lane)
  const dualLane = docSteps.length > 0 && dbSteps.length > 0
  const docSources = lineage.sources.filter((s) => s.kind === "pdf" || s.kind === "edgar")
  const dbSources = lineage.sources.filter((s) => s.kind === "sql_table" || s.kind === "csv")
  const preSteps = otherSteps.filter((s) => ["classify_intent", "schema_retrieval", "table_selection", "few_shot_retrieval", "sql_generation", "sql_validation"].includes(s.name) || s.tool === "intent_classifier")
  const postSteps = otherSteps.filter((s) => !preSteps.includes(s))
  const rewritten = lineage.standalone_query && lineage.standalone_query !== lineage.question

  return (
    <div className="p-3 space-y-0">
      <div className="flex flex-wrap items-center gap-1.5 mb-3">
        <Chip tone="cyan">{lineage.mode}</Chip>
        {lineage.intent && <Chip>intent: {lineage.intent}</Chip>}
        {lineage.persona && <Chip>{lineage.persona}</Chip>}
        {lineage.cache_hit && <Chip tone="green">cache hit</Chip>}
        <Chip>{fmtMs(lineage.total_latency_ms)} total</Chip>
      </div>

      <Node icon={<MessageSquare className="w-3 h-3" />} title="Question">
        <p className="text-[12px] text-[var(--color-text-primary)] leading-snug break-words">{lineage.question ?? "–"}</p>
        {rewritten && (
          <p className="text-[11px] text-[var(--color-text-tertiary)] mt-1.5 leading-snug">
            <span className="uppercase text-[9px] tracking-wider mr-1">Rewritten</span>{lineage.standalone_query}
          </p>
        )}
        {lineage.expanded_queries.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {lineage.expanded_queries.map((q, i) => (
              <Chip key={i} title="Query variant searched"><Search className="w-2.5 h-2.5" />{q.length > 36 ? `${q.slice(0, 36)}…` : q}</Chip>
            ))}
          </div>
        )}
      </Node>

      {preSteps.length > 0 && (
        <>
          <Connector />
          <div className="rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] p-3">
            <ul>{preSteps.map((s, i) => <StepRow key={`${s.name}-${i}`} step={s} maxMs={maxMs} />)}</ul>
          </div>
        </>
      )}

      {lineage.sub_questions.length > 0 && (
        <>
          <Connector label="decomposed" />
          <Node icon={<Layers className="w-3 h-3" />} title={`Sub-questions (${lineage.sub_questions.length})`}>
            <ol className="list-decimal pl-4 space-y-0.5">
              {lineage.sub_questions.map((q, i) => <li key={i} className="text-[11px] text-[var(--color-text-secondary)] leading-snug">{q}</li>)}
            </ol>
          </Node>
        </>
      )}

      <Connector label={dualLane ? "parallel" : "retrieve"} />
      {dualLane ? (
        <div className="space-y-2">
          <Lane title="Documents lane" tone="cyan" icon={<FileText className="w-3 h-3" />} steps={docSteps} sources={docSources} lineage={lineage} maxMs={maxMs} />
          <Lane title="Database lane" tone="amber" icon={<Database className="w-3 h-3" />} steps={dbSteps} sources={dbSources} lineage={lineage} maxMs={maxMs} />
        </div>
      ) : (
        <div className="rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] p-3 space-y-2">
          {[...docSteps, ...dbSteps].length > 0 && (
            <ul>{[...docSteps, ...dbSteps].map((s, i) => <StepRow key={`${s.name}-${i}`} step={s} maxMs={maxMs} />)}</ul>
          )}
          {lineage.sources.length > 0 ? (
            <div className="space-y-1">
              {lineage.sources.slice(0, 8).map((s) => <SourceLink key={s.id} source={s} lineage={lineage} />)}
              {lineage.sources.length > 8 && <div className="text-[10px] text-[var(--color-text-quaternary)] pl-1">+{lineage.sources.length - 8} more in Sources</div>}
            </div>
          ) : (
            [...docSteps, ...dbSteps].length === 0 && <p className="text-[11px] text-[var(--color-text-tertiary)]">No external sources were used.</p>
          )}
        </div>
      )}

      {lineage.discrepancies.length > 0 && (
        <>
          <Connector label="cross-check" />
          <div className="rounded-[5px] border border-[var(--color-amber-500)]/50 bg-[var(--color-amber-500)]/5 p-3">
            <div className="flex items-center gap-1.5 mb-2 text-[var(--color-amber-500)]">
              <AlertTriangle className="w-3 h-3" />
              <span className="text-[10px] font-semibold uppercase tracking-wider">
                {lineage.discrepancies.length} discrepanc{lineage.discrepancies.length === 1 ? "y" : "ies"} between docs and data
              </span>
            </div>
            <div className="space-y-2">
              {lineage.discrepancies.map((d, i) => (
                <div key={i} className="text-[11px]">
                  <div className="font-medium text-[var(--color-text-primary)]">{d.label}</div>
                  <div className="grid grid-cols-3 gap-2 mt-0.5 text-[var(--color-text-secondary)] tabular-nums">
                    <span><span className="text-[9px] uppercase text-[var(--color-text-quaternary)] block">Doc</span>{fmtNum(d.doc_value)}</span>
                    <span><span className="text-[9px] uppercase text-[var(--color-text-quaternary)] block">DB</span>{fmtNum(d.db_value)}</span>
                    <span><span className="text-[9px] uppercase text-[var(--color-text-quaternary)] block">Delta</span>{fmtNum(d.delta)}{d.pct != null ? ` (${d.pct.toFixed(1)}%)` : ""}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {postSteps.length > 0 && (
        <>
          <Connector label="synthesize" />
          <div className="rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)] p-3">
            <ul>{postSteps.map((s, i) => <StepRow key={`${s.name}-${i}`} step={s} maxMs={maxMs} />)}</ul>
          </div>
        </>
      )}

      <Connector />
      <Node icon={<Sparkles className="w-3 h-3" />} title="Answer">
        <div className="flex flex-wrap gap-1">
          <Chip>{lineage.sources.length} sources</Chip>
          <Chip>{lineage.steps.length} steps</Chip>
          {lineage.pii.masked && <Chip tone="green">PII masked</Chip>}
        </div>
      </Node>

      <div className="pt-3 flex items-center gap-1.5 text-[10px] text-[var(--color-text-quaternary)]">
        <GitBranch className="w-3 h-3" />
        <SectionLabel>run {lineage.run_id.slice(0, 8)}</SectionLabel>
      </div>
    </div>
  )
}
