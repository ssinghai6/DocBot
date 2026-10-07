"use client"

import { useEffect, useRef } from "react"
import { Database, FileText } from "lucide-react"
import type { Lineage, LineageSource } from "@/components/types"
import { useUIStore } from "@/store/uiStore"
import { Chip, EmptyTab, ScoreBar, SectionLabel } from "./shared"

interface LegacyCitation {
  source?: string
  page?: number | string
  text?: string
}

function SourceCard({ source, focused }: { source: LineageSource; focused: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" })
  }, [focused])
  const Icon = source.kind === "sql_table" || source.kind === "csv" ? Database : FileText
  return (
    <div
      ref={ref}
      className={`p-3 rounded-[5px] border bg-[var(--color-bg-inset)] transition-colors ${
        focused ? "border-[var(--color-cyan-500)]" : "border-[var(--color-border-subtle)]"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <Icon className="w-3 h-3 flex-none text-[var(--color-cyan-500)]" />
          <span className="text-[12px] font-medium text-[var(--color-text-primary)] truncate">{source.label}</span>
        </div>
        <div className="flex items-center gap-1 flex-none">
          {source.page != null && <Chip>p{source.page}</Chip>}
          <Chip tone={source.cited ? "green" : "neutral"}>{source.cited ? "cited" : "retrieved only"}</Chip>
        </div>
      </div>
      {source.sub_question && (
        <p className="text-[10px] text-[var(--color-text-quaternary)] mt-1">for: {source.sub_question}</p>
      )}
      {source.snippet && (
        <p className="text-[11px] leading-relaxed text-[var(--color-text-secondary)] mt-2 line-clamp-6">{source.snippet}</p>
      )}
      {(source.rerank_score != null || source.score != null) && (
        <div className="mt-2 space-y-1">
          {source.rerank_score != null && <ScoreBar label="Rerank" value={source.rerank_score} />}
          {source.score != null && <ScoreBar label="Vector" value={source.score} />}
        </div>
      )}
    </div>
  )
}

export default function SourcesTab({ lineage }: { lineage: Lineage | null }) {
  const focusSourceId = useUIStore((s) => s.focusSourceId)
  const artifact = useUIStore((s) => s.selectedArtifact)
  const legacy = (artifact?.type === "citations" ? (artifact.payload?.citations as LegacyCitation[] | undefined) : undefined) ?? []

  if (lineage && lineage.sources.length > 0) {
    const cited = lineage.sources.filter((s) => s.cited)
    const uncited = lineage.sources.filter((s) => !s.cited)
    return (
      <div className="p-3 space-y-2">
        <SectionLabel>Sources ({lineage.sources.length})</SectionLabel>
        {[...cited, ...uncited].map((s) => <SourceCard key={s.id} source={s} focused={s.id === focusSourceId} />)}
      </div>
    )
  }

  if (legacy.length > 0) {
    return (
      <div className="p-3 space-y-2">
        <SectionLabel>Citations ({legacy.length})</SectionLabel>
        {legacy.map((c, i) => (
          <div key={i} className="p-3 rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)]">
            <div className="text-[12px] font-medium text-[var(--color-text-primary)] truncate">{c.source || "Source"}</div>
            {c.page !== undefined && c.page !== "" && <div className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5">Page {c.page}</div>}
            {c.text && <p className="text-[11px] leading-relaxed text-[var(--color-text-secondary)] mt-1.5 line-clamp-6">{c.text}</p>}
          </div>
        ))}
      </div>
    )
  }

  return <div className="p-4"><EmptyTab label="sources" hint="Documents, tables and files used by an answer appear here with excerpts and relevance scores." /></div>
}
