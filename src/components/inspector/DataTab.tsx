"use client"

import { useMemo } from "react"
import { Download } from "lucide-react"
import type { Lineage } from "@/components/types"
import { useUIStore } from "@/store/uiStore"
import { EmptyTab, SectionLabel } from "./shared"

function cell(value: unknown): string {
  if (value == null) return ""
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

function ResultTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const columns = useMemo(() => (rows[0] ? Object.keys(rows[0]) : []), [rows])
  if (columns.length === 0) return null
  return (
    <div className="overflow-x-auto rounded-[5px] border border-[var(--color-border-subtle)]">
      <table className="w-full text-[11px]">
        <thead className="bg-[var(--color-bg-inset)]">
          <tr>
            {columns.map((c) => (
              <th key={c} className="text-left font-semibold px-2 py-1.5 text-[var(--color-text-tertiary)] whitespace-nowrap">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-[var(--color-border-subtle)]">
              {columns.map((c) => (
                <td key={c} className="px-2 py-1 text-[var(--color-text-secondary)] whitespace-nowrap tabular-nums">{cell(r[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function DataTab({ lineage }: { lineage: Lineage | null }) {
  const artifact = useUIStore((s) => s.selectedArtifact)
  const b64 = artifact?.type === "chart" ? String(artifact.payload?.b64 ?? "") : ""
  const meta = (artifact?.type === "chart" ? (artifact.payload?.meta as Record<string, unknown> | undefined) : undefined) ?? undefined
  const dataUrl = useMemo(() => (b64 ? `data:image/png;base64,${b64}` : ""), [b64])
  const rows = lineage?.sql?.result_preview ?? []

  if (!dataUrl && rows.length === 0) {
    return <div className="p-4"><EmptyTab label="data" hint="Result rows and charts from an answer appear here. Click a chart in the chat to view it full-size." /></div>
  }

  const title = (meta?.title as string) || "Chart"
  const xLabel = (meta?.x_label as string) || ""
  const yLabel = (meta?.y_label as string) || ""

  const download = () => {
    const a = document.createElement("a")
    a.href = dataUrl
    a.download = `${title.replace(/[^a-z0-9-_]+/gi, "_").toLowerCase() || "chart"}.png`
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  return (
    <div className="p-3 space-y-4">
      {dataUrl && (
        <div className="space-y-3">
          <div className="flex items-start justify-between gap-2">
            <div className="text-[13px] font-semibold text-[var(--color-text-primary)] truncate">{title}</div>
            <button
              onClick={download}
              className="flex-none flex items-center gap-1.5 text-[11px] px-2.5 h-7 rounded-[5px] border border-[var(--color-border-default)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-cyan-500)] transition-colors"
            >
              <Download className="w-3 h-3" />
              PNG
            </button>
          </div>
          <div className="rounded-[5px] border border-[var(--color-border-subtle)] bg-white overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={dataUrl} alt={title} className="w-full h-auto block" />
          </div>
          {(xLabel || yLabel) && (
            <div className="text-[11px] text-[var(--color-text-secondary)]">
              {xLabel && <span>X: {xLabel}</span>}
              {xLabel && yLabel && <span className="mx-2 text-[var(--color-text-quaternary)]">·</span>}
              {yLabel && <span>Y: {yLabel}</span>}
            </div>
          )}
        </div>
      )}

      {rows.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Result preview ({rows.length}{lineage?.sql?.row_count && lineage.sql.row_count > rows.length ? ` of ${lineage.sql.row_count}` : ""} rows)</SectionLabel>
          <ResultTable rows={rows} />
          {lineage?.pii.masked && <p className="text-[10px] text-[var(--color-text-quaternary)]">Sensitive values are masked.</p>}
        </div>
      )}
    </div>
  )
}
