"use client"

import { Highlight, themes } from "prism-react-renderer"
import type { Lineage } from "@/components/types"
import { useUIStore } from "@/store/uiStore"
import { Chip, CopyButton, EmptyTab, SectionLabel, fmtMs, fmtNum } from "./shared"

export default function QueryTab({ lineage }: { lineage: Lineage | null }) {
  const artifact = useUIStore((s) => s.selectedArtifact)
  const fromCard = artifact?.type === "sql" ? String(artifact.payload?.sql ?? "") : ""
  const explanation = artifact?.type === "sql" ? String(artifact.payload?.explanation ?? "") : ""
  const sqlInfo = lineage?.sql ?? null
  const sql = fromCard || sqlInfo?.sql || ""

  if (!sql) {
    return (
      <div className="p-4">
        <EmptyTab label="query" hint="SQL generated for an answer, with the tables it used and run stats, shows up here." />
      </div>
    )
  }

  const considered = sqlInfo?.tables_considered ?? []
  const selected = new Set(sqlInfo?.tables_selected ?? [])
  const skipped = considered.filter((t) => !selected.has(t))

  return (
    <div className="p-3 space-y-3">
      <SectionLabel>SQL</SectionLabel>
      <Highlight code={sql} language="sql" theme={themes.vsDark}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`${className} text-[12px] leading-[1.55] p-3 rounded-[5px] overflow-x-auto border border-[var(--color-border-subtle)]`}
            style={{ ...style, background: "var(--color-bg-inset)", fontFamily: "var(--font-jetbrains-mono), monospace" }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                <span className="inline-block w-6 select-none text-[var(--color-text-quaternary)] pr-2 text-right">{i + 1}</span>
                {line.map((token, key) => <span key={key} {...getTokenProps({ token })} />)}
              </div>
            ))}
          </pre>
        )}
      </Highlight>

      {sqlInfo && (
        <div className="flex flex-wrap gap-1.5">
          {sqlInfo.row_count != null && <Chip>{fmtNum(sqlInfo.row_count, 0)} rows</Chip>}
          {sqlInfo.execution_time_ms != null && <Chip>{fmtMs(sqlInfo.execution_time_ms)}</Chip>}
          {sqlInfo.drift_retry && <Chip tone="amber" title="Schema changed; refreshed and retried once">schema drift retry</Chip>}
          <Chip tone="green" title="Validated by sqlglot AST, read-only transaction">read-only</Chip>
        </div>
      )}

      {sqlInfo && sqlInfo.tables_selected.length > 0 && (
        <div className="space-y-1.5">
          <SectionLabel>Tables used</SectionLabel>
          <div className="flex flex-wrap gap-1">
            {sqlInfo.tables_selected.map((t) => <Chip key={t} tone="cyan">{t}</Chip>)}
          </div>
          {skipped.length > 0 && (
            <>
              <div className="text-[10px] text-[var(--color-text-quaternary)] pt-1">Considered, not used ({skipped.length})</div>
              <div className="flex flex-wrap gap-1">
                {skipped.slice(0, 12).map((t) => <Chip key={t}>{t}</Chip>)}
                {skipped.length > 12 && <span className="text-[10px] text-[var(--color-text-quaternary)]">+{skipped.length - 12}</span>}
              </div>
            </>
          )}
        </div>
      )}

      {explanation && (
        <>
          <SectionLabel>Explanation</SectionLabel>
          <p className="text-[12px] leading-relaxed text-[var(--color-text-secondary)] whitespace-pre-wrap">{explanation}</p>
        </>
      )}

      <div className="pt-1"><CopyButton text={sql} label="Copy SQL" /></div>
    </div>
  )
}
