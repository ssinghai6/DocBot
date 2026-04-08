"use client"

import { useMemo } from "react"
import { Highlight, themes } from "prism-react-renderer"
import { PanelRight, Database, Table as TableIcon, FileText, Image as ImageIcon, Code2, Download, ExternalLink } from "lucide-react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui"
import { useUIStore, type InspectorTab } from "@/store/uiStore"

interface InspectorPanelProps {
  onClose?: () => void
}

const TABS: Array<{ id: InspectorTab; label: string; icon: React.ReactNode }> = [
  { id: "query",    label: "Query",    icon: <Code2 className="w-3 h-3" /> },
  { id: "schema",   label: "Schema",   icon: <Database className="w-3 h-3" /> },
  { id: "profile",  label: "Profile",  icon: <TableIcon className="w-3 h-3" /> },
  { id: "metadata", label: "Metadata", icon: <FileText className="w-3 h-3" /> },
  { id: "artifact", label: "Artifact", icon: <ImageIcon className="w-3 h-3" /> },
]

export default function InspectorPanel({ onClose }: InspectorPanelProps) {
  const inspectorTab = useUIStore((s) => s.inspectorTab)
  const setInspectorTab = useUIStore((s) => s.setInspectorTab)
  const selectedArtifact = useUIStore((s) => s.selectedArtifact)

  return (
    <aside className="h-full min-w-0 w-full flex flex-col bg-[var(--color-bg-surface)] border-l border-[var(--color-border-subtle)] overflow-hidden">
      {/* Header */}
      <header className="h-11 flex items-center justify-between px-3 border-b border-[var(--color-border-subtle)] flex-none min-w-0">
        <div className="flex items-center gap-2">
          <PanelRight className="w-3.5 h-3.5 text-[var(--color-text-tertiary)]" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
            Inspector
          </span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            aria-label="Close inspector"
            className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors text-[11px]"
          >
            Hide
          </button>
        )}
      </header>

      {/* Tabs */}
      <Tabs
        value={inspectorTab}
        onValueChange={(v) => setInspectorTab(v as InspectorTab)}
        className="flex-1 flex flex-col min-h-0"
      >
        <TabsList className="px-2 flex-none overflow-x-auto">
          {TABS.map((t) => (
            <TabsTrigger key={t.id} value={t.id} className="flex items-center gap-1.5">
              {t.icon}
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <div className="flex-1 overflow-y-auto">
          <TabsContent value="query" className="p-0">
            <QueryTab artifact={selectedArtifact} />
          </TabsContent>
          <TabsContent value="schema" className="p-4">
            <EmptyTab label="Schema" hint="Connect a database to browse tables and columns here." />
          </TabsContent>
          <TabsContent value="profile" className="p-4">
            <EmptyTab label="Profile" hint="Upload a CSV to see dtypes, null counts, and summary stats." />
          </TabsContent>
          <TabsContent value="metadata" className="p-0">
            <MetadataTab artifact={selectedArtifact} />
          </TabsContent>
          <TabsContent value="artifact" className="p-0">
            <ArtifactTab artifact={selectedArtifact} />
          </TabsContent>
        </div>
      </Tabs>
    </aside>
  )
}

/* -------------------- Query tab -------------------- */

function QueryTab({ artifact }: { artifact: ReturnType<typeof useUIStore.getState>["selectedArtifact"] }) {
  const sql = artifact?.type === "sql" ? String(artifact.payload?.sql ?? "") : ""
  const explanation = artifact?.type === "sql" ? String(artifact.payload?.explanation ?? "") : ""

  if (!sql) {
    return <div className="p-4"><EmptyTab label="Query" hint="Click a SQL card in the chat to inspect the full query here." /></div>
  }

  return (
    <div className="p-3 space-y-3">
      <SectionLabel>SQL</SectionLabel>
      <Highlight code={sql} language="sql" theme={themes.vsDark}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`${className} text-[12px] leading-[1.55] p-3 rounded-[5px] overflow-x-auto border border-[var(--color-border-subtle)]`}
            style={{ ...style, background: "var(--color-bg-inset)", fontFamily: "var(--font-jetbrains-mono), monospace" }}
          >
            {tokens.map((line, i) => {
              const lineProps = getLineProps({ line })
              return (
                <div key={i} {...lineProps}>
                  <span className="inline-block w-6 select-none text-[var(--color-text-quaternary)] pr-2 text-right">{i + 1}</span>
                  {line.map((token, key) => {
                    const tokenProps = getTokenProps({ token })
                    return <span key={key} {...tokenProps} />
                  })}
                </div>
              )
            })}
          </pre>
        )}
      </Highlight>

      {explanation && (
        <>
          <SectionLabel>Explanation</SectionLabel>
          <p className="text-[12px] leading-relaxed text-[var(--color-text-secondary)] whitespace-pre-wrap">
            {explanation}
          </p>
        </>
      )}

      <div className="pt-2 flex items-center gap-2">
        <button
          onClick={() => navigator.clipboard.writeText(sql)}
          className="text-[11px] px-2.5 h-7 rounded-[5px] border border-[var(--color-border-default)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-cyan-500)] transition-colors"
        >
          Copy SQL
        </button>
      </div>
    </div>
  )
}

/* -------------------- Artifact tab (charts) -------------------- */

function ArtifactTab({ artifact }: { artifact: ReturnType<typeof useUIStore.getState>["selectedArtifact"] }) {
  const b64 = artifact?.type === "chart" ? String(artifact.payload?.b64 ?? "") : ""
  const meta = (artifact?.type === "chart" ? (artifact.payload?.meta as Record<string, unknown> | undefined) : undefined) ?? undefined

  const dataUrl = useMemo(() => (b64 ? `data:image/png;base64,${b64}` : ""), [b64])

  if (!dataUrl) {
    return <div className="p-4"><EmptyTab label="Artifact" hint="Click a chart in the chat to inspect it full-size here." /></div>
  }

  const title = (meta?.title as string) || "Chart"
  const chartType = (meta?.type as string) || ""
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
    <div className="p-3 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[13px] font-semibold text-[var(--color-text-primary)] truncate">{title}</div>
          {chartType && (
            <div className="text-[10px] uppercase tracking-wider text-[var(--color-text-tertiary)] mt-0.5">
              {chartType}
            </div>
          )}
        </div>
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
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          {xLabel && (
            <div>
              <div className="text-[var(--color-text-tertiary)] uppercase tracking-wider text-[10px]">X-axis</div>
              <div className="text-[var(--color-text-secondary)] mt-0.5">{xLabel}</div>
            </div>
          )}
          {yLabel && (
            <div>
              <div className="text-[var(--color-text-tertiary)] uppercase tracking-wider text-[10px]">Y-axis</div>
              <div className="text-[var(--color-text-secondary)] mt-0.5">{yLabel}</div>
            </div>
          )}
        </div>
      )}

      <div className="pt-1 text-[10px] text-[var(--color-text-tertiary)] flex items-center gap-1">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--color-amber-500)]" />
        AI-generated
      </div>
    </div>
  )
}

/* -------------------- Metadata tab (citations) -------------------- */

interface Citation {
  source?: string
  page?: number | string
  text?: string
}

function MetadataTab({ artifact }: { artifact: ReturnType<typeof useUIStore.getState>["selectedArtifact"] }) {
  const citations = (artifact?.type === "citations" ? (artifact.payload?.citations as Citation[] | undefined) : undefined) ?? []

  if (!citations.length) {
    return <div className="p-4"><EmptyTab label="Metadata" hint="Click a citation in the chat to see the source excerpt here." /></div>
  }

  return (
    <div className="p-3 space-y-2">
      <SectionLabel>Citations ({citations.length})</SectionLabel>
      <div className="space-y-2">
        {citations.map((c, i) => (
          <div
            key={i}
            className="p-3 rounded-[5px] border border-[var(--color-border-subtle)] bg-[var(--color-bg-inset)]"
          >
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <div className="min-w-0 flex-1">
                <div className="text-[12px] font-medium text-[var(--color-text-primary)] truncate">
                  {c.source || "Source"}
                </div>
                {c.page !== undefined && c.page !== "" && (
                  <div className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5">
                    Page {c.page}
                  </div>
                )}
              </div>
              <ExternalLink className="w-3 h-3 text-[var(--color-text-quaternary)] flex-none mt-0.5" />
            </div>
            {c.text && (
              <p className="text-[11px] leading-relaxed text-[var(--color-text-secondary)] line-clamp-6">
                {c.text}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

/* -------------------- Shared primitives -------------------- */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
      {children}
    </div>
  )
}

function EmptyTab({ label, hint }: { label: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <p className="text-[13px] text-[var(--color-text-secondary)] mb-1">No {label.toLowerCase()} selected</p>
      <p className="text-[11px] text-[var(--color-text-tertiary)] max-w-[240px]">{hint}</p>
    </div>
  )
}
