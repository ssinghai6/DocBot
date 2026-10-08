"use client"

import { PanelRight, GitBranch, BookOpen, Code2, Table2, Gauge } from "lucide-react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui"
import { useUIStore, normalizeInspectorTab, type InspectorTab } from "@/store/uiStore"
import LineageTab from "./inspector/LineageTab"
import SourcesTab from "./inspector/SourcesTab"
import QueryTab from "./inspector/QueryTab"
import DataTab from "./inspector/DataTab"
import RunTab from "./inspector/RunTab"

interface InspectorPanelProps {
  onClose?: () => void
}

const TABS: Array<{ id: InspectorTab; label: string; icon: React.ReactNode }> = [
  { id: "lineage", label: "Lineage", icon: <GitBranch className="w-3 h-3" /> },
  { id: "sources", label: "Sources", icon: <BookOpen className="w-3 h-3" /> },
  { id: "query",   label: "Query",   icon: <Code2 className="w-3 h-3" /> },
  { id: "data",    label: "Data",    icon: <Table2 className="w-3 h-3" /> },
  { id: "run",     label: "Run",     icon: <Gauge className="w-3 h-3" /> },
]

export default function InspectorPanel({ onClose }: InspectorPanelProps) {
  const inspectorTab = normalizeInspectorTab(useUIStore((s) => s.inspectorTab))
  const setInspectorTab = useUIStore((s) => s.setInspectorTab)
  const lineage = useUIStore((s) => s.selectedLineage)

  const badge = (id: InspectorTab): number | null => {
    if (!lineage) return null
    if (id === "sources") return lineage.sources.length || null
    if (id === "lineage") return lineage.discrepancies.length || null
    return null
  }

  return (
    <aside className="h-full min-w-[280px] w-full flex flex-col bg-[var(--color-bg-surface)] border-l border-[var(--color-border-subtle)] overflow-hidden">
      <header className="h-11 flex items-center justify-between gap-2 px-3 border-b border-[var(--color-border-subtle)] flex-none min-w-0">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <PanelRight className="w-3.5 h-3.5 text-[var(--color-text-tertiary)] flex-none" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)] truncate whitespace-nowrap">
            Inspector
          </span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            aria-label="Close inspector"
            className="flex-none text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors text-[11px] whitespace-nowrap"
          >
            Hide
          </button>
        )}
      </header>

      <Tabs
        value={inspectorTab}
        onValueChange={(v) => setInspectorTab(normalizeInspectorTab(v))}
        className="flex-1 flex flex-col min-h-0"
      >
        <TabsList className="px-2 flex-none overflow-x-auto whitespace-nowrap">
          {TABS.map((t) => {
            const n = badge(t.id)
            return (
              <TabsTrigger key={t.id} value={t.id} className="flex items-center gap-1.5 shrink-0 whitespace-nowrap">
                <span className="flex-none">{t.icon}</span>
                <span>{t.label}</span>
                {n != null && (
                  <span className={`text-[9px] tabular-nums px-1 rounded-[3px] ${t.id === "lineage" ? "bg-[var(--color-amber-500)]/20 text-[var(--color-amber-500)]" : "bg-[var(--color-bg-inset)] text-[var(--color-text-tertiary)]"}`}>
                    {n}
                  </span>
                )}
              </TabsTrigger>
            )
          })}
        </TabsList>

        <div className="flex-1 overflow-y-auto">
          <TabsContent value="lineage" className="p-0"><LineageTab lineage={lineage} /></TabsContent>
          <TabsContent value="sources" className="p-0"><SourcesTab lineage={lineage} /></TabsContent>
          <TabsContent value="query" className="p-0"><QueryTab lineage={lineage} /></TabsContent>
          <TabsContent value="data" className="p-0"><DataTab lineage={lineage} /></TabsContent>
          <TabsContent value="run" className="p-0"><RunTab lineage={lineage} /></TabsContent>
        </div>
      </Tabs>
    </aside>
  )
}
