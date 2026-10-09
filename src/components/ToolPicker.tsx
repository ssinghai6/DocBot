"use client"

import React, { useEffect, useState } from "react"
import { Command } from "cmdk"
import {
  Search, Database, Sparkles, ShoppingCart,
  Wand2, Layers, FileText, Wrench,
} from "lucide-react"
import { useUIStore } from "@/store/uiStore"
import { ToolsResponseSchema } from "@/components/types"
import type { ToolInfo, PickedTool } from "@/components/types"

interface ToolPickerProps {
  isOpen: boolean
  onClose: () => void
  onPick: (tool: PickedTool) => void
}

const CATEGORY_LABELS: Record<string, string> = {
  pipeline: "Pipelines",
  tool: "Autopilot Tools",
  persona: "Personas",
  connector: "Connectors",
}

const CATEGORY_ORDER = ["pipeline", "tool", "persona", "connector"]

function iconFor(tool: ToolInfo): React.ReactNode {
  switch (tool.icon) {
    case "wand": return <Wand2 className="w-3.5 h-3.5 text-[var(--color-amber-500)]" />
    case "database": return <Database className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />
    case "layers": return <Layers className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />
    case "file-text": return <FileText className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />
    case "search": return <Search className="w-3.5 h-3.5 text-[var(--color-amber-500)]" />
    case "chart": return <Wrench className="w-3.5 h-3.5 text-[var(--color-success-500)]" />
    case "shopping-cart": return <ShoppingCart className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />
    default: return <Sparkles className="w-3.5 h-3.5 text-[var(--color-amber-500)]" />
  }
}

/** Fetch + Zod-validate the tool registry (DOCBOT-1514). */
export function useToolRegistry(enabled: boolean) {
  const [tools, setTools] = useState<ToolInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    setLoading(true)
    setError(null)
    fetch("/api/tools")
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const parsed = ToolsResponseSchema.safeParse(data)
        if (!parsed.success) throw new Error("Invalid /api/tools response")
        if (!cancelled) setTools(parsed.data.tools)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load tools")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [enabled])

  return { tools, loading, error }
}

export function useToolPicker() {
  const isOpen = useUIStore((s) => s.toolPickerOpen)
  const setIsOpen = useUIStore((s) => s.setToolPickerOpen)
  const pickedTool = useUIStore((s) => s.pickedTool)
  const setPickedTool = useUIStore((s) => s.setPickedTool)

  return {
    isOpen,
    setIsOpen,
    onClose: () => setIsOpen(false),
    pickedTool,
    setPickedTool,
    clearPickedTool: () => setPickedTool(null),
  }
}

export default function ToolPicker({ isOpen, onClose, onPick }: ToolPickerProps) {
  const [value, setValue] = useState("")
  const { tools, loading, error } = useToolRegistry(isOpen)

  useEffect(() => {
    if (isOpen) setValue("")
  }, [isOpen])

  if (!isOpen) return null

  const selectTool = (tool: ToolInfo) => {
    onPick({ key: tool.key, name: tool.name, category: tool.category })
    onClose()
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[18vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-[var(--bg-scrim,rgba(0,0,0,0.6))] backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />

      {/* Picker */}
      <div
        className="relative w-full max-w-[560px] mx-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-[12px] shadow-[var(--elev-4)] overflow-hidden"
        role="dialog"
        aria-label="Tool picker"
      >
        <Command
          shouldFilter
          value={value}
          onValueChange={setValue}
          loop
          className="flex flex-col"
        >
          {/* Search */}
          <div className="flex items-center gap-3 px-4 h-[52px] border-b border-[var(--color-border-subtle)]">
            <Search className="w-4 h-4 text-[var(--color-text-tertiary)] shrink-0" />
            <Command.Input
              autoFocus
              placeholder="Pick a tool for this message…"
              className="flex-1 bg-transparent text-[14px] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-quaternary)] outline-none"
            />
            <kbd className="text-[10px] h-5 px-1.5 inline-flex items-center rounded-[3px] border border-[var(--color-border-default)] bg-[var(--color-bg-overlay)] text-[var(--color-text-tertiary)] font-mono">
              ESC
            </kbd>
          </div>

          {/* Results */}
          <Command.List className="max-h-[340px] overflow-y-auto py-2">
            {loading && (
              <div className="px-4 py-8 text-center text-[12px] text-[var(--color-text-tertiary)]">
                Loading tools…
              </div>
            )}
            {error && !loading && (
              <div className="px-4 py-8 text-center text-[12px] text-[var(--color-danger-500)]">
                {error}
              </div>
            )}
            {!loading && !error && (
              <Command.Empty className="px-4 py-8 text-center text-[12px] text-[var(--color-text-tertiary)]">
                No tools found
              </Command.Empty>
            )}

            {!loading && !error && CATEGORY_ORDER.map((category) => {
              const items = tools.filter((t) => t.category === category)
              if (!items.length) return null
              return (
                <Command.Group
                  key={category}
                  heading={CATEGORY_LABELS[category] ?? category}
                  className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-[var(--color-text-quaternary)]"
                >
                  {items.map((tool) => (
                    <Command.Item
                      key={tool.key}
                      value={`${tool.name} ${tool.description}`}
                      onSelect={() => selectTool(tool)}
                      className="group relative flex items-center gap-3 px-3 h-9 mx-2 rounded-[5px] cursor-pointer text-[13px] text-[var(--color-text-secondary)] data-[selected=true]:bg-[var(--color-bg-overlay)] data-[selected=true]:text-[var(--color-text-primary)] data-[selected=true]:before:content-[''] data-[selected=true]:before:absolute data-[selected=true]:before:left-0 data-[selected=true]:before:top-1.5 data-[selected=true]:before:bottom-1.5 data-[selected=true]:before:w-[2px] data-[selected=true]:before:bg-[var(--color-cyan-500)] data-[selected=true]:before:rounded-full"
                    >
                      <span className="flex-none">{iconFor(tool)}</span>
                      <span className="flex-1 min-w-0 truncate">{tool.name}</span>
                      <span className="flex-none text-[11px] text-[var(--color-text-tertiary)] truncate max-w-[220px]">
                        {tool.description}
                      </span>
                    </Command.Item>
                  ))}
                </Command.Group>
              )
            })}
          </Command.List>

          {/* Footer */}
          <div className="flex items-center justify-between px-4 h-8 border-t border-[var(--color-border-subtle)] text-[10px] text-[var(--color-text-tertiary)]">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <Kbd>↑</Kbd><Kbd>↓</Kbd> navigate
              </span>
              <span className="flex items-center gap-1">
                <Kbd>↵</Kbd> use for next message
              </span>
            </div>
          </div>
        </Command>
      </div>
    </div>
  )
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[16px] h-4 px-1 rounded-[3px] border border-[var(--color-border-default)] bg-[var(--color-bg-overlay)] text-[9px] font-mono text-[var(--color-text-tertiary)]">
      {children}
    </kbd>
  )
}
