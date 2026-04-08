"use client"

import React, { useState, useEffect } from "react"
import { Command } from "cmdk"
import {
  Search, Database, Sparkles, ShoppingCart,
  Trash2, Download, BookOpen, Wand2,
} from "lucide-react"
import { useUIStore } from "@/store/uiStore"

interface CommandItem {
  id: string
  label: string
  description: string
  icon: React.ReactNode
  action: () => void
  keywords: string[]
  group: "Navigate" | "Personas" | "Actions"
}

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
  commands: CommandItem[]
}

export function useCommandPalette() {
  // Kept for backwards-compat with handlers in chat/page.tsx; the store is the source of truth.
  const isOpen = useUIStore((s) => s.commandPaletteOpen)
  const setIsOpen = useUIStore((s) => s.setCommandPaletteOpen)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        setIsOpen(!isOpen)
      }
      if (e.key === "Escape") {
        setIsOpen(false)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [isOpen, setIsOpen])

  return { isOpen, setIsOpen, onClose: () => setIsOpen(false) }
}

export function buildCommands(actions: {
  onConnectDatabase?: () => void
  onSearchEdgar?: () => void
  onAddConnector?: () => void
  onClearChat?: () => void
  onExportChat?: () => void
  onSwitchPersona?: (name: string) => void
  onSetAutoMode?: (v: boolean) => void
  onToggleInspector?: () => void
}): CommandItem[] {
  const commands: CommandItem[] = []

  if (actions.onConnectDatabase) {
    commands.push({
      id: "connect-db",
      label: "Connect Database",
      description: "PostgreSQL, MySQL, SQLite, Azure SQL",
      icon: <Database className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />,
      action: actions.onConnectDatabase,
      keywords: ["database", "connect", "postgresql", "mysql", "sql", "live"],
      group: "Navigate",
    })
  }

  if (actions.onSearchEdgar) {
    commands.push({
      id: "search-edgar",
      label: "Search SEC Filings",
      description: "Search and ingest SEC EDGAR filings",
      icon: <BookOpen className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />,
      action: actions.onSearchEdgar,
      keywords: ["edgar", "sec", "filing", "10-k", "10-q", "annual"],
      group: "Navigate",
    })
  }

  if (actions.onAddConnector) {
    commands.push({
      id: "add-connector",
      label: "Add Marketplace Connector",
      description: "Connect Amazon or Shopify",
      icon: <ShoppingCart className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />,
      action: actions.onAddConnector,
      keywords: ["marketplace", "amazon", "shopify", "connector"],
      group: "Navigate",
    })
  }

  if (actions.onToggleInspector) {
    commands.push({
      id: "toggle-inspector",
      label: "Toggle Inspector",
      description: "Show or hide the right panel",
      icon: <Sparkles className="w-3.5 h-3.5 text-[var(--color-cyan-500)]" />,
      action: actions.onToggleInspector,
      keywords: ["inspector", "panel", "right", "toggle"],
      group: "Navigate",
    })
  }

  if (actions.onSetAutoMode) {
    const setAuto = actions.onSetAutoMode
    commands.push({
      id: "persona-auto",
      label: "Auto-routing",
      description: "Let AI pick the best expert",
      icon: <Wand2 className="w-3.5 h-3.5 text-[var(--color-amber-500)]" />,
      action: () => setAuto(true),
      keywords: ["auto", "routing", "persona"],
      group: "Personas",
    })
  }

  const personas = [
    { name: "Finance Expert", desc: "Financial & investment analysis" },
    { name: "Data Analyst", desc: "Quantitative analysis with SQL transparency" },
    { name: "Generalist", desc: "General-purpose assistant" },
    { name: "Consultant", desc: "Strategic business advisory" },
  ]

  if (actions.onSwitchPersona && actions.onSetAutoMode) {
    const switchFn = actions.onSwitchPersona
    const setAuto = actions.onSetAutoMode
    for (const p of personas) {
      commands.push({
        id: `persona-${p.name}`,
        label: `Switch to ${p.name}`,
        description: p.desc,
        icon: <Sparkles className="w-3.5 h-3.5 text-[var(--color-amber-500)]" />,
        action: () => { setAuto(false); switchFn(p.name) },
        keywords: ["persona", "switch", p.name.toLowerCase()],
        group: "Personas",
      })
    }
  }

  if (actions.onExportChat) {
    commands.push({
      id: "export-chat",
      label: "Export Chat",
      description: "Export as text, markdown, or JSON",
      icon: <Download className="w-3.5 h-3.5 text-[var(--color-text-tertiary)]" />,
      action: actions.onExportChat,
      keywords: ["export", "download", "save"],
      group: "Actions",
    })
  }

  if (actions.onClearChat) {
    commands.push({
      id: "clear-chat",
      label: "Clear Chat",
      description: "Remove all messages",
      icon: <Trash2 className="w-3.5 h-3.5 text-[var(--color-danger-500)]" />,
      action: actions.onClearChat,
      keywords: ["clear", "chat", "reset"],
      group: "Actions",
    })
  }

  return commands
}

export default function CommandPalette({ isOpen, onClose, commands }: CommandPaletteProps) {
  const [value, setValue] = useState("")

  useEffect(() => {
    if (isOpen) setValue("")
  }, [isOpen])

  if (!isOpen) return null

  const groups: CommandItem["group"][] = ["Navigate", "Personas", "Actions"]

  const executeCommand = (cmd: CommandItem) => {
    onClose()
    cmd.action()
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[18vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-[var(--bg-scrim,rgba(0,0,0,0.6))] backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />

      {/* Palette */}
      <div
        className="relative w-full max-w-[560px] mx-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-[12px] shadow-[var(--elev-4)] overflow-hidden"
        role="dialog"
        aria-label="Command palette"
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
              placeholder="Type a command or search…"
              className="flex-1 bg-transparent text-[14px] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-quaternary)] outline-none"
            />
            <kbd className="text-[10px] h-5 px-1.5 inline-flex items-center rounded-[3px] border border-[var(--color-border-default)] bg-[var(--color-bg-overlay)] text-[var(--color-text-tertiary)] font-mono">
              ESC
            </kbd>
          </div>

          {/* Results */}
          <Command.List className="max-h-[340px] overflow-y-auto py-2">
            <Command.Empty className="px-4 py-8 text-center text-[12px] text-[var(--color-text-tertiary)]">
              No commands found
            </Command.Empty>

            {groups.map((group) => {
              const items = commands.filter((c) => c.group === group)
              if (!items.length) return null
              return (
                <Command.Group
                  key={group}
                  heading={group}
                  className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-[var(--color-text-quaternary)]"
                >
                  {items.map((cmd) => (
                    <Command.Item
                      key={cmd.id}
                      value={`${cmd.label} ${cmd.description} ${cmd.keywords.join(" ")}`}
                      onSelect={() => executeCommand(cmd)}
                      className="group relative flex items-center gap-3 px-3 h-9 mx-2 rounded-[5px] cursor-pointer text-[13px] text-[var(--color-text-secondary)] data-[selected=true]:bg-[var(--color-bg-overlay)] data-[selected=true]:text-[var(--color-text-primary)] data-[selected=true]:before:content-[''] data-[selected=true]:before:absolute data-[selected=true]:before:left-0 data-[selected=true]:before:top-1.5 data-[selected=true]:before:bottom-1.5 data-[selected=true]:before:w-[2px] data-[selected=true]:before:bg-[var(--color-cyan-500)] data-[selected=true]:before:rounded-full"
                    >
                      <span className="flex-none">{cmd.icon}</span>
                      <span className="flex-1 min-w-0 truncate">{cmd.label}</span>
                      <span className="flex-none text-[11px] text-[var(--color-text-tertiary)] truncate max-w-[220px]">
                        {cmd.description}
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
                <Kbd>↵</Kbd> select
              </span>
            </div>
            <span className="flex items-center gap-1">
              <Kbd>⌘K</Kbd> toggle
            </span>
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
