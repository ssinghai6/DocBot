"use client"

import React, { useState, useEffect, useRef, useCallback } from "react"
import {
  Search, Database, FileText, Sparkles, ShoppingCart,
  Trash2, Download, BookOpen, Wand2, X,
} from "lucide-react"

interface Command {
  id: string
  label: string
  description: string
  icon: React.ReactNode
  action: () => void
  keywords: string[]
}

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
  commands: Command[]
}

export function useCommandPalette() {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        setIsOpen(prev => !prev)
      }
      if (e.key === "Escape") {
        setIsOpen(false)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

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
}): Command[] {
  const commands: Command[] = []

  if (actions.onConnectDatabase) {
    commands.push({
      id: "connect-db",
      label: "Connect Database",
      description: "Connect to PostgreSQL, MySQL, or Azure SQL",
      icon: <Database className="w-4 h-4 text-[#f97316]" />,
      action: actions.onConnectDatabase,
      keywords: ["database", "connect", "postgresql", "mysql", "sql", "live"],
    })
  }

  if (actions.onSearchEdgar) {
    commands.push({
      id: "search-edgar",
      label: "Search SEC Filings",
      description: "Search and ingest SEC EDGAR filings",
      icon: <BookOpen className="w-4 h-4 text-[#10b981]" />,
      action: actions.onSearchEdgar,
      keywords: ["edgar", "sec", "filing", "10-k", "10-q", "annual"],
    })
  }

  if (actions.onAddConnector) {
    commands.push({
      id: "add-connector",
      label: "Add Marketplace Connector",
      description: "Connect Amazon or Shopify",
      icon: <ShoppingCart className="w-4 h-4 text-[#667eea]" />,
      action: actions.onAddConnector,
      keywords: ["marketplace", "amazon", "shopify", "connector"],
    })
  }

  if (actions.onSetAutoMode) {
    const setAuto = actions.onSetAutoMode
    commands.push({
      id: "persona-auto",
      label: "Auto-routing",
      description: "Let AI pick the best expert",
      icon: <Wand2 className="w-4 h-4 text-[#667eea]" />,
      action: () => setAuto(true),
      keywords: ["auto", "routing", "persona"],
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
        icon: <Sparkles className="w-4 h-4 text-[#667eea]" />,
        action: () => { setAuto(false); switchFn(p.name); },
        keywords: ["persona", "switch", p.name.toLowerCase()],
      })
    }
  }

  if (actions.onClearChat) {
    commands.push({
      id: "clear-chat",
      label: "Clear Chat",
      description: "Clear all messages",
      icon: <Trash2 className="w-4 h-4 text-red-400" />,
      action: actions.onClearChat,
      keywords: ["clear", "chat", "reset"],
    })
  }

  if (actions.onExportChat) {
    commands.push({
      id: "export-chat",
      label: "Export Chat",
      description: "Export as text, markdown, or JSON",
      icon: <Download className="w-4 h-4 text-gray-400" />,
      action: actions.onExportChat,
      keywords: ["export", "download", "save"],
    })
  }

  return commands
}

export default function CommandPalette({ isOpen, onClose, commands }: CommandPaletteProps) {
  const [query, setQuery] = useState("")
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const filtered = query.trim()
    ? commands.filter(cmd => {
        const q = query.toLowerCase()
        return (
          cmd.label.toLowerCase().includes(q) ||
          cmd.description.toLowerCase().includes(q) ||
          cmd.keywords.some(k => k.includes(q))
        )
      })
    : commands

  useEffect(() => {
    if (isOpen) {
      setQuery("")
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  const executeCommand = useCallback((cmd: Command) => {
    onClose()
    cmd.action()
  }, [onClose])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault()
      setSelectedIndex(i => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setSelectedIndex(i => Math.max(i - 1, 0))
    } else if (e.key === "Enter" && filtered[selectedIndex]) {
      e.preventDefault()
      executeCommand(filtered[selectedIndex])
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Palette */}
      <div className="relative w-full max-w-lg bg-[#12121a]/95 border border-[#ffffff10] rounded-2xl shadow-2xl shadow-black/40 overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[#ffffff08]">
          <Search className="w-4 h-4 text-gray-500 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command..."
            className="flex-1 bg-transparent text-white text-sm outline-none placeholder-gray-500"
          />
          <kbd className="px-1.5 py-0.5 text-[10px] text-gray-500 bg-[#ffffff08] rounded border border-[#ffffff10]">ESC</kbd>
        </div>

        {/* Results */}
        <div className="max-h-72 overflow-y-auto py-1">
          {filtered.length === 0 ? (
            <div className="px-4 py-6 text-center text-sm text-gray-500">
              No commands found
            </div>
          ) : (
            filtered.map((cmd, idx) => (
              <button
                key={cmd.id}
                onClick={() => executeCommand(cmd)}
                onMouseEnter={() => setSelectedIndex(idx)}
                className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                  idx === selectedIndex ? "bg-[#667eea]/10" : "hover:bg-[#ffffff06]"
                }`}
              >
                {cmd.icon}
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${idx === selectedIndex ? "text-white" : "text-gray-300"}`}>
                    {cmd.label}
                  </p>
                  <p className="text-xs text-gray-500 truncate">{cmd.description}</p>
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
