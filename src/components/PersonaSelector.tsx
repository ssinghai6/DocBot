"use client"

import React from "react"
import {
  Sparkles, Brain, TrendingUp,
  Briefcase, BarChart2, Wand2, UserCog,
  CheckCircle2, Info, Zap, Layers,
} from "lucide-react"

// ── Persona definitions (mirrors EXPERT_PERSONAS in page.tsx exactly) ────────

type PersonaConfig = {
  icon: React.ReactNode
  description: string
  color: string
  gradient: string
  accent: string
}

// All personas — full registry (backend still supports all 8)
const ALL_EXPERT_PERSONAS: Record<string, PersonaConfig> = {
  Generalist: {
    icon: <Sparkles className="w-5 h-5" />,
    description: "Balanced, general-purpose assistant for any document",
    color: "text-[var(--color-cyan-500)]",
    gradient: "from-[var(--color-cyan-500)] to-[var(--color-cyan-600)]",
    accent: "#667eea",
  },
  "Finance Expert": {
    icon: <TrendingUp className="w-5 h-5" />,
    description: "Financial & investment analysis - parse reports, statements & SEC filings",
    color: "text-[var(--color-amber-500)]",
    gradient: "from-[var(--color-amber-500)] to-[var(--color-amber-600)]",
    accent: "#f97316",
  },
  "Data Analyst": {
    icon: <BarChart2 className="w-5 h-5" />,
    description: "Quantitative analysis with full SQL transparency and data quality flags",
    color: "text-[var(--color-amber-500)]",
    gradient: "from-[var(--color-amber-500)] to-[var(--color-amber-600)]",
    accent: "#f97316",
  },
  Consultant: {
    icon: <Briefcase className="w-5 h-5" />,
    description: "Strategic business advisory - market sizing, competitive analysis & planning",
    color: "text-[var(--color-cyan-500)]",
    gradient: "from-[var(--color-cyan-500)] to-[var(--color-cyan-600)]",
    accent: "#667eea",
  },
}

// Display names (UI labels — keys match backend persona names)
const DISPLAY_NAMES: Record<string, string> = {
  Consultant: "Strategy Analyst",
}

// Visible personas (finance-vertical focus)
const EXPERT_PERSONAS = ALL_EXPERT_PERSONAS

interface PersonaSelectorProps {
  selectedPersona: string
  suggestedPersona: string | null
  isAutoMode: boolean
  deepResearch: boolean
  onSelectPersona: (name: string) => void
  onSetAutoMode: (value: boolean) => void
  onDeepResearchChange: (value: boolean) => void
}

export default function PersonaSelector({
  selectedPersona,
  suggestedPersona,
  isAutoMode,
  deepResearch,
  onSelectPersona,
  onSetAutoMode,
  onDeepResearchChange,
}: PersonaSelectorProps) {
  return (
    <div className="flex-1">
      {suggestedPersona && suggestedPersona !== "Generalist" && (
        <div className="bg-[var(--color-cyan-500)]/10 border border-[var(--color-cyan-500)]/20 p-3 rounded-xl mb-3 text-sm flex items-start gap-2">
          <Zap className="w-4 h-4 text-[var(--color-cyan-500)] mt-0.5 shrink-0" />
          <div>
            <span className="text-gray-300">Recommended: </span>
            <strong className="text-[var(--color-text-primary)]">{suggestedPersona}</strong>
          </div>
        </div>
      )}

      {/* AUTO / Manual toggle */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => onSetAutoMode(true)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isAutoMode
              ? "bg-[var(--color-cyan-500)]/10 text-[var(--color-cyan-600)] border border-[var(--color-cyan-500)]/40"
              : "bg-[var(--color-bg-surface)] text-[var(--color-text-secondary)] border border-[var(--color-border-default)] hover:bg-[var(--color-bg-elevated)]"
          }`}
        >
          <Wand2 className="w-3 h-3" />
          AUTO
        </button>
        <button
          onClick={() => onSetAutoMode(false)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            !isAutoMode
              ? "bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] border border-[var(--color-border-strong)]"
              : "bg-[var(--color-bg-surface)] text-[var(--color-text-secondary)] border border-[var(--color-border-default)] hover:bg-[var(--color-bg-elevated)]"
          }`}
        >
          <UserCog className="w-3 h-3" />
          Manual
        </button>
      </div>

      {isAutoMode ? (
        <p className="text-[11px] text-[var(--color-text-tertiary)] mb-3">Routing automatically based on your question</p>
      ) : (
        <div>
          {/* Persona Cards Grid */}
          <div className="grid grid-cols-2 gap-2 mb-2">
            {Object.entries(EXPERT_PERSONAS).map(([name, data]) => {
              const isSelected = selectedPersona === name
              return (
                <button
                  key={name}
                  onClick={() => onSelectPersona(name)}
                  className={`group relative p-3 rounded-xl text-left transition-all duration-200 ${
                    isSelected
                      ? "bg-gradient-to-br " + data.gradient + " shadow-lg scale-[1.02]"
                      : "bg-[var(--color-bg-elevated)]/50 border border-[var(--color-border-subtle)] hover:border-[var(--color-border-default)] hover:bg-[var(--color-bg-elevated)]/80"
                  }`}
                >
                  <div className={`${isSelected ? "text-[var(--color-text-primary)]" : data.color} mb-1`}>
                    {data.icon}
                  </div>
                  <div className={`text-xs font-medium ${isSelected ? "text-[var(--color-text-primary)]" : "text-gray-300"}`}>
                    {DISPLAY_NAMES[name] || name}
                  </div>
                  {isSelected && (
                    <div className="absolute top-2 right-2">
                      <CheckCircle2 className="w-3 h-3 text-[var(--color-text-primary)]" />
                    </div>
                  )}
                </button>
              )
            })}
          </div>
          <button
            onClick={() => onSetAutoMode(true)}
            className="text-[10px] text-[var(--color-cyan-500)] hover:underline mt-1"
          >
            &#8617; Reset to Auto
          </button>
        </div>
      )}

      {!isAutoMode && (
        <div className="mt-3 p-3 bg-[var(--color-bg-elevated)]/30 rounded-xl border border-[var(--color-border-subtle)]">
          <p className="text-xs text-[var(--color-text-secondary)] flex items-start gap-2">
            <Info className="w-3 h-3 mt-0.5 shrink-0 text-[var(--color-cyan-500)]" />
            {EXPERT_PERSONAS[selectedPersona]?.description}
          </p>
        </div>
      )}

      {/* Deep Research Toggle */}
      {selectedPersona !== "Generalist" && (
        <div className="mt-4 bg-[var(--color-bg-elevated)]/50 rounded-xl p-3 border border-[var(--color-border-subtle)]">
          <h3 className="text-sm font-semibold mb-3 text-[var(--color-text-primary)] flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#764ba2]" />
            Analysis Mode
          </h3>
          <label className="flex items-center justify-between cursor-pointer group">
            <div className="flex items-center gap-3">
              <div className={`w-9 h-5 flex items-center bg-gray-700/50 rounded-full p-0.5 transition-colors ${deepResearch ? "bg-[#764ba2]" : ""}`}>
                <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepResearch ? "translate-x-4" : ""}`} />
              </div>
              <div>
                <span className="text-sm font-medium text-gray-300 group-hover:text-[var(--color-text-primary)] transition-colors">Deep Research</span>
                <p className="text-[10px] text-[var(--color-text-tertiary)]">Enhanced reasoning</p>
              </div>
            </div>
            <input
              type="checkbox"
              className="hidden"
              checked={deepResearch}
              onChange={() => onDeepResearchChange(!deepResearch)}
            />
          </label>
          {deepResearch && (
            <p className="text-xs text-[var(--color-text-tertiary)] mt-2 ml-12 flex items-center gap-1">
              <Brain className="w-3 h-3 text-[#764ba2]" /> Advanced analysis enabled
            </p>
          )}
        </div>
      )}
    </div>
  )
}
