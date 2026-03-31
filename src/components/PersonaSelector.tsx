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
    color: "text-[#667eea]",
    gradient: "from-[#667eea] to-[#764ba2]",
    accent: "#667eea",
  },
  "Finance Expert": {
    icon: <TrendingUp className="w-5 h-5" />,
    description: "Financial & investment analysis - parse reports, statements & SEC filings",
    color: "text-[#f97316]",
    gradient: "from-[#f97316] to-[#ea580c]",
    accent: "#f97316",
  },
  "Data Analyst": {
    icon: <BarChart2 className="w-5 h-5" />,
    description: "Quantitative analysis with full SQL transparency and data quality flags",
    color: "text-[#f97316]",
    gradient: "from-[#f97316] to-[#ea580c]",
    accent: "#f97316",
  },
  Consultant: {
    icon: <Briefcase className="w-5 h-5" />,
    description: "Strategic business advisory - market sizing, competitive analysis & planning",
    color: "text-[#667eea]",
    gradient: "from-[#667eea] to-[#764ba2]",
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
        <div className="bg-[#667eea]/10 border border-[#667eea]/20 p-3 rounded-xl mb-3 text-sm flex items-start gap-2">
          <Zap className="w-4 h-4 text-[#667eea] mt-0.5 shrink-0" />
          <div>
            <span className="text-gray-300">Recommended: </span>
            <strong className="text-white">{suggestedPersona}</strong>
          </div>
        </div>
      )}

      {/* AUTO / Manual toggle */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => onSetAutoMode(true)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isAutoMode
              ? "bg-[#667eea]/20 text-[#667eea] border border-[#667eea]/40"
              : "bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10"
          }`}
        >
          <Wand2 className="w-3 h-3" />
          AUTO
        </button>
        <button
          onClick={() => onSetAutoMode(false)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            !isAutoMode
              ? "bg-white/10 text-gray-200 border border-white/20"
              : "bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10"
          }`}
        >
          <UserCog className="w-3 h-3" />
          Manual
        </button>
      </div>

      {isAutoMode ? (
        <p className="text-[11px] text-gray-500 mb-3">Routing automatically based on your question</p>
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
                      : "bg-[#1a1a24]/50 border border-[#ffffff06] hover:border-[#ffffff15] hover:bg-[#1a1a24]/80"
                  }`}
                >
                  <div className={`${isSelected ? "text-white" : data.color} mb-1`}>
                    {data.icon}
                  </div>
                  <div className={`text-xs font-medium ${isSelected ? "text-white" : "text-gray-300"}`}>
                    {DISPLAY_NAMES[name] || name}
                  </div>
                  {isSelected && (
                    <div className="absolute top-2 right-2">
                      <CheckCircle2 className="w-3 h-3 text-white" />
                    </div>
                  )}
                </button>
              )
            })}
          </div>
          <button
            onClick={() => onSetAutoMode(true)}
            className="text-[10px] text-[#667eea] hover:underline mt-1"
          >
            &#8617; Reset to Auto
          </button>
        </div>
      )}

      {!isAutoMode && (
        <div className="mt-3 p-3 bg-[#1a1a24]/30 rounded-xl border border-[#ffffff06]">
          <p className="text-xs text-gray-400 flex items-start gap-2">
            <Info className="w-3 h-3 mt-0.5 shrink-0 text-[#667eea]" />
            {EXPERT_PERSONAS[selectedPersona]?.description}
          </p>
        </div>
      )}

      {/* Deep Research Toggle */}
      {selectedPersona !== "Generalist" && (
        <div className="mt-4 bg-[#1a1a24]/50 rounded-xl p-3 border border-[#ffffff06]">
          <h3 className="text-sm font-semibold mb-3 text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#764ba2]" />
            Analysis Mode
          </h3>
          <label className="flex items-center justify-between cursor-pointer group">
            <div className="flex items-center gap-3">
              <div className={`w-9 h-5 flex items-center bg-gray-700/50 rounded-full p-0.5 transition-colors ${deepResearch ? "bg-[#764ba2]" : ""}`}>
                <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepResearch ? "translate-x-4" : ""}`} />
              </div>
              <div>
                <span className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">Deep Research</span>
                <p className="text-[10px] text-gray-500">Enhanced reasoning</p>
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
            <p className="text-xs text-gray-500 mt-2 ml-12 flex items-center gap-1">
              <Brain className="w-3 h-3 text-[#764ba2]" /> Advanced analysis enabled
            </p>
          )}
        </div>
      )}
    </div>
  )
}
