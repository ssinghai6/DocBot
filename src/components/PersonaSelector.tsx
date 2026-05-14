"use client"

import React from "react"
import {
  Wand2, UserCog,
  CheckCircle2, Info,
} from "lucide-react"

import { EXPERT_PERSONAS } from "@/components/personas"

// Visible personas in the Tools tab (Generalist, Finance Expert, Data Analyst,
// Strategy Analyst). Lawyer + Doctor stay in EXPERT_PERSONAS for auto-routing
// but are not surfaced here, since the visible product is finance-vertical.
const VISIBLE_PERSONA_KEYS = ["Generalist", "Finance Expert", "Data Analyst", "Strategy Analyst"] as const

interface PersonaSelectorProps {
  selectedPersona: string
  isAutoMode: boolean
  onSelectPersona: (name: string) => void
  onSetAutoMode: (value: boolean) => void
}

export default function PersonaSelector({
  selectedPersona,
  isAutoMode,
  onSelectPersona,
  onSetAutoMode,
}: PersonaSelectorProps) {
  return (
    <div className="flex-1">

      {/* AUTO / Manual toggle */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => onSetAutoMode(true)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isAutoMode
              ? "bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-500)] border border-[var(--color-cyan-500)]/40"
              : "bg-white/5 text-[var(--color-text-secondary)] border border-white/10 hover:bg-white/10"
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
              : "bg-white/5 text-[var(--color-text-secondary)] border border-white/10 hover:bg-white/10"
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
            {VISIBLE_PERSONA_KEYS.map((name) => {
              const data = EXPERT_PERSONAS[name]
              if (!data) return null
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
                    {name}
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

    </div>
  )
}
