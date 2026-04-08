"use client"

import React from "react"

type CardAccent = "none" | "cyan" | "amber" | "success" | "warning" | "danger"

interface CardProps {
  children: React.ReactNode
  className?: string
  hoverable?: boolean
  onClick?: () => void
  /** Optional 2px left rule — signals message/artifact origin */
  accent?: CardAccent
}

const accentClasses: Record<CardAccent, string> = {
  none: "",
  cyan: "border-l-2 border-l-[var(--color-cyan-500)]",
  amber: "border-l-2 border-l-[var(--color-amber-500)]",
  success: "border-l-2 border-l-[var(--color-success-500)]",
  warning: "border-l-2 border-l-[var(--color-warning-500)]",
  danger: "border-l-2 border-l-[var(--color-danger-500)]",
}

export default function Card({
  children,
  className = "",
  hoverable = false,
  onClick,
  accent = "none",
}: CardProps) {
  const isClickable = !!onClick

  return (
    <div
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      onClick={onClick}
      onKeyDown={
        isClickable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                onClick()
              }
            }
          : undefined
      }
      className={[
        "bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] rounded-[8px]",
        accentClasses[accent],
        hoverable || isClickable
          ? "transition-colors duration-[var(--duration-fast)] hover:border-[var(--color-border-default)]"
          : "",
        isClickable ? "cursor-pointer" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </div>
  )
}
