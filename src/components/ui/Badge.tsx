"use client"

import React from "react"

type BadgeVariant = "default" | "success" | "warning" | "error" | "info" | "cyan" | "amber"

interface BadgeProps {
  variant?: BadgeVariant
  children: React.ReactNode
  className?: string
}

const variantClasses: Record<BadgeVariant, string> = {
  default:
    "bg-[var(--color-bg-inset)] text-[var(--color-text-tertiary)] border-[var(--color-border-default)]",
  success:
    "bg-[var(--color-success-900)] text-[var(--color-success-500)] border-[var(--color-success-500)]/40",
  warning:
    "bg-[var(--color-warning-900)] text-[var(--color-warning-500)] border-[var(--color-warning-500)]/40",
  error:
    "bg-[var(--color-danger-900)] text-[var(--color-danger-500)] border-[var(--color-danger-500)]/40",
  info:
    "bg-[var(--color-info-900)] text-[var(--color-info-500)] border-[var(--color-info-500)]/40",
  cyan:
    "bg-[var(--color-cyan-900)] text-[var(--color-cyan-200)] border-[var(--color-cyan-500)]/40",
  amber:
    "bg-[var(--color-amber-900)] text-[var(--color-amber-200)] border-[var(--color-amber-500)]/40",
}

export default function Badge({
  variant = "default",
  children,
  className = "",
}: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center px-1.5 py-0.5 rounded-[3px] text-[10px] font-medium border uppercase tracking-wider",
        variantClasses[variant],
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </span>
  )
}
