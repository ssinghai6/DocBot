"use client"

import React from "react"
import { Loader2 } from "lucide-react"

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "amber"
type ButtonSize = "xs" | "sm" | "md" | "lg"

interface ButtonProps {
  variant?: ButtonVariant
  size?: ButtonSize
  disabled?: boolean
  loading?: boolean
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void
  children: React.ReactNode
  className?: string
  type?: "button" | "submit" | "reset"
  title?: string
  "aria-label"?: string
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--color-cyan-500)] text-[var(--color-text-inverse)] border border-[var(--color-cyan-500)] hover:bg-[var(--color-cyan-600)] hover:border-[var(--color-cyan-600)]",
  secondary:
    "bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] border border-[var(--color-border-default)] hover:bg-[var(--color-bg-overlay)] hover:border-[var(--color-border-strong)]",
  ghost:
    "bg-transparent text-[var(--color-text-secondary)] border border-transparent hover:bg-[var(--color-bg-overlay)] hover:text-[var(--color-text-primary)]",
  danger:
    "bg-[var(--color-danger-900)] text-[var(--color-danger-500)] border border-[var(--color-danger-500)] hover:bg-[var(--color-danger-500)] hover:text-[var(--color-text-inverse)]",
  amber:
    "bg-[var(--color-amber-500)] text-[var(--color-text-inverse)] border border-[var(--color-amber-500)] hover:bg-[var(--color-amber-600)] hover:border-[var(--color-amber-600)]",
}

const sizeClasses: Record<ButtonSize, string> = {
  xs: "h-6 px-2 text-[11px] rounded-[5px] gap-1",
  sm: "h-7 px-2.5 text-xs rounded-[5px] gap-1.5",
  md: "h-8 px-3 text-[13px] rounded-[5px] gap-1.5",
  lg: "h-10 px-4 text-sm rounded-[6px] gap-2",
}

export default function Button({
  variant = "primary",
  size = "md",
  disabled = false,
  loading = false,
  onClick,
  children,
  className = "",
  type = "button",
  title,
  "aria-label": ariaLabel,
}: ButtonProps) {
  const isDisabled = disabled || loading

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={isDisabled}
      title={title}
      aria-label={ariaLabel}
      className={[
        "inline-flex items-center justify-center font-medium whitespace-nowrap",
        "transition-colors duration-[var(--duration-fast)] cursor-pointer",
        variantClasses[variant],
        sizeClasses[size],
        isDisabled ? "opacity-40 cursor-not-allowed pointer-events-none" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {loading && <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" aria-hidden="true" />}
      {children}
    </button>
  )
}
