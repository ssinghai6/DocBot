"use client"

import React from "react"

type BadgeVariant = "default" | "success" | "warning" | "error" | "info"

interface BadgeProps {
  variant?: BadgeVariant
  children: React.ReactNode
  className?: string
}

const variantClasses: Record<BadgeVariant, string> = {
  default: "bg-gray-700 text-gray-300 border-gray-600",
  success: "bg-green-900/40 text-green-400 border-green-700/50",
  warning: "bg-yellow-900/40 text-yellow-400 border-yellow-700/50",
  error: "bg-red-900/40 text-red-400 border-red-700/50",
  info: "bg-blue-900/40 text-blue-400 border-blue-700/50",
}

export default function Badge({
  variant = "default",
  children,
  className = "",
}: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border",
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
