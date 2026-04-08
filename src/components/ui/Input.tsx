"use client"

import React from "react"

interface InputProps {
  label?: string
  placeholder?: string
  value?: string
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void
  type?: React.HTMLInputTypeAttribute
  error?: string
  disabled?: boolean
  className?: string
  id?: string
  autoFocus?: boolean
}

export default function Input({
  label,
  placeholder,
  value,
  onChange,
  type = "text",
  error,
  disabled = false,
  className = "",
  id,
  autoFocus,
}: InputProps) {
  const inputId = id ?? (label ? label.toLowerCase().replace(/\s+/g, "-") : undefined)

  return (
    <div className={["flex flex-col gap-1.5", className].filter(Boolean).join(" ")}>
      {label && (
        <label
          htmlFor={inputId}
          className="text-[11px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider"
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        autoFocus={autoFocus}
        aria-invalid={!!error}
        aria-describedby={error ? `${inputId}-error` : undefined}
        className={[
          "w-full px-3 h-9 rounded-[5px] text-[13px]",
          "bg-[var(--color-bg-inset)] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-quaternary)]",
          "border transition-colors duration-[var(--duration-fast)] outline-none",
          error
            ? "border-[var(--color-danger-500)] focus:border-[var(--color-danger-500)]"
            : "border-[var(--color-border-default)] focus:border-[var(--color-cyan-500)]",
          disabled ? "opacity-40 cursor-not-allowed" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      />
      {error && (
        <p id={`${inputId}-error`} className="text-[11px] text-[var(--color-danger-500)]" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
