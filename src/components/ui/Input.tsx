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
}: InputProps) {
  const inputId = id ?? (label ? label.toLowerCase().replace(/\s+/g, "-") : undefined)

  return (
    <div className={["flex flex-col gap-1", className].filter(Boolean).join(" ")}>
      {label && (
        <label
          htmlFor={inputId}
          className="text-sm font-medium text-gray-300"
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
        aria-invalid={!!error}
        aria-describedby={error ? `${inputId}-error` : undefined}
        className={[
          "w-full px-3 py-2 rounded-lg text-white placeholder-gray-500 text-sm",
          "bg-gray-800 border transition-all duration-150",
          error
            ? "border-red-500 focus:border-red-400 focus:ring-2 focus:ring-red-500/20"
            : "border-gray-700 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20",
          disabled ? "opacity-50 cursor-not-allowed" : "",
          "outline-none",
        ]
          .filter(Boolean)
          .join(" ")}
      />
      {error && (
        <p id={`${inputId}-error`} className="text-xs text-red-400 mt-0.5" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
