"use client"

import React from "react"

type TooltipPosition = "top" | "bottom" | "left" | "right"

interface TooltipProps {
  content: string
  children: React.ReactNode
  position?: TooltipPosition
}

const positionClasses: Record<TooltipPosition, { tooltip: string; arrow: string }> = {
  top: {
    tooltip: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    arrow: "top-full left-1/2 -translate-x-1/2 border-t-gray-700 border-x-transparent border-b-transparent border-4",
  },
  bottom: {
    tooltip: "top-full left-1/2 -translate-x-1/2 mt-2",
    arrow: "bottom-full left-1/2 -translate-x-1/2 border-b-gray-700 border-x-transparent border-t-transparent border-4",
  },
  left: {
    tooltip: "right-full top-1/2 -translate-y-1/2 mr-2",
    arrow: "left-full top-1/2 -translate-y-1/2 border-l-gray-700 border-y-transparent border-r-transparent border-4",
  },
  right: {
    tooltip: "left-full top-1/2 -translate-y-1/2 ml-2",
    arrow: "right-full top-1/2 -translate-y-1/2 border-r-gray-700 border-y-transparent border-l-transparent border-4",
  },
}

export default function Tooltip({
  content,
  children,
  position = "top",
}: TooltipProps) {
  const pos = positionClasses[position]

  return (
    <div className="relative inline-flex group">
      {children}
      <div
        role="tooltip"
        className={[
          "absolute z-50 pointer-events-none",
          "px-2 py-1 text-xs font-medium text-white whitespace-nowrap",
          "bg-gray-700 border border-gray-600 rounded-md shadow-lg",
          "opacity-0 group-hover:opacity-100 transition-opacity duration-150",
          pos.tooltip,
        ].join(" ")}
      >
        {content}
        <span
          className={[
            "absolute border",
            pos.arrow,
          ].join(" ")}
          aria-hidden="true"
        />
      </div>
    </div>
  )
}
