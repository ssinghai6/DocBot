"use client"

import React from "react"
import * as RadixTooltip from "@radix-ui/react-tooltip"

type TooltipPosition = "top" | "bottom" | "left" | "right"

interface TooltipProps {
  content: React.ReactNode
  children: React.ReactNode
  position?: TooltipPosition
  delayDuration?: number
}

export default function Tooltip({
  content,
  children,
  position = "top",
  delayDuration = 300,
}: TooltipProps) {
  return (
    <RadixTooltip.Provider delayDuration={delayDuration}>
      <RadixTooltip.Root>
        <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
        <RadixTooltip.Portal>
          <RadixTooltip.Content
            side={position}
            sideOffset={6}
            className="z-50 px-2 py-1 text-[11px] font-medium rounded-[5px] border bg-[var(--color-bg-overlay)] text-[var(--color-text-primary)] border-[var(--color-border-default)] shadow-lg select-none pointer-events-none"
          >
            {content}
            <RadixTooltip.Arrow className="fill-[var(--color-border-default)]" />
          </RadixTooltip.Content>
        </RadixTooltip.Portal>
      </RadixTooltip.Root>
    </RadixTooltip.Provider>
  )
}
