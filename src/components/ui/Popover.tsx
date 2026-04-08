"use client"

import * as RadixPopover from "@radix-ui/react-popover"
import type { ReactNode } from "react"

interface PopoverProps {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  children: ReactNode
}

export function Popover({ open, onOpenChange, children }: PopoverProps) {
  return (
    <RadixPopover.Root open={open} onOpenChange={onOpenChange}>
      {children}
    </RadixPopover.Root>
  )
}

export const PopoverTrigger = RadixPopover.Trigger

interface PopoverContentProps {
  children: ReactNode
  className?: string
  side?: "top" | "right" | "bottom" | "left"
  align?: "start" | "center" | "end"
  sideOffset?: number
}

export function PopoverContent({
  children,
  className = "",
  side = "bottom",
  align = "start",
  sideOffset = 6,
}: PopoverContentProps) {
  return (
    <RadixPopover.Portal>
      <RadixPopover.Content
        side={side}
        align={align}
        sideOffset={sideOffset}
        className={[
          "z-[90] min-w-[200px]",
          "bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-[8px]",
          "shadow-[var(--elev-3)] p-1",
          className,
        ].join(" ")}
      >
        {children}
      </RadixPopover.Content>
    </RadixPopover.Portal>
  )
}
