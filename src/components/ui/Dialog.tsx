"use client"

import * as RadixDialog from "@radix-ui/react-dialog"
import { X } from "lucide-react"
import type { ReactNode } from "react"

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: ReactNode
}

export function Dialog({ open, onOpenChange, children }: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      {children}
    </RadixDialog.Root>
  )
}

interface DialogContentProps {
  children: ReactNode
  className?: string
  title?: string
  description?: string
  showClose?: boolean
}

export function DialogContent({
  children,
  className = "",
  title,
  description,
  showClose = true,
}: DialogContentProps) {
  return (
    <RadixDialog.Portal>
      <RadixDialog.Overlay
        className="fixed inset-0 z-[100] bg-[var(--bg-scrim)] backdrop-blur-sm data-[state=open]:animate-in data-[state=open]:fade-in-0"
      />
      <RadixDialog.Content
        className={[
          "fixed left-1/2 top-[20vh] z-[101] -translate-x-1/2",
          "w-full max-w-lg",
          "bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-[12px]",
          "shadow-[var(--elev-4)]",
          "p-5",
          className,
        ].join(" ")}
      >
        {title && (
          <RadixDialog.Title className="text-[15px] font-semibold text-[var(--color-text-primary)] mb-1">
            {title}
          </RadixDialog.Title>
        )}
        {description && (
          <RadixDialog.Description className="text-[12px] text-[var(--color-text-tertiary)] mb-4">
            {description}
          </RadixDialog.Description>
        )}
        {children}
        {showClose && (
          <RadixDialog.Close
            aria-label="Close"
            className="absolute top-3 right-3 p-1 rounded-[5px] text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-overlay)] transition-colors duration-[var(--duration-fast)]"
          >
            <X className="w-4 h-4" />
          </RadixDialog.Close>
        )}
      </RadixDialog.Content>
    </RadixDialog.Portal>
  )
}

export const DialogTrigger = RadixDialog.Trigger
export const DialogClose = RadixDialog.Close
