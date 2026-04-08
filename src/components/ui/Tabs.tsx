"use client"

import * as RadixTabs from "@radix-ui/react-tabs"
import type { ReactNode } from "react"

interface TabsProps {
  value: string
  onValueChange: (value: string) => void
  children: ReactNode
  className?: string
}

export function Tabs({ value, onValueChange, children, className = "" }: TabsProps) {
  return (
    <RadixTabs.Root value={value} onValueChange={onValueChange} className={className}>
      {children}
    </RadixTabs.Root>
  )
}

interface TabsListProps {
  children: ReactNode
  className?: string
}

export function TabsList({ children, className = "" }: TabsListProps) {
  return (
    <RadixTabs.List
      className={[
        "flex items-center gap-0 border-b border-[var(--color-border-subtle)]",
        className,
      ].join(" ")}
    >
      {children}
    </RadixTabs.List>
  )
}

interface TabsTriggerProps {
  value: string
  children: ReactNode
  className?: string
}

export function TabsTrigger({ value, children, className = "" }: TabsTriggerProps) {
  return (
    <RadixTabs.Trigger
      value={value}
      className={[
        "relative h-8 px-3 text-[12px] font-medium transition-colors duration-[var(--duration-fast)] cursor-pointer",
        "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]",
        "data-[state=active]:text-[var(--color-text-primary)]",
        "data-[state=active]:after:content-[''] data-[state=active]:after:absolute data-[state=active]:after:left-0 data-[state=active]:after:right-0 data-[state=active]:after:-bottom-px data-[state=active]:after:h-[2px] data-[state=active]:after:bg-[var(--color-cyan-500)]",
        className,
      ].join(" ")}
    >
      {children}
    </RadixTabs.Trigger>
  )
}

interface TabsContentProps {
  value: string
  children: ReactNode
  className?: string
}

export function TabsContent({ value, children, className = "" }: TabsContentProps) {
  return (
    <RadixTabs.Content value={value} className={className}>
      {children}
    </RadixTabs.Content>
  )
}
