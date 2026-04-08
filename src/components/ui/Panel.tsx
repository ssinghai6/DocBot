"use client"

import { Group, Panel as RPPanel, Separator } from "react-resizable-panels"
import type { ComponentProps } from "react"

export type { PanelImperativeHandle } from "react-resizable-panels"

/**
 * Thin wrappers around react-resizable-panels v4 API.
 *
 *   <PanelGroup>  →  Group    (was PanelGroup in v0.x)
 *   <Panel>       →  Panel
 *   <ResizeHandle>→  Separator (was PanelResizeHandle in v0.x)
 */

export const PanelGroup = Group
export const Panel = RPPanel

type SeparatorProps = ComponentProps<typeof Separator>

export function ResizeHandle(props: SeparatorProps) {
  const { className = "", ...rest } = props
  return (
    <Separator
      {...rest}
      className={[
        "group relative w-[4px] shrink-0 bg-transparent cursor-col-resize transition-colors duration-[var(--duration-fast)]",
        "hover:bg-[var(--color-cyan-500)]/20 data-[resizing=true]:bg-[var(--color-cyan-500)]/40",
        className,
      ].join(" ")}
    >
      <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-[var(--color-border-subtle)] group-hover:bg-[var(--color-cyan-500)] group-data-[resizing=true]:bg-[var(--color-cyan-500)]" />
    </Separator>
  )
}
