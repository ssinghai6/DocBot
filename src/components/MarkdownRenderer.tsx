"use client"

/**
 * MarkdownRenderer (DOCBOT-2006) — the single renderer for every AI response.
 *
 * Used by chat messages, autopilot step output and the inspector views so all
 * model text gets identical formatting:
 * - GFM (tables, task lists, strikethrough) with tables in a keyboard-reachable
 *   horizontal scroll region and zebra rows (styles in globals.css `.md-*`).
 * - Fenced code blocks highlighted via prism-react-renderer, with a copy button.
 * - KaTeX math (remark-math + rehype-katex), delimiters normalized first.
 * - Safety: raw HTML skipped, links open in a new tab with noopener/noreferrer,
 *   images only rendered for `data:image/` and `blob:` sources.
 */

import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import ReactMarkdown, {
  defaultUrlTransform,
  type Components,
  type ExtraProps,
  type Options as MarkdownOptions,
  type UrlTransform,
} from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import { Highlight, themes } from "prism-react-renderer"
import { Check, Copy, TriangleAlert } from "lucide-react"

import { normalizeLatexDelimiters } from "@/lib/latex"
import { tieOutMetric, type TieOutCard } from "@/lib/schemas/tieout"
import { useUIStore } from "@/store/uiStore"
import type { Lineage } from "./types"

// ── Plugins (module scope → stable identity across renders) ────────────────

const remarkPlugins: MarkdownOptions["remarkPlugins"] = [
  remarkGfm,
  [remarkMath, { singleDollarTextMath: false }],
]
const rehypePlugins: MarkdownOptions["rehypePlugins"] = [rehypeKatex]

// ── URL policy ──────────────────────────────────────────────────────────────

const SAFE_IMAGE_SRC = /^(?:data:image\/[a-z0-9.+-]+[;,]|blob:)/i

export function isSafeImageSrc(src: unknown): src is string {
  return typeof src === "string" && SAFE_IMAGE_SRC.test(src.trim())
}

/** Images: only data:/blob:. Everything else: react-markdown's safe default. */
const urlTransform: UrlTransform = (url, key, node) => {
  if (key === "src" && node.tagName === "img") {
    return isSafeImageSrc(url) ? url : ""
  }
  return defaultUrlTransform(url)
}

// ── DOCBOT-2002: inline TieOutCard badges ──────────────────────────────────

// Marker tokens the synthesis prompts place in the answer text; each maps to
// a structured TieOutCard in the lineage payload.
const TIE_OUT_MARKER_RE = /\[\[discrepancy:(d-[0-9a-f]{8})\]\]/g
// Legacy literal tags (pre-2002 answers) never render as raw text.
const LEGACY_TAG_RE = /\[\s*\/?\s*DISCREPANCY\s*\]/gi
const TIE_OUT_HREF = "#tieout-"

/**
 * Rewrite marker tokens into fragment links (rendered as badges by the `a`
 * override, inline within the paragraph) and strip legacy literal tags.
 */
function preprocessTieOut(content: string, discrepancies?: TieOutCard[]): string {
  if (!content || !content.toLowerCase().includes("discrepan")) return content
  let out = content.replace(LEGACY_TAG_RE, "**Discrepancy:**")
  const hasCards = !!discrepancies?.length
  out = out.replace(TIE_OUT_MARKER_RE, (_m, id: string) => {
    const card = hasCards ? discrepancies?.find((d) => d.id === id) : undefined
    const label = card
      ? `${tieOutMetric(card)}${card.period ? ` (${card.period})` : ""}`
      : "Tie-out"
    return `[${label.replace(/[[\]]/g, "")}](${TIE_OUT_HREF}${id})`
  })
  return out
}

function TieOutBadge({ id, lineage }: { id: string; lineage?: Lineage | null }) {
  const selectLineage = useUIStore((s) => s.selectLineage)
  const setInspectorOpen = useUIStore((s) => s.setInspectorOpen)
  const card = lineage?.discrepancies.find((d) => d.id === id)
  const label = card ? tieOutMetric(card) : "Tie-out"
  const title = card
    ? `${label} — report ${card.report_value ?? card.doc_value ?? "?"} vs books ${card.books_value ?? card.db_value ?? "?"}`
    : "Doc vs books tie-out"
  return (
    <button
      type="button"
      disabled={!lineage}
      onClick={() => {
        if (!lineage) return
        selectLineage(lineage, { tab: "lineage", focusDiscrepancyId: id })
        setInspectorOpen(true)
      }}
      title={title}
      className="md-tieout-badge"
    >
      <TriangleAlert className="w-2.5 h-2.5" aria-hidden />
      <span className="max-w-[180px] truncate">{label}</span>
    </button>
  )
}

// ── hast helpers ───────────────────────────────────────────────────────────

type HastElement = NonNullable<ExtraProps["node"]>
type HastChild = HastElement["children"][number]

function hastText(nodes: HastChild[]): string {
  let out = ""
  for (const n of nodes) {
    if (n.type === "text") out += n.value
    else if (n.type === "element") out += hastText(n.children)
  }
  return out
}

function codeLanguage(code: HastElement): string {
  const cls: unknown = code.properties?.className // @types/hast 3.0.5 types it string[]; keep the string guard
  const list = Array.isArray(cls) ? cls : typeof cls === "string" ? cls.split(" ") : []
  for (const c of list) {
    if (typeof c === "string" && c.startsWith("language-")) return c.slice("language-".length).toLowerCase()
  }
  return ""
}

// prism-react-renderer bundles a subset of Prism grammars; map common aliases.
const LANGUAGE_ALIASES: Record<string, string> = {
  py: "python",
  js: "javascript",
  ts: "typescript",
  sh: "bash",
  shell: "bash",
  zsh: "bash",
  yml: "yaml",
  html: "markup",
  xml: "markup",
  postgres: "sql",
  postgresql: "sql",
  mysql: "sql",
  sqlite: "sql",
  tsql: "sql",
}

// ── Copy button ────────────────────────────────────────────────────────────

function CodeCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current)
  }, [])

  const onCopy = useCallback(() => {
    if (!navigator.clipboard) return
    navigator.clipboard.writeText(text).then(
      () => {
        setCopied(true)
        if (timer.current) clearTimeout(timer.current)
        timer.current = setTimeout(() => setCopied(false), 1500)
      },
      () => setCopied(false),
    )
  }, [text])

  return (
    <button
      type="button"
      onClick={onCopy}
      aria-label={copied ? "Copied" : "Copy code"}
      className="md-code-copy"
    >
      {copied ? <Check className="w-3 h-3" aria-hidden /> : <Copy className="w-3 h-3" aria-hidden />}
      <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  )
}

// ── Element overrides ──────────────────────────────────────────────────────

function CodeBlock({ code, language }: { code: string; language: string }) {
  const prismLanguage = LANGUAGE_ALIASES[language] ?? (language || "text")
  return (
    <div className="md-code">
      <div className="md-code-header">
        <span className="md-code-lang">{language || "text"}</span>
        <CodeCopyButton text={code} />
      </div>
      <Highlight code={code} language={prismLanguage} theme={themes.vsDark}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={className} style={{ ...style, backgroundColor: "transparent" }}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => <span key={key} {...getTokenProps({ token })} />)}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  )
}

function Pre({ node, children }: React.ComponentPropsWithoutRef<"pre"> & ExtraProps) {
  const codeEl = node?.children.find(
    (c): c is HastElement => c.type === "element" && c.tagName === "code",
  )
  if (!codeEl) return <pre>{children}</pre>
  const code = hastText(codeEl.children).replace(/\n$/, "")
  return <CodeBlock code={code} language={codeLanguage(codeEl)} />
}

function Table({ children }: React.ComponentPropsWithoutRef<"table"> & ExtraProps) {
  return (
    <div className="md-table-wrap" role="region" tabIndex={0} aria-label="Data table (scrolls horizontally)">
      <table>{children}</table>
    </div>
  )
}

function Image({ src, alt, title }: React.ComponentPropsWithoutRef<"img"> & ExtraProps) {
  if (!isSafeImageSrc(src)) return null
  // eslint-disable-next-line @next/next/no-img-element -- inline data:/blob: images, next/image can't optimize these
  return <img src={src} title={title} alt={alt ?? ""} loading="lazy" />
}

function buildComponents(lineage: Lineage | null): Components {
  return {
    pre: Pre,
    table: Table,
    a: ({ children, href, title }) => {
      // DOCBOT-2002: fragment links produced by preprocessTieOut render as
      // inline badges that focus their TieOutCard in the inspector.
      if (typeof href === "string" && href.startsWith(TIE_OUT_HREF)) {
        return <TieOutBadge id={href.slice(TIE_OUT_HREF.length)} lineage={lineage} />
      }
      return (
        <a href={href} title={title} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      )
    },
    img: Image,
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export interface MarkdownRendererProps {
  /** Raw markdown from the model. */
  content: string
  /** `compact` = denser type scale for autopilot steps / inspector panes. */
  variant?: "default" | "compact"
  className?: string
  /**
   * DOCBOT-2002: lineage of the answer being rendered — its structured
   * discrepancies turn inline `[[discrepancy:<id>]]` markers into TieOutCard
   * badges that focus the card in the inspector. Omit for non-answer text.
   */
  lineage?: Lineage | null
}

function MarkdownRenderer({ content, variant = "default", className, lineage }: MarkdownRendererProps) {
  const components = useMemo(() => buildComponents(lineage ?? null), [lineage])
  const classes = ["prose", "max-w-none", variant === "compact" ? "prose-compact" : "", className ?? ""]
    .filter(Boolean)
    .join(" ")
  return (
    <div className={classes}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={components}
        urlTransform={urlTransform}
        skipHtml
      >
        {normalizeLatexDelimiters(preprocessTieOut(content, lineage?.discrepancies))}
      </ReactMarkdown>
    </div>
  )
}

export default memo(MarkdownRenderer)
