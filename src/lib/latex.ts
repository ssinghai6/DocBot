/**
 * Converts LLM-emitted LaTeX delimiter variants to remark-math syntax
 * BEFORE markdown parsing so math renders instead of leaking as raw text
 * (and so underscores inside formulas survive GFM emphasis parsing).
 *
 * Handle with care: anything between ``` fences is left untouched, and
 * conversions are gated on the text actually looking like LaTeX so
 * markdown links, [DISCREPANCY] tags, and currency are never mangled.
 *
 * remark-math runs with singleDollarTextMath:false, so ALL math here is
 * emitted in the two-dollar form: micromark-extension-math has no currency
 * guard and would otherwise pair "between $4.2M and $5.1B" as one formula.
 *
 * - `\[ ... \]` display  -> fenced `$$ ... $$` block
 * - `\( ... \)` inline   -> `$$ ... $$` math text
 * - `[ \text{...} ]` (bare-bracket variant seen in the wild, line-scoped)
 *                        -> fenced `$$ ... $$` block
 */
const LATEX_COMMAND_HINT = /\\(?:text|frac|times|div|sum|prod|int|overline|underline|sqrt|left|right|begin|end|bar|hat|vec|alpha|beta|gamma|delta|lambda|mu|pi|sigma|omega|Delta|Sigma|Omega|Pi|Theta|Lambda|Phi|cdot|infty|approx|leq|geq|neq|pm|to|rightarrow|partial|lim|max|min|log|ln|exp|sin|cos|tan)/

export function normalizeLatexDelimiters(content: string): string {
  return content
    .split("```")
    .map((segment, index) => (index % 2 === 0 ? normalizeSegment(segment) : segment))
    .join("```")
}

function normalizeSegment(segment: string): string {
  const display = segment.replace(
    /\\\[([\s\S]*?)\\\]/g,
    (match, inner: string) =>
      LATEX_COMMAND_HINT.test(inner) ? `\n$$\n${inner.trim()}\n$$\n` : match
  )
  const inline = display.replace(
    /\\\(([\s\S]*?)\\\)/g,
    (match, inner: string) =>
      LATEX_COMMAND_HINT.test(inner) ? `$$${inner.trim()}$$` : match
  )
  return inline.replace(
    /^[ \t]*\[([^\n]*?)[ \t]*\][ \t]*$/gm,
    (match, inner: string) =>
      LATEX_COMMAND_HINT.test(inner) ? `\n$$\n${inner.trim()}\n$$\n` : match
  )
}
