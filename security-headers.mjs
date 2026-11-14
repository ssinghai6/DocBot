// Security response headers for the Next.js frontend (DOCBOT-1619, SEC-11).
//
// Plain ESM (no TypeScript) so `next.config.ts` can import it and the pytest
// suite can evaluate it with a bare `node` binary (tests/unit/test_security_headers.py).
//
// CSP rationale, directive by directive:
// - script-src (DOCBOT-2337): per-request nonce + 'strict-dynamic', no
//   'unsafe-inline'. src/proxy.ts mints the nonce and sets this CSP on both the
//   request (Next reads the nonce from it and stamps its inline RSC/bootstrap
//   scripts) and the response. That forces dynamic rendering (root layout calls
//   connection()). 'strict-dynamic' lets nonced scripts load the Turnstile and
//   Vercel Analytics scripts they inject; the host entries stay as a fallback
//   for CSP2-only browsers, which ignore 'strict-dynamic'. The CSP is therefore
//   NOT in the static next.config headers: a second static policy would be
//   intersected by the browser.
// - 'unsafe-eval' is DEV ONLY (React dev tooling / HMR). Never in production.
// - style-src 'unsafe-inline': React `style={}` props, KaTeX inline sizing
//   styles and next/font's injected @font-face rules.
// - img-src data:/blob:: E2B charts arrive as base64 PNG data URIs; exported
//   files use object URLs. No remote images are loaded.
// - font-src 'self' data:: next/font self-hosts Inter/JetBrains Mono and the
//   KaTeX fonts are bundled from node_modules — no Google Fonts CDN at runtime.
// - connect-src: API calls go through the same-origin /api rewrite; the backend
//   origin (BACKEND_URL, legacy RAILWAY_BACKEND_URL) is listed too so a direct call never silently fails. Vercel Web
//   Analytics and the MSAL (Azure SQL Entra sign-in) token endpoint are allowed.
// - Vercel preview deployments inject the vercel.live feedback toolbar; its
//   origins are allowed only when VERCEL_ENV === "preview".
// - Cloudflare Turnstile (DOCBOT-1713/2315): TurnstileWidget loads
//   challenges.cloudflare.com/turnstile/v0/api.js and the widget renders an
//   iframe from the same origin, so it is in script-src and frame-src (prod and
//   preview). No connect-src entry: the widget talks to Cloudflare from inside
//   its own iframe, not from our page.

const VERCEL_ANALYTICS = ["https://va.vercel-scripts.com"]
const VERCEL_INSIGHTS = ["https://vitals.vercel-insights.com"]
const MSAL_LOGIN = ["https://login.microsoftonline.com"]
const VERCEL_LIVE_SCRIPT = ["https://vercel.live"]
const VERCEL_LIVE_CONNECT = ["https://vercel.live", "wss://ws-us3.pusher.com"]
const VERCEL_LIVE_IMG = ["https://vercel.live", "https://vercel.com"]
const TURNSTILE = ["https://challenges.cloudflare.com"]

export const HSTS_VALUE = "max-age=31536000; includeSubDomains"
export const PERMISSIONS_POLICY =
  "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()"

/**
 * Backend URL for the /api and /v1 rewrites and the CSP connect-src.
 * BACKEND_URL wins; RAILWAY_BACKEND_URL is the legacy name and still works on
 * its own, so existing deploys see no change. A bare host gets https://.
 * @param {Record<string, string | undefined>} [env]
 * @returns {string | undefined}
 */
export function backendUrlFromEnv(env = process.env) {
  const raw = (env.BACKEND_URL ?? "").trim() || (env.RAILWAY_BACKEND_URL ?? "").trim()
  if (!raw) return undefined
  return raw.startsWith("http") ? raw : `https://${raw}`
}

/**
 * Normalize a backend URL (with or without scheme) to its origin, or null.
 * @param {string | undefined | null} raw
 * @returns {string | null}
 */
export function backendOrigin(raw) {
  const trimmed = (raw ?? "").trim()
  if (!trimmed) return null
  const withScheme = trimmed.startsWith("http") ? trimmed : `https://${trimmed}`
  try {
    return new URL(withScheme).origin
  } catch {
    return null
  }
}

/**
 * @param {{ nonce: string, isDev: boolean, isPreview?: boolean, backendUrl?: string | null }} opts
 * @returns {string}
 */
export function buildCsp({ nonce, isDev, isPreview = false, backendUrl = null }) {
  const api = backendOrigin(backendUrl)
  const live = isPreview && !isDev

  /** @type {Record<string, string[]>} */
  const directives = {
    "default-src": ["'self'"],
    "script-src": [
      "'self'",
      `'nonce-${nonce}'`,
      "'strict-dynamic'",
      ...(isDev ? ["'unsafe-eval'"] : []),
      ...VERCEL_ANALYTICS,
      ...TURNSTILE,
      ...(live ? VERCEL_LIVE_SCRIPT : []),
    ],
    "style-src": ["'self'", "'unsafe-inline'"],
    "img-src": ["'self'", "data:", "blob:", ...(live ? VERCEL_LIVE_IMG : [])],
    "font-src": ["'self'", "data:"],
    "media-src": ["'self'", "blob:"],
    "connect-src": [
      "'self'",
      ...(api ? [api] : []),
      ...VERCEL_ANALYTICS,
      ...VERCEL_INSIGHTS,
      ...MSAL_LOGIN,
      ...(live ? VERCEL_LIVE_CONNECT : []),
      // Next dev server HMR websocket.
      ...(isDev ? ["ws:", "http://127.0.0.1:8000", "http://localhost:8000"] : []),
    ],
    "frame-src": ["'self'", ...MSAL_LOGIN, ...TURNSTILE, ...(live ? VERCEL_LIVE_SCRIPT : [])],
    "worker-src": ["'self'", "blob:"],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
  }

  // No `upgrade-insecure-requests`: every source is same-origin or an explicit
  // https:// origin, HSTS already forces https, and the directive would break
  // `next start` on http://localhost.
  return Object.entries(directives)
    .map(([k, v]) => `${k} ${v.join(" ")}`)
    .join("; ")
}

/**
 * Static header list for Next.js `headers()`. CSP is set per request in src/proxy.ts.
 * @returns {{ key: string, value: string }[]}
 */
export function securityHeaders() {
  return [
    { key: "Strict-Transport-Security", value: HSTS_VALUE },
    { key: "X-Frame-Options", value: "DENY" },
    { key: "X-Content-Type-Options", value: "nosniff" },
    { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    { key: "Permissions-Policy", value: PERMISSIONS_POLICY },
  ]
}
