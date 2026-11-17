// Public site origin for canonical/OG/sitemap/JSON-LD (DOCBOT-2536).
// NEXT_PUBLIC_SITE_URL wins. Without it, production builds (prod and preview) use the live
// canonical host, so preview hashes and localhost never leak into canonicals or the sitemap.
// docbot.space 308-redirects to www, so www is the canonical host.
const host =
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.NODE_ENV === "production" ? "https://www.docbot.space" : "http://localhost:3000")
export const SITE_URL = (host.startsWith("http") ? host : `https://${host}`).replace(/\/+$/, "")

export const SITE_DESCRIPTION =
  "DocBot reconciles your client's PDFs against their live books and flags every number that doesn't tie out, with a citation for each side."
