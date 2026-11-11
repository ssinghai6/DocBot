// Public site origin for canonical/OG/sitemap/JSON-LD (DOCBOT-2536).
// NEXT_PUBLIC_SITE_URL wins; Vercel exposes the production domain (no scheme) as
// NEXT_PUBLIC_VERCEL_PROJECT_PRODUCTION_URL, so preview hashes never leak into canonicals.
const host = process.env.NEXT_PUBLIC_SITE_URL || process.env.NEXT_PUBLIC_VERCEL_PROJECT_PRODUCTION_URL
export const SITE_URL = host ? (host.startsWith("http") ? host : `https://${host}`) : "http://localhost:3000"

export const SITE_DESCRIPTION =
  "DocBot reconciles your client's PDFs against their live books and flags every number that doesn't tie out, with a citation for each side."
