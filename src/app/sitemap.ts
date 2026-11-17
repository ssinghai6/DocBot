import type { MetadataRoute } from "next"
import { SITE_URL } from "@/lib/site"

// /landing duplicates / and app routes are disallowed in robots.ts, so neither is listed.
export default function sitemap(): MetadataRoute.Sitemap {
  return ["", "/signup", "/privacy", "/terms"].map((path) => ({ url: `${SITE_URL}${path}` }))
}
