import type { MetadataRoute } from "next"
import { SITE_URL } from "@/lib/site"

// App routes are disallowed in robots.ts, so they are not listed.
export default function sitemap(): MetadataRoute.Sitemap {
  return ["", "/signup", "/privacy", "/terms"].map((path) => ({ url: `${SITE_URL}${path}` }))
}
