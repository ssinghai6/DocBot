import type { MetadataRoute } from "next"
import { SITE_URL } from "@/lib/site"

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: ["/chat", "/api/", "/onboarding", "/verify"] },
    sitemap: `${SITE_URL}/sitemap.xml`,
  }
}
