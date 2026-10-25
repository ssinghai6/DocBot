import type { NextConfig } from "next";
import { securityHeaders } from "./security-headers.mjs";

const rawUrl = process.env.RAILWAY_BACKEND_URL?.trim();
const railwayBackendUrl = rawUrl && !rawUrl.startsWith("http") ? `https://${rawUrl}` : rawUrl;

if (process.env.NODE_ENV === "production" && !railwayBackendUrl) {
  // Hard error at build time — there is no Python serverless fallback.
  // The entire backend runs on Railway. Set RAILWAY_BACKEND_URL in Vercel project settings.
  throw new Error(
    "[next.config.ts] RAILWAY_BACKEND_URL is not set. " +
      "All API requests proxy to the Railway backend — this env var is required for production builds. " +
      "Add it in Vercel → Project Settings → Environment Variables."
  );
}

const nextConfig: NextConfig = {
  // Required for Docker standalone build (DOCBOT-605)
  output: process.env.DOCKER_BUILD ? "standalone" : undefined,
  // DOCBOT-1619: CSP + HSTS + framing/nosniff/referrer/permissions headers.
  // Policy and rationale live in security-headers.mjs.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders({
          isDev: process.env.NODE_ENV === "development",
          isPreview: process.env.VERCEL_ENV === "preview",
          backendUrl: railwayBackendUrl,
        }),
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/api/:path*"
            : `${railwayBackendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
