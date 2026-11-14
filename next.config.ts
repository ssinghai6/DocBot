import type { NextConfig } from "next";
import { backendUrlFromEnv, securityHeaders } from "./security-headers.mjs";

// BACKEND_URL, falling back to the legacy RAILWAY_BACKEND_URL (host-neutral).
const backendUrl = backendUrlFromEnv();

if (process.env.NODE_ENV === "production" && !backendUrl) {
  // Hard error at build time — there is no Python serverless fallback.
  // The entire backend runs on a separate host (Railway today, AWS later).
  throw new Error(
    "[next.config.ts] BACKEND_URL (or legacy RAILWAY_BACKEND_URL) is not set. " +
      "All API requests proxy to the backend — this env var is required for production builds. " +
      "Add it in Vercel → Project Settings → Environment Variables."
  );
}

const nextConfig: NextConfig = {
  // Required for Docker standalone build (DOCBOT-605)
  output: process.env.DOCKER_BUILD ? "standalone" : undefined,
  // DOCBOT-1619: HSTS + framing/nosniff/referrer/permissions headers.
  // Policy and rationale live in security-headers.mjs.
  async headers() {
    return [
      {
        source: "/:path*",
        // CSP is per-request (nonce) in src/proxy.ts, not here (DOCBOT-2337).
        headers: securityHeaders(),
      },
    ];
  },
  async rewrites() {
    const backend =
      process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : backendUrl;
    // DOCBOT-2339: /v1/* (Developer API, DOCBOT-1530) proxies exactly like /api/*.
    return ["api", "v1"].map((prefix) => ({
      source: `/${prefix}/:path*`,
      destination: `${backend}/${prefix}/:path*`,
    }));
  },
};

export default nextConfig;
