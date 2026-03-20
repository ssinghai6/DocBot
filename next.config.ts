import type { NextConfig } from "next";

const rawUrl = process.env.RAILWAY_BACKEND_URL?.trim();
const railwayBackendUrl = rawUrl && !rawUrl.startsWith("http") ? `https://${rawUrl}` : rawUrl;

if (process.env.NODE_ENV === "production" && !railwayBackendUrl) {
  console.warn(
    "[next.config.ts] WARNING: RAILWAY_BACKEND_URL is not set. " +
      "API requests will fall back to the Vercel serverless function. " +
      "Set RAILWAY_BACKEND_URL in Vercel project settings to route to Railway."
  );
}

const productionDestination = railwayBackendUrl
  ? `${railwayBackendUrl}/api/:path*`
  : "/api/index";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/api/:path*"
            : productionDestination,
      },
    ];
  },
};

export default nextConfig;
