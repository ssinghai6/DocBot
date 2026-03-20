import type { NextConfig } from "next";

const railwayBackendUrl = process.env.RAILWAY_BACKEND_URL;

if (process.env.NODE_ENV === "production" && !railwayBackendUrl) {
  throw new Error(
    "[next.config.ts] RAILWAY_BACKEND_URL is not set. " +
      "Set it as a build-time environment variable in your Vercel project settings."
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
