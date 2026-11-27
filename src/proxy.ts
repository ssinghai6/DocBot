// DOCBOT-2337: per-request CSP nonce. Policy lives in security-headers.mjs.
import { NextResponse, type NextRequest } from "next/server";
import { backendUrlFromEnv, buildCsp } from "../security-headers.mjs";

export function proxy(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const csp = buildCsp({
    nonce,
    isDev: process.env.NODE_ENV === "development",
    isPreview: process.env.VERCEL_ENV === "preview",
    backendUrl: backendUrlFromEnv(),
  });
  // Next reads the nonce from the request CSP and applies it to its scripts.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("Content-Security-Policy", csp);
  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", csp);
  return response;
}

export const config = {
  matcher: [
    {
      source: "/((?!api/|v1/|admin/|_next/static|_next/image|favicon.ico).*)",
      missing: [
        { type: "header", key: "next-router-prefetch" },
        { type: "header", key: "purpose", value: "prefetch" },
      ],
    },
  ],
};
