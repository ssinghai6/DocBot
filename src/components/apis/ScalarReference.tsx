"use client"

// DOCBOT-3905: Scalar API reference for /v1, bundled from npm (no CDN).
// Security: the test key is passed in config only (never storage, never URL);
// persistAuth is off, and the Scalar agent, MCP CTA, developer tools,
// telemetry and Scalar-hosted fonts/proxy are all disabled so nothing leaves
// docbot.space. Try it hits the page origin, i.e. the same-origin /v1 rewrite.

import { useMemo } from "react"
import { ApiReferenceReact } from "@scalar/api-reference-react"

type Props = {
  /** In-memory test key; null keeps the reference read-only (no Try it). */
  testKey: string | null
  mode: "light" | "dark"
}

export default function ScalarReference({ testKey, mode }: Props) {
  const configuration = useMemo(
    () => ({
      url: "/v1/openapi.json",
      // Spec lists https://docbot.space; use the current origin so preview and
      // local dev also stay same-origin.
      servers: [{ url: window.location.origin, description: "This site (same-origin /v1)" }],
      persistAuth: false,
      telemetry: false,
      withDefaultFonts: false,
      showDeveloperTools: "never" as const,
      agent: { disabled: true },
      mcp: { disabled: true },
      hideClientButton: true,
      hideTestRequestButton: !testKey,
      hideDarkModeToggle: true,
      forceDarkModeState: mode,
      darkMode: mode === "dark",
      documentDownloadType: "json" as const,
      authentication: testKey
        ? { preferredSecurityScheme: "bearerAuth", securitySchemes: { bearerAuth: { token: testKey } } }
        : undefined,
    }),
    [testKey, mode],
  )
  // Remount on key/mode change: Scalar reads auth and forced color mode only at mount.
  return <ApiReferenceReact key={`${testKey ?? "read-only"}-${mode}`} configuration={configuration} />
}
