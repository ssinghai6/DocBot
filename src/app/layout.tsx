import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import { connection } from "next/server";
import { SITE_DESCRIPTION, SITE_URL } from "@/lib/site";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "DocBot | Reconcile client PDFs against live books",
  description: SITE_DESCRIPTION,
  keywords: ["tie-out", "financial statement reconciliation", "month-end close", "fractional CFO software", "10-K vs general ledger"],
  // "./" resolves per route, so every page self-canonicalizes.
  alternates: { canonical: "./" },
  authors: [{ name: "Sanshrit Singhai" }],
  icons: {
    icon: [
      { url: "/favicon-small.svg", sizes: "16x16", type: "image/svg+xml" },
      { url: "/favicon.svg", sizes: "any", type: "image/svg+xml" },
      { url: "/favicon.ico", sizes: "16x16 32x32 48x48" },
    ],
    shortcut: "/favicon.ico",
    apple: "/logo-hybrid-light.svg",
  },
  openGraph: {
    title: "DocBot | Reconcile client PDFs against live books",
    description: SITE_DESCRIPTION,
    url: "./",
    siteName: "DocBot",
    type: "website",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "DocBot | Reconcile client PDFs against live books",
    description: SITE_DESCRIPTION,
    images: ["/og-image.png"],
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Nonce CSP (DOCBOT-2337): render per request so Next can stamp the nonce.
  await connection();
  return (
    <html lang="en" className="dark" data-theme="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
      >
        {children}
        <Analytics />
      </body>
    </html>
  );
}
