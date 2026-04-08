import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
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
  title: "DocBot — AI Document + Database Analyst",
  description:
    "Analytical-grade AI that chats with your PDFs, queries live databases, and runs Python sandboxes. Hybrid Docs+DB synthesis with discrepancy detection.",
  keywords: ["AI", "analytics", "document", "database", "SQL", "hybrid", "finance", "investor"],
  authors: [{ name: "Sanshrit Singhai" }],
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  openGraph: {
    title: "DocBot — AI Document + Database Analyst",
    description: "Analytical-grade AI for docs, databases, and hybrid analysis.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
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
