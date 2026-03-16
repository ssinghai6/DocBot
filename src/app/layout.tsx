import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DocBot - AI Document Assistant",
  description: "Chat with your PDF documents using AI. Upload documents, ask questions, and get instant answers powered by Llama 3.3",
  keywords: ["AI", "document", "PDF", "chat", "Llama", "assistant", "OCR"],
  authors: [{ name: "Sanshrit Singhai" }],
  openGraph: {
    title: "DocBot - AI Document Assistant",
    description: "Chat with your PDF documents using AI",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
