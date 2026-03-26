"use client"

import React from "react"
import Link from "next/link"
import {
  Database,
  FileText,
  Layers,
  Brain,
  ShieldCheck,
  BarChart2,
  ArrowRight,
  CheckCircle2,
  MessageSquare,
  Upload,
  Zap,
  GitBranch,
  Github,
} from "lucide-react"
import { Button, Badge, Card } from "@/components/ui"

// ── Feature data ──────────────────────────────────────────────────────────────

const features = [
  {
    icon: <Layers className="w-6 h-6" />,
    title: "Hybrid Analysis",
    description:
      "Ask questions that span your PDFs and live database simultaneously. Get unified answers with dual citations and automatic discrepancy detection.",
    badge: "Core Feature",
    badgeVariant: "info" as const,
    iconColor: "text-blue-400",
    iconBg: "bg-blue-900/30",
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: "8 Expert Personas",
    description:
      "Specialized AI analysts: Financial, Legal, Medical, Technical, and more. Each persona is tuned for domain-specific reasoning and output conventions.",
    badge: "AI-Powered",
    badgeVariant: "info" as const,
    iconColor: "text-purple-400",
    iconBg: "bg-purple-900/30",
  },
  {
    icon: <Database className="w-6 h-6" />,
    title: "Live Database Chat",
    description:
      "Connect PostgreSQL, MySQL, SQLite, and Azure SQL. Query in plain English through a secure 7-step bounded pipeline — no SQL knowledge required.",
    badge: "4 DB Types",
    badgeVariant: "success" as const,
    iconColor: "text-green-400",
    iconBg: "bg-green-900/30",
  },
  {
    icon: <BarChart2 className="w-6 h-6" />,
    title: "CSV Intelligence",
    description:
      "Upload spreadsheets and get instant pandas-powered analysis via E2B sandboxes. Charts, aggregations, and trend analysis — all from natural language.",
    badge: "E2B Sandbox",
    badgeVariant: "warning" as const,
    iconColor: "text-yellow-400",
    iconBg: "bg-yellow-900/30",
  },
  {
    icon: <GitBranch className="w-6 h-6" />,
    title: "Deep Research",
    description:
      "Multi-step investigation using a LangGraph 5-node state machine: plan, retrieve, evaluate, gap-fill, and synthesize across all your data sources.",
    badge: "LangGraph",
    badgeVariant: "info" as const,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-900/30",
  },
  {
    icon: <ShieldCheck className="w-6 h-6" />,
    title: "Enterprise Security",
    description:
      "RBAC with viewer/analyst/admin roles, append-only audit logging, PII auto-masking, SAML SSO, and Fernet-encrypted credential storage.",
    badge: "WCAG AA",
    badgeVariant: "success" as const,
    iconColor: "text-emerald-400",
    iconBg: "bg-emerald-900/30",
  },
]

// ── Stats data ─────────────────────────────────────────────────────────────────

const stats = [
  { label: "Tests Passing", value: "385+" },
  { label: "AI Personas", value: "8" },
  { label: "Database Types", value: "4" },
  { label: "Deployed On", value: "Railway" },
]

// ── How it works steps ────────────────────────────────────────────────────────

const steps = [
  {
    icon: <Upload className="w-6 h-6" />,
    title: "Upload or Connect",
    description:
      "Upload PDF documents or CSV files, or connect to your live PostgreSQL, MySQL, or SQLite database — no schema expertise needed.",
    color: "from-blue-600 to-blue-700",
    glow: "shadow-blue-900/40",
  },
  {
    icon: <MessageSquare className="w-6 h-6" />,
    title: "Ask in Plain English",
    description:
      "Type your question naturally. DocBot's smart auto-router selects the right AI persona and retrieval strategy for your query automatically.",
    color: "from-purple-600 to-purple-700",
    glow: "shadow-purple-900/40",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "Get Cited Answers",
    description:
      "Receive accurate, grounded responses with source citations from your documents and query results from your database — side by side.",
    color: "from-green-600 to-green-700",
    glow: "shadow-green-900/40",
  },
]

// ── Mock chat messages ────────────────────────────────────────────────────────

type MockMessage = {
  role: "user" | "assistant"
  text: string
  persona?: string
  citations?: number
  loading?: boolean
}

const mockMessages: MockMessage[] = [
  {
    role: "user",
    text: "What was our Q3 revenue vs what the annual report forecasted?",
  },
  {
    role: "assistant",
    text: "Based on your database (Q3 actuals) and the uploaded annual report:\n\n**Database:** Q3 revenue = $4.2M\n**PDF Forecast:** $3.8M projected for Q3\n\nYou exceeded forecast by **10.5%**. Note: the report used conservative assumptions — discrepancy flagged.",
    persona: "Finance Expert",
    citations: 2,
  },
  {
    role: "user",
    text: "Show me a breakdown by product line",
  },
  {
    role: "assistant",
    text: "Running pandas analysis on your CSV upload...",
    persona: "Finance Expert",
    loading: true,
  },
]

// ── Component ─────────────────────────────────────────────────────────────────

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white" style={{ scrollBehavior: "smooth" }}>

      {/* ── Navigation ── */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-gray-800/60 backdrop-blur-md bg-gray-950/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-white text-lg">DocBot</span>
          </div>
          <div className="hidden md:flex items-center gap-6 text-sm text-gray-400">
            <a
              href="#features"
              className="hover:text-white transition-colors duration-150"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="hover:text-white transition-colors duration-150"
            >
              How It Works
            </a>
          </div>
          <Link href="/">
            <Button variant="primary" size="sm">
              Launch App
              <ArrowRight className="w-3.5 h-3.5" />
            </Button>
          </Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="relative pt-32 pb-24 px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Animated gradient background */}
        <div
          className="absolute inset-0 -z-10 pointer-events-none"
          aria-hidden="true"
        >
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] rounded-full opacity-20 blur-3xl bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 animate-pulse" />
          <div className="absolute top-0 right-0 w-96 h-96 rounded-full opacity-10 blur-3xl bg-purple-600" />
          <div className="absolute bottom-0 left-0 w-80 h-80 rounded-full opacity-10 blur-3xl bg-blue-600" />
        </div>

        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: headline and CTAs */}
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-900/30 border border-blue-700/40 text-blue-400 text-sm font-medium mb-6">
                <Zap className="w-3.5 h-3.5" />
                AI-Powered Document + Database Analyst
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight text-white mb-6">
                Your Documents &amp; Databases,{" "}
                <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-blue-300 bg-clip-text text-transparent">
                  Finally Talking to Each Other
                </span>
              </h1>

              <p className="text-lg sm:text-xl text-gray-400 leading-relaxed mb-8 max-w-lg">
                Ask questions in plain English. Get answers from PDFs, live
                databases, and data files — all at once, with citations and
                discrepancy detection.
              </p>

              <div className="flex flex-col sm:flex-row gap-3">
                <Link href="/">
                  <Button variant="primary" size="lg">
                    Start for Free
                    <ArrowRight className="w-5 h-5" />
                  </Button>
                </Link>
                <a href="#how-it-works">
                  <Button variant="secondary" size="lg">
                    See How It Works
                  </Button>
                </a>
              </div>

              <div className="flex items-center gap-4 mt-8 text-sm text-gray-500">
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-green-500" />
                  No SQL knowledge required
                </span>
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-green-500" />
                  WCAG AA accessible
                </span>
              </div>
            </div>

            {/* Right: mock chat interface */}
            <div className="relative">
              <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden shadow-2xl shadow-black/50">
                {/* Chat header */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800 bg-gray-950/50">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500/70" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                    <div className="w-3 h-3 rounded-full bg-green-500/70" />
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <Brain className="w-3.5 h-3.5 text-purple-400" />
                    Finance Expert Persona
                  </div>
                  <Badge variant="success" className="ml-auto">Live</Badge>
                </div>

                {/* Chat messages */}
                <div className="p-4 space-y-4 min-h-[320px]">
                  {mockMessages.map((msg, i) => (
                    <div
                      key={i}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      {msg.role === "assistant" && (
                        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0 mr-2 mt-0.5">
                          <Brain className="w-3.5 h-3.5 text-white" />
                        </div>
                      )}
                      <div
                        className={[
                          "max-w-[78%] rounded-xl px-3 py-2 text-sm",
                          msg.role === "user"
                            ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white"
                            : "bg-gray-800 border border-gray-700 text-gray-200",
                        ].join(" ")}
                      >
                        {msg.loading ? (
                          <div className="flex items-center gap-2 text-gray-400">
                            <span className="flex gap-0.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce [animation-delay:-0.3s]" />
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce [animation-delay:-0.15s]" />
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" />
                            </span>
                            <span className="text-xs">Analyzing...</span>
                          </div>
                        ) : (
                          <p className="whitespace-pre-line leading-relaxed">{msg.text}</p>
                        )}
                        {msg.citations && (
                          <div className="mt-1.5 flex items-center gap-1 text-xs text-gray-500">
                            <FileText className="w-3 h-3" />
                            {msg.citations} sources cited
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Chat input mock */}
                <div className="px-4 pb-4">
                  <div className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2">
                    <span className="flex-1 text-sm text-gray-500">Ask a question about your data...</span>
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center">
                      <ArrowRight className="w-3.5 h-3.5 text-white" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating badges */}
              <div className="absolute -left-4 top-12 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl hidden lg:flex items-center gap-2 text-xs text-gray-300">
                <Database className="w-3.5 h-3.5 text-green-400" />
                PostgreSQL connected
              </div>
              <div className="absolute -right-4 bottom-16 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl hidden lg:flex items-center gap-2 text-xs text-gray-300">
                <FileText className="w-3.5 h-3.5 text-blue-400" />
                3 PDFs indexed
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats bar ── */}
      <section className="border-y border-gray-800 bg-gray-900/40 py-8 px-4">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {stat.value}
              </div>
              <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features grid ── */}
      <section id="features" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge variant="info" className="mb-4">What DocBot Does</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              One platform. Every data source.
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              DocBot unifies your documents and databases into a single conversational
              interface — no data pipelines, no SQL, no switching tools.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((feature) => (
              <Card key={feature.title} hoverable className="group">
                <div className={`w-12 h-12 rounded-xl ${feature.iconBg} ${feature.iconColor} flex items-center justify-center mb-4 transition-transform duration-300 group-hover:scale-110`}>
                  {feature.icon}
                </div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h3 className="font-semibold text-white text-base leading-snug">
                    {feature.title}
                  </h3>
                  <Badge variant={feature.badgeVariant} className="shrink-0 mt-0.5">
                    {feature.badge}
                  </Badge>
                </div>
                <p className="text-sm text-gray-400 leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section id="how-it-works" className="py-24 px-4 sm:px-6 lg:px-8 bg-gray-900/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge variant="success" className="mb-4">Simple by Design</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              From question to answer in three steps
            </h2>
            <p className="text-lg text-gray-400 max-w-xl mx-auto">
              DocBot handles the complexity of multi-source retrieval so you can
              focus on the insights, not the infrastructure.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connector line (desktop) */}
            <div
              className="hidden md:block absolute top-10 left-[calc(16.67%+1rem)] right-[calc(16.67%+1rem)] h-px bg-gradient-to-r from-blue-600/50 via-purple-600/50 to-green-600/50"
              aria-hidden="true"
            />

            {steps.map((step, i) => (
              <div key={step.title} className="relative flex flex-col items-center text-center">
                {/* Step number + icon */}
                <div className={`relative w-20 h-20 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center mb-6 shadow-xl ${step.glow} text-white z-10`}>
                  {step.icon}
                  <span className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-gray-950 border border-gray-700 flex items-center justify-center text-xs font-bold text-gray-400">
                    {i + 1}
                  </span>
                </div>

                <h3 className="text-lg font-semibold text-white mb-3">{step.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed max-w-xs">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA section ── */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        <div className="absolute inset-0 -z-10 pointer-events-none" aria-hidden="true">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-950/20 to-transparent" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full opacity-15 blur-3xl bg-gradient-to-r from-blue-600 to-purple-600" />
        </div>

        <div className="max-w-2xl mx-auto text-center">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-8 shadow-xl shadow-blue-900/40">
            <Zap className="w-8 h-8 text-white" />
          </div>

          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Ready to query your data in plain English?
          </h2>
          <p className="text-lg text-gray-400 mb-10 leading-relaxed">
            Connect your first database or upload a document in under 60 seconds.
            No credit card. No setup scripts.
          </p>

          <Link href="/">
            <Button variant="primary" size="lg" className="text-base px-8 py-3.5">
              Launch DocBot
              <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>

          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-8 text-sm text-gray-500">
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              Free to start
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              Deployed on Railway
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              385+ tests passing
            </span>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-gray-800 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-500">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <FileText className="w-3 h-3 text-white" />
            </div>
            <span className="font-medium text-gray-400">DocBot</span>
            <span className="hidden sm:inline text-gray-700">—</span>
            <span className="hidden sm:inline">Built with Claude Code</span>
          </div>

          <div className="flex items-center gap-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-gray-300 transition-colors duration-150"
              aria-label="GitHub repository"
            >
              <Github className="w-4 h-4" />
              GitHub
            </a>
            <span className="text-gray-700">|</span>
            <span>&copy; 2026 DocBot</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
