"use client"

import React from "react"
import Link from "next/link"
import {
  Database,
  Server,
  Cloud,
  HardDrive,
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
  Wand2,
  ShoppingCart,
  XCircle,
  Play,
  Lock,
  ScrollText,
  KeyRound,
  EyeOff,
  FileLock2,
  AlertTriangle,
  Volume2,
  VolumeX,
} from "lucide-react"
import { Button, Badge, Card } from "@/components/ui"

// ── Feature data ──────────────────────────────────────────────────────────────

const features = [
  {
    icon: <Layers className="w-6 h-6" />,
    title: "Hybrid Mode",
    description:
      "Ask questions that span your PDFs and live database simultaneously. Get unified answers with dual citations and automatic discrepancy detection.",
    badge: "Core Feature",
    badgeVariant: "info" as const,
    iconColor: "text-[var(--color-info-500)]",
    iconBg: "bg-[var(--color-info-900)]",
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: "4 Expert Personas",
    description:
      "Specialized AI analysts: Finance Expert, Data Analyst, Strategy Analyst, and Generalist. Each persona is tuned for domain-specific reasoning and output conventions.",
    badge: "AI-Powered",
    badgeVariant: "info" as const,
    iconColor: "text-[var(--color-viz-6)]",
    iconBg: "bg-[var(--color-viz-6)]/15",
  },
  {
    icon: <Database className="w-6 h-6" />,
    title: "Live Database Chat",
    description:
      "Connect PostgreSQL, MySQL, SQLite, and Azure SQL. Query in plain English through a secure 7-step bounded pipeline. No SQL knowledge required.",
    badge: "4 DB Types",
    badgeVariant: "success" as const,
    iconColor: "text-[var(--color-success-500)]",
    iconBg: "bg-[var(--color-success-900)]",
  },
  {
    icon: <BarChart2 className="w-6 h-6" />,
    title: "CSV Intelligence",
    description:
      "Upload spreadsheets and get instant pandas-powered analysis via E2B sandboxes. Charts, aggregations, and trend analysis, all from natural language.",
    badge: "E2B Sandbox",
    badgeVariant: "amber" as const,
    iconColor: "text-[var(--color-amber-500)]",
    iconBg: "bg-[var(--color-amber-900)]",
  },
  {
    icon: <Wand2 className="w-6 h-6" />,
    title: "Autopilot",
    description:
      "Auto-triggers multi-step investigations when it detects complex questions. Plans steps, queries databases, searches documents, runs Python analysis, and synthesizes findings.",
    badge: "Auto-Triggers",
    badgeVariant: "cyan" as const,
    iconColor: "text-[var(--color-cyan-500)]",
    iconBg: "bg-[var(--color-cyan-900)]",
  },
  {
    icon: <ShieldCheck className="w-6 h-6" />,
    title: "Enterprise Security",
    description:
      "RBAC with viewer/analyst/admin roles, append-only audit logging, PII auto-masking, SAML SSO, and Fernet-encrypted credential storage.",
    iconColor: "text-[var(--color-success-500)]",
    iconBg: "bg-[var(--color-success-900)]",
  },
  {
    icon: <ShoppingCart className="w-6 h-6" />,
    title: "Marketplace Connector",
    description:
      "Connect Amazon Seller Central. Sync orders and financials, then query everything with natural language alongside your documents and databases.",
    badge: "Amazon SP-API",
    badgeVariant: "warning" as const,
    iconColor: "text-[var(--color-warning-500)]",
    iconBg: "bg-[var(--color-warning-900)]",
  },
  {
    icon: <MessageSquare className="w-6 h-6" />,
    title: "Conversational Memory",
    description:
      "Follow-up questions work naturally across all data sources. Ask 'show revenue by month' then 'filter to Q1 only'. DocBot remembers context.",
    badge: "All Pipelines",
    badgeVariant: "success" as const,
    iconColor: "text-[var(--color-viz-7)]",
    iconBg: "bg-[var(--color-viz-7)]/15",
  },
]

// ── Stats data ─────────────────────────────────────────────────────────────────
// Outcome-facing claims only — engineering/CI details (test counts, pipeline
// step counts) belong in CLAUDE.md, not marketing copy. The trust-signal strip
// below already covers RBAC/Audit Log/SSO/PII/Encryption.

const stats = [
  { label: "Expert AI Personas", value: "4" },
  { label: "Data Sources", value: "6+" },
  { label: "Database Types", value: "4" },
  { label: "Discrepancy Detection", value: "Automatic" },
]

// ── Trust signal data ─────────────────────────────────────────────────────────
// Every badge here maps to a real, shipped capability (see CLAUDE.md):
//   RBAC              -> api/rbac_service.py (viewer/analyst/admin roles)
//   Audit Log         -> api/audit_service.py (append-only, PostgreSQL immutability trigger)
//   SAML SSO          -> api/auth_service.py (SAML 2.0, Okta/Azure AD)
//   PII Masking       -> PII auto-masking applied at SSE/sandbox/audit response boundaries
//   Encrypted at Rest -> Fernet-encrypted credentials (DB_ENCRYPTION_KEY)
const trustSignals = [
  { icon: <Lock className="w-4 h-4" />, label: "RBAC" },
  { icon: <ScrollText className="w-4 h-4" />, label: "Audit Log" },
  { icon: <KeyRound className="w-4 h-4" />, label: "SAML SSO" },
  { icon: <EyeOff className="w-4 h-4" />, label: "PII Masking" },
  { icon: <FileLock2 className="w-4 h-4" />, label: "Encrypted at Rest" },
]

// ── How it works steps ────────────────────────────────────────────────────────

const steps = [
  {
    icon: <Upload className="w-6 h-6" />,
    title: "Upload or Connect",
    description:
      "Upload PDF documents or CSV files, or connect to your live PostgreSQL, MySQL, or SQLite database. No schema expertise needed.",
    color: "from-[var(--color-cyan-600)] to-[var(--color-cyan-500)]",
    glow: "shadow-[var(--color-cyan-900)]",
  },
  {
    icon: <MessageSquare className="w-6 h-6" />,
    title: "Ask in Plain English",
    description:
      "Type your question naturally. DocBot's smart auto-router selects the right AI persona and retrieval strategy for your query automatically.",
    color: "from-[var(--color-info-500)] to-[var(--color-viz-6)]",
    glow: "shadow-[var(--color-info-900)]",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "Get Cited Answers",
    description:
      "Receive accurate, grounded responses with source citations from your documents and query results from your database, side by side.",
    color: "from-[var(--color-success-500)] to-[var(--color-viz-4)]",
    glow: "shadow-[var(--color-success-900)]",
  },
]

// ── Mock chat messages ────────────────────────────────────────────────────────
// Content mirrors the real sandbox demo fixture (api/demo_service.py,
// "quickbite" dataset) byte-for-byte: QuickBite Inc. FY2025 10-K text chunks
// vs. the matching SQLite tables, including the deliberately planted
// doc-vs-database discrepancies used to showcase discrepancy detection.

type MockDiscrepancy = {
  label: string
  dbValue: string
  docValue: string
}

type MockMessage = {
  role: "user" | "assistant"
  text: string
  persona?: string
  citations?: number
  discrepancy?: MockDiscrepancy
}

const mockMessages: MockMessage[] = [
  {
    role: "user",
    text: "What was Q4 net income according to the database vs. the annual report?",
  },
  {
    role: "assistant",
    text: "Cross-checked the `quarterly` table against the QuickBite-10K-2025.pdf filing for Q4 2025:\n\n**Database:** $325M\n**10-K filing (p.7):** $330M",
    persona: "Finance Expert",
    citations: 2,
    discrepancy: {
      label: "Q4 2025 Net Income",
      dbValue: "$325M (database)",
      docValue: "$330M (10-K, p.7)",
    },
  },
  {
    role: "user",
    text: "Are there any other discrepancies?",
  },
  {
    role: "assistant",
    text: "Yes. Restaurant Advertising revenue also differs between sources:\n\n**Database:** $885M\n**10-K filing (p.5):** $890M",
    persona: "Finance Expert",
    citations: 2,
    discrepancy: {
      label: "Restaurant Advertising Revenue",
      dbValue: "$885M (database)",
      docValue: "$890M (10-K, p.5)",
    },
  },
]

// ── Demo Video Component ─────────────────────────────────────────────────────
// Set DEMO_VIDEO_URL to a direct MP4 URL, YouTube embed URL, or leave empty for placeholder.
// Examples:
//   MP4:     "https://your-cdn.com/docbot-demo.mp4"
//   YouTube: "https://www.youtube.com/embed/VIDEO_ID"
//   Loom:    "https://www.loom.com/embed/VIDEO_ID"
const DEMO_VIDEO_URL: string = "/docbot-demo.mp4"

// Burned-in-style captions, timed ~2s apart over the 8s loop. The video is
// muted by default for most visitors, so the message has to land without sound.
const DEMO_CAPTIONS = [
  "Ask your data anything.",
  "Docs. Databases. One answer.",
  "$885M in the DB. $890M in the 10-K.",
  "DocBot catches what you'd miss.",
]

function DemoVideo() {
  const [isPlaying, setIsPlaying] = React.useState(false)
  const [isMuted, setIsMuted] = React.useState(true)
  const [captionIndex, setCaptionIndex] = React.useState(0)
  const videoRef = React.useRef<HTMLVideoElement>(null)

  // YouTube / Loom / Vimeo embed
  if (DEMO_VIDEO_URL && (DEMO_VIDEO_URL.includes("youtube") || DEMO_VIDEO_URL.includes("loom") || DEMO_VIDEO_URL.includes("vimeo"))) {
    return (
      <div className="aspect-video rounded-2xl overflow-hidden border border-[var(--color-border-default)] shadow-2xl shadow-black/50">
        <iframe
          src={DEMO_VIDEO_URL}
          className="w-full h-full"
          allow="autoplay; fullscreen; picture-in-picture"
          allowFullScreen
          title="DocBot Demo"
        />
      </div>
    )
  }

  // Direct MP4 video
  if (DEMO_VIDEO_URL) {
    const toggleMute = (e: React.MouseEvent) => {
      e.stopPropagation()
      if (!videoRef.current) return
      const nextMuted = !videoRef.current.muted
      videoRef.current.muted = nextMuted
      setIsMuted(nextMuted)
    }

    return (
      <div className="aspect-video rounded-2xl overflow-hidden border border-[var(--color-border-default)] shadow-2xl shadow-black/50 relative group">
        <video
          ref={videoRef}
          src={DEMO_VIDEO_URL}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
          loop
          onPlay={() => setIsPlaying(true)}
          onTimeUpdate={(e) => {
            const t = e.currentTarget.currentTime
            setCaptionIndex(Math.min(DEMO_CAPTIONS.length - 1, Math.floor(t / 2)))
          }}
          onClick={() => {
            if (videoRef.current?.paused) {
              videoRef.current.play()
              setIsPlaying(true)
            } else {
              videoRef.current?.pause()
              setIsPlaying(false)
            }
          }}
        />

        {/* Burned-in-style caption bar */}
        <div
          className="absolute inset-x-0 bottom-0 flex items-center justify-center px-6 py-4 bg-gradient-to-t from-black/85 via-black/55 to-transparent pointer-events-none"
          aria-hidden="true"
        >
          <p className="text-white text-base sm:text-lg font-semibold text-center leading-snug drop-shadow-md">
            {DEMO_CAPTIONS[captionIndex]}
          </p>
        </div>

        {/* Mute / unmute toggle */}
        <button
          type="button"
          onClick={toggleMute}
          aria-label={isMuted ? "Unmute video" : "Mute video"}
          className="absolute top-3 right-3 z-10 w-9 h-9 rounded-full bg-black/60 border border-white/20 flex items-center justify-center text-white hover:bg-black/80 transition-colors duration-150"
        >
          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
        </button>

        {!isPlaying && (
          <div
            className="absolute inset-0 flex items-center justify-center bg-black/30 cursor-pointer"
            onClick={() => {
              videoRef.current?.play()
              setIsPlaying(true)
            }}
          >
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[var(--color-cyan-500)] to-[var(--color-info-500)] flex items-center justify-center shadow-2xl shadow-black/40 group-hover:scale-110 transition-transform duration-300">
              <Play className="w-8 h-8 text-white ml-1" />
            </div>
          </div>
        )}
      </div>
    )
  }

  return null
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)]" style={{ scrollBehavior: "smooth" }}>

      {/* ── Navigation ── */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-[var(--color-border-subtle)] backdrop-blur-md bg-[var(--color-bg-base)]/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--color-cyan-500)] to-[var(--color-info-500)] flex items-center justify-center">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-[var(--color-text-primary)] text-lg">DocBot</span>
          </div>
          <div className="hidden md:flex items-center gap-6 text-sm text-[var(--color-text-secondary)]">
            <a
              href="#features"
              className="hover:text-[var(--color-text-primary)] transition-colors duration-150"
            >
              Features
            </a>
            <a
              href="#integrations"
              className="hover:text-[var(--color-text-primary)] transition-colors duration-150"
            >
              Integrations
            </a>
            <a
              href="#how-it-works"
              className="hover:text-[var(--color-text-primary)] transition-colors duration-150"
            >
              How It Works
            </a>
          </div>
          <Link href="/chat">
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
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] rounded-full opacity-20 blur-3xl bg-gradient-to-r from-[var(--color-cyan-600)] via-[var(--color-info-500)] to-[var(--color-cyan-900)] animate-pulse" />
          <div className="absolute top-0 right-0 w-96 h-96 rounded-full opacity-10 blur-3xl bg-[var(--color-info-500)]" />
          <div className="absolute bottom-0 left-0 w-80 h-80 rounded-full opacity-10 blur-3xl bg-[var(--color-cyan-500)]" />
        </div>

        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: headline and CTAs */}
            <div>
              <Badge variant="info" className="mb-6 text-sm px-3 py-1 gap-2 normal-case tracking-normal">
                <Zap className="w-3.5 h-3.5" />
                Enterprise Document Intelligence
              </Badge>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight text-[var(--color-text-primary)] mb-6">
                Your Documents &amp; Databases,{" "}
                <span className="bg-gradient-to-r from-[var(--color-cyan-500)] via-[var(--color-info-500)] to-[var(--color-cyan-200)] bg-clip-text text-transparent">
                  Finally Talking to Each Other
                </span>
              </h1>

              <p className="text-lg sm:text-xl text-[var(--color-text-secondary)] leading-relaxed mb-8 max-w-lg">
                The only platform that cross-references your documents against live databases in real time. Get cited answers, catch discrepancies automatically, and connect your marketplace data. All through plain English.
              </p>

              <div className="flex flex-col sm:flex-row gap-3">
                <Link href="/chat">
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

              <div className="flex items-center gap-4 mt-8 text-sm text-[var(--color-text-tertiary)]">
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-[var(--color-success-500)]" />
                  No SQL knowledge required
                </span>
              </div>
            </div>

            {/* Right: mock chat interface */}
            <div className="relative">
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-default)] rounded-2xl overflow-hidden shadow-2xl shadow-black/50">
                {/* Chat header */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-base)]/50">
                  <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
                    <Brain className="w-3.5 h-3.5 text-[var(--color-viz-6)]" />
                    Finance Expert Persona
                  </div>
                  <Badge variant="default" className="ml-auto">Example</Badge>
                </div>

                {/* Chat messages */}
                <div className="p-4 space-y-4 min-h-[320px]">
                  {mockMessages.map((msg, i) => (
                    <div
                      key={i}
                      className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      {msg.role === "assistant" && (
                        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[var(--color-cyan-500)] to-[var(--color-info-500)] flex items-center justify-center shrink-0 mr-2 mt-0.5">
                          <Brain className="w-3.5 h-3.5 text-white" />
                        </div>
                      )}
                      <div
                        className={[
                          "max-w-[78%] rounded-xl px-3 py-2 text-sm",
                          msg.role === "user"
                            ? "bg-gradient-to-r from-[var(--color-cyan-600)] to-[var(--color-info-500)] text-white"
                            : "bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] text-[var(--color-text-secondary)]",
                        ].join(" ")}
                      >
                        <p className="whitespace-pre-line leading-relaxed">{msg.text}</p>
                        {msg.discrepancy && (
                          <div className="mt-2 rounded-[5px] border border-[var(--color-warning-500)]/40 bg-[var(--color-warning-500)]/5 p-2.5">
                            <div className="flex items-center gap-1.5 mb-1.5">
                              <AlertTriangle className="w-3.5 h-3.5 text-[var(--color-warning-500)] shrink-0" />
                              <span className="text-[10px] font-semibold text-[var(--color-warning-500)] uppercase tracking-wider">
                                Discrepancy Detected
                              </span>
                            </div>
                            <div className="text-xs text-[var(--color-text-secondary)] font-mono">
                              <div className="font-medium text-[var(--color-text-primary)]">{msg.discrepancy.label}</div>
                              <div className="mt-0.5">DB: {msg.discrepancy.dbValue}</div>
                              <div>Doc: {msg.discrepancy.docValue}</div>
                            </div>
                          </div>
                        )}
                        {msg.citations && (
                          <div className="mt-1.5 flex items-center gap-1 text-xs text-[var(--color-text-tertiary)]">
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
                  <div className="flex items-center gap-2 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-xl px-3 py-2">
                    <span className="flex-1 text-sm text-[var(--color-text-tertiary)]">Ask a question about your data...</span>
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-r from-[var(--color-cyan-600)] to-[var(--color-info-500)] flex items-center justify-center">
                      <ArrowRight className="w-3.5 h-3.5 text-white" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating badges */}
              <div className="absolute -left-4 top-12 bg-[var(--color-bg-surface)] border border-[var(--color-border-default)] rounded-lg px-3 py-2 shadow-xl hidden lg:flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                <Database className="w-3.5 h-3.5 text-[var(--color-success-500)]" />
                PostgreSQL connected
              </div>
              <div className="absolute -right-4 bottom-16 bg-[var(--color-bg-surface)] border border-[var(--color-border-default)] rounded-lg px-3 py-2 shadow-xl hidden lg:flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                <FileText className="w-3.5 h-3.5 text-[var(--color-info-500)]" />
                3 PDFs indexed
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats bar ── */}
      <section className="border-y border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)]/60 py-8 px-4">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-[var(--color-cyan-500)] to-[var(--color-info-500)] bg-clip-text text-transparent">
                {stat.value}
              </div>
              <div className="text-sm text-[var(--color-text-tertiary)] mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Trust signals ── */}
      <section className="py-6 px-4 sm:px-6 lg:px-8 border-b border-[var(--color-border-subtle)]">
        <div className="max-w-5xl mx-auto">
          <p className="text-center text-xs text-[var(--color-text-tertiary)] mb-4 uppercase tracking-wider font-medium">
            Built for Compliance Review
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            {trustSignals.map((signal) => (
              <div
                key={signal.label}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[var(--color-success-900)] border border-[var(--color-success-500)]/40 text-[var(--color-success-500)] text-xs font-medium"
              >
                {signal.icon}
                {signal.label}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Integrations ── */}
      <section id="integrations" className="py-12 px-4 sm:px-6 lg:px-8 border-b border-[var(--color-border-subtle)]">
        <div className="max-w-5xl mx-auto">
          <p className="text-center text-sm text-[var(--color-text-tertiary)] mb-8 uppercase tracking-wider font-medium">
            Integrates With
          </p>
          <div className="flex flex-wrap items-center justify-center gap-8 md:gap-12">
            {[
              { icon: <Database className="w-5 h-5" />, label: "PostgreSQL" },
              { icon: <Server className="w-5 h-5" />, label: "MySQL" },
              { icon: <Cloud className="w-5 h-5" />, label: "Azure SQL" },
              { icon: <HardDrive className="w-5 h-5" />, label: "SQLite" },
              { icon: <ShoppingCart className="w-5 h-5" />, label: "Amazon SP-API" },
              { icon: <BarChart2 className="w-5 h-5" />, label: "E2B Sandbox" },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors">
                {item.icon}
                <span className="text-xs font-medium">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features grid ── */}
      <section id="features" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge variant="info" className="mb-4">What DocBot Does</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--color-text-primary)] mb-4">
              Enterprise-grade intelligence across every data source
            </h2>
            <p className="text-lg text-[var(--color-text-secondary)] max-w-2xl mx-auto">
              DocBot unifies your documents and databases into a single conversational
              interface. No data pipelines, no SQL, no switching tools.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((feature) => (
              <Card key={feature.title} hoverable className="p-6 group">
                <div className={`w-12 h-12 rounded-xl ${feature.iconBg} ${feature.iconColor} flex items-center justify-center mb-4 transition-transform duration-300 group-hover:scale-110`}>
                  {feature.icon}
                </div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h3 className="font-semibold text-[var(--color-text-primary)] text-base leading-snug">
                    {feature.title}
                  </h3>
                  {feature.badge && (
                    <Badge variant={feature.badgeVariant} className="shrink-0 mt-0.5">
                      {feature.badge}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* ── Comparison ── */}
      <section className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <Badge variant="info" className="mb-4">Why DocBot?</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--color-text-primary)] mb-4">
              The only platform that does it all
            </h2>
            <p className="text-lg text-[var(--color-text-secondary)] max-w-2xl mx-auto">
              Other tools handle one source at a time. DocBot cross-references both in real time.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border-default)]">
                  <th className="text-left py-4 px-4 text-[var(--color-text-secondary)] font-medium">Capability</th>
                  <th className="py-4 px-4 text-[var(--color-text-secondary)] font-medium">ChatGPT</th>
                  <th className="py-4 px-4 text-[var(--color-text-secondary)] font-medium">Custom RAG</th>
                  <th className="py-4 px-4 text-[var(--color-text-secondary)] font-medium">BI Tools</th>
                  <th className="py-4 px-4 text-[var(--color-text-primary)] font-bold bg-gradient-to-r from-[var(--color-cyan-900)] to-[var(--color-info-900)] rounded-t-lg">DocBot</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-text-secondary)]">
                {[
                  ["Document Q&A with citations", false, true, false, true],
                  ["Live database queries", false, false, true, true],
                  ["Cross-source discrepancy detection", false, false, false, true],
                  ["Marketplace data sync", false, false, false, true],
                  ["Multi-step auto-investigation", false, false, false, true],
                  ["No infrastructure to manage", true, false, false, true],
                ].map(([label, ...values], i) => (
                  <tr key={i} className="border-b border-[var(--color-border-subtle)]">
                    <td className="py-3 px-4 text-[var(--color-text-secondary)]">{label as string}</td>
                    {(values as boolean[]).map((v, j) => (
                      <td key={j} className={`py-3 px-4 text-center ${j === 3 ? "bg-gradient-to-r from-[var(--color-cyan-900)]/40 to-[var(--color-info-900)]/40" : ""}`}>
                        {v ? <CheckCircle2 className="w-5 h-5 text-[var(--color-success-500)] mx-auto" /> : <XCircle className="w-5 h-5 text-[var(--color-text-quaternary)] mx-auto" />}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section id="how-it-works" className="py-24 px-4 sm:px-6 lg:px-8 bg-[var(--color-bg-surface)]/40">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge variant="success" className="mb-4">Simple by Design</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--color-text-primary)] mb-4">
              From question to answer in three steps
            </h2>
            <p className="text-lg text-[var(--color-text-secondary)] max-w-xl mx-auto">
              DocBot handles the complexity of multi-source retrieval so you can
              focus on the insights, not the infrastructure.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connector line (desktop) */}
            <div
              className="hidden md:block absolute top-10 left-[calc(16.67%+1rem)] right-[calc(16.67%+1rem)] h-px bg-gradient-to-r from-[var(--color-cyan-500)]/50 via-[var(--color-info-500)]/50 to-[var(--color-success-500)]/50"
              aria-hidden="true"
            />

            {steps.map((step, i) => (
              <div key={step.title} className="relative flex flex-col items-center text-center">
                {/* Step number + icon */}
                <div className={`relative w-20 h-20 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center mb-6 shadow-xl ${step.glow} text-white z-10`}>
                  {step.icon}
                  <span className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-[var(--color-bg-base)] border border-[var(--color-border-default)] flex items-center justify-center text-xs font-bold text-[var(--color-text-tertiary)]">
                    {i + 1}
                  </span>
                </div>

                <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-3">{step.title}</h3>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed max-w-xs">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Demo Video ── */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <Badge variant="info" className="mb-4">See It In Action</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--color-text-primary)] mb-4">
              Watch DocBot analyze real data
            </h2>
            <p className="text-[var(--color-text-secondary)] text-lg max-w-xl mx-auto">
              Upload a case study, ask complex financial questions, and get cited answers with calculations in seconds.
            </p>
          </div>
          <DemoVideo />
        </div>
      </section>

      {/* ── CTA section ── */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        <div className="absolute inset-0 -z-10 pointer-events-none" aria-hidden="true">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[var(--color-cyan-900)]/20 to-transparent" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full opacity-15 blur-3xl bg-gradient-to-r from-[var(--color-cyan-600)] to-[var(--color-info-500)]" />
        </div>

        <div className="max-w-2xl mx-auto text-center">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[var(--color-cyan-500)] to-[var(--color-info-500)] flex items-center justify-center mx-auto mb-8 shadow-xl shadow-black/40">
            <Zap className="w-8 h-8 text-white" />
          </div>

          <h2 className="text-3xl sm:text-4xl font-bold text-[var(--color-text-primary)] mb-4">
            Stop switching between tools. Start getting answers.
          </h2>
          <p className="text-lg text-[var(--color-text-secondary)] mb-10 leading-relaxed">
            Connect your first data source in under 60 seconds. No credit card required.
          </p>

          <Link href="/chat">
            <Button variant="primary" size="lg" className="text-base px-8 py-3.5">
              Launch DocBot
              <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>

          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-8 text-sm text-[var(--color-text-tertiary)]">
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-[var(--color-success-500)]" />
              Free to start
            </span>
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-[var(--color-success-500)]" />
              Enterprise-ready from day one
            </span>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-[var(--color-border-subtle)] py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-[var(--color-text-tertiary)]">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-[var(--color-cyan-500)] to-[var(--color-info-500)] flex items-center justify-center">
              <FileText className="w-3 h-3 text-white" />
            </div>
            <span className="font-medium text-[var(--color-text-secondary)]">DocBot</span>
            <span className="hidden sm:inline text-[var(--color-text-quaternary)]">|</span>
            <span className="hidden sm:inline">AI-Powered Document Intelligence</span>
          </div>

          <div className="flex items-center gap-4">
            <span>&copy; 2026 DocBot</span>
            <span className="text-[var(--color-text-quaternary)]">|</span>
            <span>Built by <a href="https://sanshrit-singhai.vercel.app/" target="_blank" rel="noopener noreferrer" className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors duration-150">Sanshrit Singhai</a></span>
          </div>
        </div>
      </footer>
    </div>
  )
}
