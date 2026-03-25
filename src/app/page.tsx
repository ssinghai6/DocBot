"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { z } from "zod"
import {
  Send, Upload, File, Loader2, Trash2,
  Sparkles, Brain, BookOpen, Stethoscope,
  TrendingUp, Code, Cpu, Scale, Briefcase, BarChart2,
  X, CheckCircle2, AlertCircle, Zap, ChevronDown,
  FileText, MessageSquare, Settings, Keyboard,
  Maximize2, Minimize2, Copy, Check, Info,
  Clock, Hash, Wand2, Layers, ArrowRight,
  Terminal, Database, Eye, EyeOff, RefreshCw,
  Menu, XCircle, AlertTriangle, HelpCircle,
  Download, FileJson, FileText as FileTxt,
  UserCog, LogOut, ShieldCheck, Users, ClipboardList,
  Filter, ChevronRight, Shield
} from "lucide-react"

// ── Zod schemas for API response validation ──────────────────────────────────

const AuthMeSchema = z.object({
  id: z.string(),
  email: z.string(),
  name: z.string(),
  role: z.enum(["viewer", "analyst", "admin"]),
  provider: z.string(),
});
type AuthUser = z.infer<typeof AuthMeSchema>;

const AdminUserSchema = z.object({
  id: z.string(),
  email: z.string(),
  name: z.string(),
  role: z.enum(["viewer", "analyst", "admin"]),
  provider: z.string(),
  last_login_at: z.string().nullable(),
  created_at: z.string().nullable(),
});
type AdminUser = z.infer<typeof AdminUserSchema>;

const AdminUsersResponseSchema = z.object({
  count: z.number(),
  users: z.array(AdminUserSchema),
});

const AuditEventSchema = z.object({
  id: z.string(),
  event_type: z.string(),
  session_id: z.string().nullable(),
  user_id: z.string().nullable(),
  detail: z.string().nullable(),
  metadata_json: z.string().nullable(),
  occurred_at: z.string().nullable(),
});
type AuditEvent = z.infer<typeof AuditEventSchema>;

const AuditLogResponseSchema = z.object({
  count: z.number(),
  events: z.array(AuditEventSchema),
});

// ── Workspace schemas (persistent sessions + DB connections per user) ─────────

const WorkspaceSessionSchema = z.object({
  session_id: z.string(),
  created_at: z.string().nullable(),
  file_count: z.number(),
  persona: z.string(),
});
type WorkspaceSession = z.infer<typeof WorkspaceSessionSchema>;

const WorkspaceConnectionSchema = z.object({
  id: z.string(),
  dialect: z.string(),
  host: z.string(),
  db_name: z.string(),
  created_at: z.string().nullable(),
});
type WorkspaceConnection = z.infer<typeof WorkspaceConnectionSchema>;

const WorkspaceSchema = z.object({
  sessions: z.array(WorkspaceSessionSchema),
  db_connections: z.array(WorkspaceConnectionSchema),
});

type Citation = {
  source: string;
  page: number;
  text: string;
};

type ChartMeta = {
  type: string;
  title: string;
  x_label: string;
  y_label: string;
  series_count: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp?: Date;
  citations?: Citation[];
  charts?: string[];           // base64 PNG strings from E2B analysis
  chartMetas?: ChartMeta[];    // DOCBOT-305: metadata per chart
  analysisCode?: string;       // Python code block, collapsible
  sql?: string;                // SQL query from metadata chunk
  explanation?: string;        // SQL explanation from metadata chunk
  autopilotSteps?: AutopilotStep[];  // DOCBOT-405: persisted investigation steps
  agentPersona?: string;       // DOCBOT-802: which persona handled this message
  agentPersonas?: string[];    // DOCBOT-802: for hybrid messages with multiple personas
};

type Toast = {
  id: string;
  type: "success" | "error" | "info" | "warning";
  message: string;
};

type FileUploadState = "idle" | "dragover" | "uploading" | "success" | "error";

// DOCBOT-504: Query History
type QueryHistoryItem = {
  id: string;
  question: string;
  sql: string;
  executed_at: string | null;
  row_count: number | null;
};

// DOCBOT-405: Autopilot step result from SSE stream
type AutopilotStep = {
  step_num: number;
  tool: string;
  step_label: string;
  content: string;
  artifact_id?: string | null;
  chart_b64?: string | null;
  error?: string | null;
};

const EXPERT_PERSONAS: Record<string, {
  icon: React.ReactNode;
  description: string;
  color: string;
  gradient: string;
  accent: string;
  response_format: "finance" | "clinical" | "legal" | "technical" | "research" | "consulting" | "data" | "general";
  detection_keywords: { primary: string[]; secondary: string[] };
  tool_preference: "sql_first" | "rag_first" | "balanced";
  output_conventions: {
    number_format: "currency" | "percentage" | "raw";
    disclaimer_position: "header" | "footer" | null;
    highlight_pattern: string | null;
    accent_color: string;
  };
}> = {
  Generalist: {
    icon: <Sparkles className="w-5 h-5" />,
    description: "Balanced, general-purpose assistant for any document",
    color: "text-[#667eea]",
    gradient: "from-[#667eea] to-[#764ba2]",
    accent: "#667eea",
    response_format: "general",
    detection_keywords: { primary: [], secondary: [] },
    tool_preference: "balanced",
    output_conventions: { number_format: "raw", disclaimer_position: null, highlight_pattern: null, accent_color: "#667eea" },
  },
  Doctor: {
    icon: <Stethoscope className="w-5 h-5" />,
    description: "Medical & healthcare perspective - analyze clinical documents",
    color: "text-[#10b981]",
    gradient: "from-[#10b981] to-[#059669]",
    accent: "#10b981",
    response_format: "clinical",
    detection_keywords: {
      primary: ["diagnosis", "patient", "clinical", "symptom", "treatment", "prescription", "dosage", "pathology", "surgery", "chronic", "medication", "lab result", "vital"],
      secondary: ["health", "medical", "hospital", "therapy", "disease", "physician", "nursing", "drug"],
    },
    tool_preference: "rag_first",
    output_conventions: { number_format: "raw", disclaimer_position: "header", highlight_pattern: "\\b(WARNING|CRITICAL|CONTRAINDICATED|ABNORMAL|RED FLAG)\\b", accent_color: "#10b981" },
  },
  "Finance Expert": {
    icon: <TrendingUp className="w-5 h-5" />,
    description: "Financial & investment analysis - parse reports & statements",
    color: "text-[#f59e0b]",
    gradient: "from-[#f59e0b] to-[#d97706]",
    accent: "#f59e0b",
    response_format: "finance",
    detection_keywords: {
      primary: ["revenue", "profit", "ebitda", "balance sheet", "cash flow", "earnings", "quarterly", "annual report", "valuation", "roi", "equity", "debt", "dividend", "fiscal", "margin"],
      secondary: ["financial", "investment", "forecast", "budget", "growth", "expense", "asset", "liability", "audit", "fund"],
    },
    tool_preference: "sql_first",
    output_conventions: { number_format: "currency", disclaimer_position: "footer", highlight_pattern: null, accent_color: "#f59e0b" },
  },
  Engineer: {
    icon: <Code className="w-5 h-5" />,
    description: "Technical & engineering focus - documentation & specs",
    color: "text-[#3b82f6]",
    gradient: "from-[#3b82f6] to-[#2563eb]",
    accent: "#3b82f6",
    response_format: "technical",
    detection_keywords: {
      primary: ["specification", "architecture", "api", "circuit", "firmware", "schematic", "protocol", "bandwidth", "latency", "deployment", "infrastructure", "algorithm", "system design", "mechanical", "structural"],
      secondary: ["technical", "engineering", "component", "interface", "dependency", "compliance", "standard", "tolerance", "performance"],
    },
    tool_preference: "balanced",
    output_conventions: { number_format: "raw", disclaimer_position: null, highlight_pattern: null, accent_color: "#3b82f6" },
  },
  "AI/ML Expert": {
    icon: <Cpu className="w-5 h-5" />,
    description: "AI, ML & data science insights - research papers & models",
    color: "text-[#8b5cf6]",
    gradient: "from-[#8b5cf6] to-[#7c3aed]",
    accent: "#8b5cf6",
    response_format: "research",
    detection_keywords: {
      primary: ["neural network", "transformer", "llm", "embedding", "gradient", "fine-tuning", "training data", "overfitting", "accuracy", "benchmark", "dataset", "classification", "nlp", "computer vision"],
      secondary: ["machine learning", "deep learning", "artificial intelligence", "model", "inference", "pipeline", "feature", "epoch", "loss function", "attention"],
    },
    tool_preference: "balanced",
    output_conventions: { number_format: "percentage", disclaimer_position: null, highlight_pattern: null, accent_color: "#8b5cf6" },
  },
  Lawyer: {
    icon: <Scale className="w-5 h-5" />,
    description: "Legal analysis & compliance - contracts & policies",
    color: "text-[#ef4444]",
    gradient: "from-[#ef4444] to-[#dc2626]",
    accent: "#ef4444",
    response_format: "legal",
    detection_keywords: {
      primary: ["contract", "agreement", "clause", "jurisdiction", "indemnity", "liability", "plaintiff", "defendant", "arbitration", "statute", "copyright", "patent", "gdpr", "compliance"],
      secondary: ["legal", "regulation", "policy", "obligation", "intellectual property", "breach", "penalty", "dispute"],
    },
    tool_preference: "rag_first",
    output_conventions: { number_format: "raw", disclaimer_position: "footer", highlight_pattern: "\\b(RISK|WARNING|VOID|BREACH|PENALTY|PROHIBITED|LIMITATION OF LIABILITY)\\b", accent_color: "#ef4444" },
  },
  Consultant: {
    icon: <Briefcase className="w-5 h-5" />,
    description: "Strategic business advisory - strategy & planning",
    color: "text-[#06b6d4]",
    gradient: "from-[#06b6d4] to-[#0891b2]",
    accent: "#06b6d4",
    response_format: "consulting",
    detection_keywords: {
      primary: ["strategy", "roadmap", "kpi", "go-to-market", "swot", "stakeholder", "competitive analysis", "market share", "transformation", "business case"],
      secondary: ["consulting", "business plan", "proposal", "operational", "market analysis", "management", "growth", "change management"],
    },
    tool_preference: "balanced",
    output_conventions: { number_format: "raw", disclaimer_position: null, highlight_pattern: null, accent_color: "#06b6d4" },
  },
  "Data Analyst": {
    icon: <BarChart2 className="w-5 h-5" />,
    description: "Quantitative analysis with full SQL transparency and data quality flags",
    color: "text-[#f97316]",
    gradient: "from-[#f97316] to-[#ea580c]",
    accent: "#f97316",
    response_format: "data",
    detection_keywords: {
      primary: ["query", "sql", "table", "row", "column", "count", "average", "group by", "join", "aggregate", "null", "outlier", "distribution", "chart"],
      secondary: ["data", "database", "metric", "percentage", "total", "filter", "report", "dashboard", "correlation", "summarize"],
    },
    tool_preference: "sql_first",
    output_conventions: { number_format: "raw", disclaimer_position: null, highlight_pattern: "\\b(NULL|ERROR|WARNING|OUTLIER|MISSING)\\b", accent_color: "#f97316" },
  },
};

// Toast Component
function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`flex items-start gap-3 p-4 rounded-xl border backdrop-blur-xl shadow-lg animate-in slide-in-from-right duration-300 ${toast.type === "success" ? "bg-[#10b981]/10 border-[#10b981]/30 text-[#10b981]" :
            toast.type === "error" ? "bg-[#ef4444]/10 border-[#ef4444]/30 text-[#ef4444]" :
              toast.type === "warning" ? "bg-[#f59e0b]/10 border-[#f59e0b]/30 text-[#f59e0b]" :
                "bg-[#3b82f6]/10 border-[#3b82f6]/30 text-[#3b82f6]"
            }`}
        >
          {toast.type === "success" && <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "error" && <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "warning" && <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />}
          {toast.type === "info" && <Info className="w-5 h-5 shrink-0 mt-0.5" />}
          <p className="text-sm font-medium flex-1 text-gray-200">{toast.message}</p>
          <button onClick={() => onDismiss(toast.id)} className="shrink-0 hover:opacity-70 transition-opacity">
            <X className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}

// Typing Indicator Component
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-4 py-3 bg-[#12121a]/80 border border-[#ffffff08] rounded-2xl rounded-bl-sm">
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-[#667eea] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 bg-[#764ba2] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 bg-[#8b5cf6] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
      <span className="text-sm text-gray-400 ml-2">Analyzing document</span>
    </div>
  )
}

// Session Info Component
function SessionInfo({ sessionId, fileCount, persona, onClear }: {
  sessionId: string | null;
  fileCount: number;
  persona: string;
  onClear: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copySessionId = () => {
    if (sessionId) {
      navigator.clipboard.writeText(sessionId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!sessionId) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 bg-[#1a1a24]/60 rounded-lg border border-[#ffffff06]">
      <div className="flex items-center gap-1.5">
        <div className="w-2 h-2 bg-[#10b981] rounded-full animate-pulse" />
        <span className="text-xs text-gray-400">Active Session</span>
      </div>
      <div className="w-px h-4 bg-[#ffffff10]" />
      <div className="flex items-center gap-1.5">
        <FileText className="w-3 h-3 text-[#667eea]" />
        <span className="text-xs text-gray-300">{fileCount} file{fileCount !== 1 ? 's' : ''}</span>
      </div>
      <div className="w-px h-4 bg-[#ffffff10]" />
      <div className="flex items-center gap-1.5">
        <Sparkles className="w-3 h-3 text-[#764ba2]" />
        <span className="text-xs text-gray-300">{persona}</span>
      </div>
      <button
        onClick={copySessionId}
        className="ml-1 p-1 hover:bg-[#ffffff08] rounded transition-colors"
        title="Copy session ID"
      >
        {copied ? <Check className="w-3 h-3 text-[#10b981]" /> : <Hash className="w-3 h-3 text-gray-500" />}
      </button>
      <button
        onClick={onClear}
        className="p-1 hover:bg-red-500/10 rounded transition-colors"
        title="Clear session"
      >
        <XCircle className="w-3 h-3 text-gray-500 hover:text-red-400" />
      </button>
    </div>
  )
}

// Chart Display Component (DOCBOT-303 / DOCBOT-305)
function ChartDisplay({ charts, chartMetas }: { charts: string[]; chartMetas?: (ChartMeta | null)[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!charts || charts.length === 0) return null;

  return (
    <div className="mt-3 space-y-3">
      {charts.map((b64, i) => {
        const meta = chartMetas?.[i];
        return (
          <div key={i} className="space-y-1">
            <div className="relative group">
              <img
                src={`data:image/png;base64,${b64}`}
                alt={meta?.title || `Analysis chart ${i + 1}`}
                className="rounded-lg border border-[#ffffff10] max-w-full cursor-pointer hover:opacity-90 transition-opacity"
                onClick={() => setExpanded(i)}
              />
              {/* Hover action bar */}
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                <button
                  onClick={() => setExpanded(i)}
                  className="bg-[#12121a]/90 backdrop-blur-sm rounded-md p-1.5 text-gray-400 hover:text-gray-200 border border-[#ffffff10]"
                  title="Expand"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
                <a
                  href={`data:image/png;base64,${b64}`}
                  download={meta?.title ? `${meta.title.replace(/\s+/g, "_")}.png` : `chart-${i + 1}.png`}
                  className="bg-[#12121a]/90 backdrop-blur-sm rounded-md px-2 py-1.5 text-xs text-gray-400 hover:text-gray-200 border border-[#ffffff10] flex items-center gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Download className="w-3 h-3" />
                  PNG
                </a>
              </div>
            </div>
            {/* DOCBOT-305: metadata caption */}
            {meta && (meta.title || meta.x_label || meta.y_label) && (
              <div className="px-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-gray-500">
                {meta.title && <span className="font-medium text-gray-400">{meta.title}</span>}
                {meta.x_label && <span>x: {meta.x_label}</span>}
                {meta.y_label && <span>y: {meta.y_label}</span>}
                {meta.series_count > 1 && <span>{meta.series_count} series</span>}
              </div>
            )}
          </div>
        );
      })}

      {/* Fullscreen zoom modal */}
      {expanded !== null && (
        <div
          className="fixed inset-0 bg-black/85 z-50 flex flex-col items-center justify-center p-4 gap-3"
          onClick={() => setExpanded(null)}
        >
          <img
            src={`data:image/png;base64,${charts[expanded]}`}
            alt="Chart full view"
            className="max-w-full max-h-[85vh] rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
          <div className="flex items-center gap-3">
            <a
              href={`data:image/png;base64,${charts[expanded]}`}
              download={chartMetas?.[expanded]?.title
                ? `${chartMetas[expanded]!.title.replace(/\s+/g, "_")}.png`
                : `chart-${expanded + 1}.png`}
              className="bg-[#1a1a24] border border-[#ffffff15] rounded-lg px-3 py-1.5 text-xs text-gray-300 hover:text-white flex items-center gap-1.5 transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              <Download className="w-3 h-3" /> Download PNG
            </a>
            <button
              onClick={() => setExpanded(null)}
              className="bg-[#1a1a24] border border-[#ffffff15] rounded-lg px-3 py-1.5 text-xs text-gray-300 hover:text-white flex items-center gap-1.5 transition-colors"
            >
              <Minimize2 className="w-3 h-3" /> Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Discrepancy Block Component (DOCBOT-403)
function DiscrepancyBlock({ content }: { content: string }) {
  return (
    <div className="mt-3 p-3 rounded-xl border border-[#f59e0b]/40 bg-[#f59e0b]/8">
      <div className="flex items-center gap-2 mb-2">
        <AlertTriangle className="w-4 h-4 text-[#f59e0b] shrink-0" />
        <span className="text-xs font-semibold text-[#f59e0b] uppercase tracking-wider">Discrepancy Detected</span>
      </div>
      <div className="text-sm text-gray-300 whitespace-pre-line leading-relaxed font-mono text-xs">
        {content.trim()}
      </div>
    </div>
  );
}

// DOCBOT-802: Client-side keyword router — returns the best-matching persona and confidence
function routeQuestion(
  question: string,
  chatMode: string,
  isDbConnected: boolean,
  hasDocSession: boolean,
  personas: typeof EXPERT_PERSONAS
): { persona: string; confidence: "high" | "medium" | "low" } {
  const q = question.toLowerCase();

  let bestPersona = "Generalist";
  let bestScore = 0;
  let secondScore = 0;

  for (const [name, config] of Object.entries(personas)) {
    if (name === "Generalist") continue;

    let score = 0;
    for (const kw of config.detection_keywords.primary) {
      const regex = new RegExp(`\\b${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      if (regex.test(q)) score += 3;
    }
    for (const kw of config.detection_keywords.secondary) {
      const regex = new RegExp(`\\b${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      if (regex.test(q)) score += 1;
    }

    if (score > bestScore) {
      secondScore = bestScore;
      bestScore = score;
      bestPersona = name;
    } else if (score > secondScore) {
      secondScore = score;
    }
  }

  if (bestScore === 0) return { persona: "Generalist", confidence: "low" };

  let confidence: "high" | "medium" | "low";
  if (bestScore >= 6 && bestScore - secondScore > 1) {
    confidence = "high";
  } else if (bestScore >= 3) {
    confidence = "medium";
  } else {
    confidence = "low";
  }

  // Context tie-break: when top two are very close and DB is connected, prefer Data Analyst
  if (bestScore - secondScore <= 1 && isDbConnected && bestPersona !== "Data Analyst") {
    // intentional no-op: keep bestPersona as-is, let tool_preference handle mode biasing
  }

  // Suppress unused-variable warnings for params used only for future tie-breaking
  void chatMode;
  void hasDocSession;

  return { persona: bestPersona, confidence };
}

// DOCBOT-805: Pre-process message content based on agent persona conventions
function applyAgentFormatting(content: string, agentPersona: string | undefined): string {
  if (!agentPersona) return content;
  const conf = EXPERT_PERSONAS[agentPersona as keyof typeof EXPERT_PERSONAS];
  if (!conf) return content;

  let processed = content;

  // Apply highlight_pattern: wrap matches in bold
  if (conf.output_conventions.highlight_pattern) {
    try {
      const re = new RegExp(conf.output_conventions.highlight_pattern, "g");
      processed = processed.replace(re, "**$&**");
    } catch {
      // invalid regex — skip
    }
  }

  // Clinical: style Medical Disclaimer block
  if (conf.response_format === "clinical") {
    processed = processed.replace(
      /## Medical Disclaimer\n/g,
      "\n---\n> **Medical Disclaimer**\n>\n> "
    );
  }

  return processed;
}

// Parse message content into text segments and discrepancy blocks (DOCBOT-403)
function renderMessageContent(content: string): React.ReactNode[] | null {
  const parts = content.split(/\[DISCREPANCY\]([\s\S]*?)\[\/DISCREPANCY\]/g);
  if (parts.length === 1) return null; // no discrepancy blocks — caller uses normal renderer
  return parts.map((part, i) =>
    i % 2 === 0
      ? part.trim()
        ? <ReactMarkdown key={i} remarkPlugins={[remarkGfm]}>{part}</ReactMarkdown>
        : null
      : <DiscrepancyBlock key={i} content={part} />
  ).filter(Boolean);
}

// DOCBOT-805: Agent-aware message content renderer
function AgentMessageContent({ msg }: { msg: Message }) {
  const agentConf = msg.agentPersona ? EXPERT_PERSONAS[msg.agentPersona as keyof typeof EXPERT_PERSONAS] : undefined;
  const responseFormat = agentConf?.response_format;
  const processedContent = applyAgentFormatting(msg.content, msg.agentPersona);
  const wrapperClass = [
    "prose prose-invert prose-sm max-w-none",
    "prose-p:leading-7 prose-p:text-gray-300",
    "prose-headings:text-white prose-headings:font-semibold prose-headings:mt-0 prose-headings:mb-2",
    "prose-strong:text-white prose-strong:font-medium",
    "prose-ul:text-gray-300 prose-ul:my-2 prose-li:my-1",
    "prose-ol:text-gray-300 prose-ol:my-2 prose-li:my-1",
    "prose-li:marker:text-[#667eea]",
    "prose-a:text-[#667eea] prose-a:no-underline hover:prose-a:underline",
    "prose-blockquote:border-l-[#667eea] prose-blockquote:bg-[#1a1a24]/50 prose-blockquote:py-1 prose-blockquote:px-3 prose-blockquote:rounded-r-lg prose-blockquote:text-gray-400 prose-blockquote:not-italic",
    "prose-code:text-[#764ba2] prose-code:bg-[#1a1a24]/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none",
    "prose-pre:bg-[#1a1a24]/80 prose-pre:border prose-pre:border-[#ffffff08]",
    responseFormat === "finance" ? "agent-finance" : "",
    responseFormat === "legal" ? "agent-legal" : "",
    responseFormat === "clinical" ? "agent-clinical" : "",
  ].filter(Boolean).join(" ");
  const financeStyle: React.CSSProperties | undefined = responseFormat === "finance"
    ? { borderLeft: "3px solid rgba(245,158,11,0.3)", paddingLeft: "0.75rem" }
    : undefined;
  return (
    <div className={wrapperClass} style={financeStyle}>
      {renderMessageContent(processedContent) ?? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {processedContent}
        </ReactMarkdown>
      )}
    </div>
  );
}

// Collapsible Code Component (DOCBOT-303)
function CollapsibleCode({ code }: { code: string }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-2 border border-[#ffffff10] rounded-lg overflow-hidden text-sm">
      <button
        className="w-full flex items-center justify-between px-3 py-2 bg-[#1a1a24]/80 hover:bg-[#1a1a24] text-gray-500 hover:text-gray-300 text-xs font-mono transition-colors"
        onClick={() => setOpen(o => !o)}
      >
        <span className="flex items-center gap-1.5">
          <Terminal className="w-3 h-3" />
          Analysis code
        </span>
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <pre className="p-3 overflow-x-auto bg-[#0a0a0f] text-[#10b981] text-xs leading-relaxed">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  // Sidebar State
  const [selectedPersona, setSelectedPersona] = useState("Generalist");
  const [suggestedPersona, setSuggestedPersona] = useState<string | null>(null);
  const [deepVisualMode, setDeepVisualMode] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [drProgress, setDrProgress] = useState<{ step: string; message: string } | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // DOCBOT-802: Auto-routing mode — when true, routeQuestion() picks the persona per message
  const [isAutoMode, setIsAutoMode] = useState(true);

  // Database connection state
  const [isDbConnected, setIsDbConnected] = useState(false);
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [dbUploadState, setDbUploadState] = useState<"idle" | "uploading" | "connected" | "error">("idle");
  const [dbFileName, setDbFileName] = useState<string | null>(null);

  // Live DB connection form state
  const [showLiveDbForm, setShowLiveDbForm] = useState(false);
  const [liveDbForm, setLiveDbForm] = useState({
    dialect: "postgresql",
    host: "",
    port: "5432",
    dbname: "",
    user: "",
    password: "",
    pii_masking_enabled: false,
  });
  const [showDbPassword, setShowDbPassword] = useState(false);
  const [liveDbConnectState, setLiveDbConnectState] = useState<"idle" | "connecting" | "error">("idle");

  // Entra interactive auth state
  const [entraToken, setEntraToken] = useState<string | null>(null);
  const [entraEmail, setEntraEmail] = useState<string | null>(null);
  const [entraSignInState, setEntraSignInState] = useState<"idle" | "signing_in" | "signed_in" | "error">("idle");
  const [liveDbError, setLiveDbError] = useState<string | null>(null);

  // DOCBOT-504: Query history panel
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null);

  // DOCBOT-305: Advanced charts
  const [chartType, setChartType] = useState<string>("auto");
  const [zoomedChart, setZoomedChart] = useState<string | null>(null);

  // DOCBOT-405: Analytical Autopilot
  const [autopilotMode, setAutopilotMode] = useState(false);
  const [autopilotRunning, setAutopilotRunning] = useState(false);
  const [autopilotSteps, setAutopilotSteps] = useState<AutopilotStep[]>([]);
  const [autopilotPlan, setAutopilotPlan] = useState<string[]>([]);

  // Auto-detect: suggest Autopilot when question contains multi-step keywords
  const AUTOPILOT_KEYWORDS = /\b(why|diagnos|investigat|analys|analy[sz]|compar|trend|forecast|explain|root.?cause|deep.?dive|break.?down|summariz|evaluat)\b/i;
  const showAutopilotNudge = isDbConnected && !autopilotMode && AUTOPILOT_KEYWORDS.test(input);

  // Chat mode: "docs" → /api/chat, "database" → /api/db/chat, "hybrid" → /api/hybrid/chat
  const [chatMode, setChatMode] = useState<"docs" | "database" | "hybrid">("docs");

  // ── DOCBOT-606 / DOCBOT-701: Auth state ──────────────────────────────────
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [samlConfigured, setSamlConfigured] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [authConfig, setAuthConfig] = useState<{ email: boolean; github: boolean; google: boolean; saml: boolean } | null>(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalTab, setAuthModalTab] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authName, setAuthName] = useState("");
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  // ── Persistent Workspace state ────────────────────────────────────────────
  const [workspaceConnections, setWorkspaceConnections] = useState<WorkspaceConnection[]>([]);

  // ── DOCBOT-606: Admin panel state ─────────────────────────────────────────
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const [adminTab, setAdminTab] = useState<"users" | "audit">("users");
  const [adminUsers, setAdminUsers] = useState<AdminUser[]>([]);
  const [adminUsersLoading, setAdminUsersLoading] = useState(false);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditEventTypeFilter, setAuditEventTypeFilter] = useState<string>("all");

  // Stable anonymous session ID used when no PDF session exists yet
  const anonymousSessionIdRef = useRef<string>(
    typeof crypto !== "undefined" ? crypto.randomUUID() : Math.random().toString(36).substring(2)
  );

  // DOCBOT-504: Load query history for a connection
  const loadQueryHistory = useCallback(async (connId: string) => {
    try {
      const res = await fetch(`/api/db/history/${connId}?limit=20`);
      if (!res.ok) return;
      const data = await res.json();
      setQueryHistory(data.history ?? []);
    } catch {
      // non-fatal — history panel just stays empty
    }
  }, []);

  // Called when a DB connection is successfully established.
  // Auto-selects the Data Analyst persona only when the user has not already
  // chosen a specific persona (i.e. still on the default Generalist).
  // If both a PDF (sessionId) and DB are connected, default to "hybrid" mode.
  const handleDbConnected = useCallback((connId: string) => {
    setIsDbConnected(true);
    setConnectionId(connId);
    // If both docs and DB are present, use hybrid mode so backend intent classifier handles routing
    setChatMode(sessionId ? "hybrid" : "database");
    setSelectedPersona(prev => (prev === "Generalist" ? "Data Analyst" : prev));
    loadQueryHistory(connId);
  }, [loadQueryHistory, sessionId]);

  const handleDbDisconnect = useCallback(() => {
    setIsDbConnected(false);
    setConnectionId(null);
    setDbFileName(null);
    setDbUploadState("idle");
    setChatMode("docs");
    setSelectedPersona(prev => (prev === "Data Analyst" ? "Generalist" : prev));
    setShowLiveDbForm(false);
    setLiveDbConnectState("idle");
    setLiveDbError(null);
    setQueryHistory([]);
    setHistoryOpen(false);
    setAutopilotMode(false);
    setAutopilotSteps([]);
    setAutopilotPlan([]);
  }, []);

  const handleDbUpload = async (file: File, type: "csv" | "sqlite") => {
    setDbUploadState("uploading");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId ?? anonymousSessionIdRef.current);
    try {
      const endpoint = type === "csv" ? "/api/db/upload/csv" : "/api/db/upload";
      const response = await fetch(endpoint, { method: "POST", body: formData });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await response.json();
      setDbFileName(file.name);
      setDbUploadState("connected");
      handleDbConnected(data.connection_id);
      showToast("success", `Connected: ${file.name}`);
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Upload failed";
      setDbUploadState("error");
      showToast("error", `DB upload failed: ${msg}`);
      setTimeout(() => setDbUploadState("idle"), 3000);
    }
  };

  const handleLiveDbConnect = async () => {
    setLiveDbConnectState("connecting");
    setLiveDbError(null);
    try {
      const isEntra = liveDbForm.dialect === "azure_sql";
      const body = isEntra
        ? {
            session_id: sessionId ?? anonymousSessionIdRef.current,
            dialect: "azure_sql",
            host: liveDbForm.host,
            port: parseInt(liveDbForm.port, 10),
            dbname: liveDbForm.dbname,
            auth_type: "entra_interactive",
            access_token: entraToken,
            pii_masking_enabled: liveDbForm.pii_masking_enabled,
          }
        : {
            session_id: sessionId ?? anonymousSessionIdRef.current,
            dialect: liveDbForm.dialect,
            host: liveDbForm.host,
            port: parseInt(liveDbForm.port, 10),
            dbname: liveDbForm.dbname,
            user: liveDbForm.user,
            password: liveDbForm.password,
            pii_masking_enabled: liveDbForm.pii_masking_enabled,
          };
      const response = await fetch("/api/db/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const err = await response.json();
        const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail) || "Connection failed";
        throw new Error(detail);
      }
      const data = await response.json();
      setDbFileName(`${liveDbForm.dialect}://${liveDbForm.host}/${liveDbForm.dbname}`);
      setDbUploadState("connected");
      setShowLiveDbForm(false);
      setLiveDbConnectState("idle");
      handleDbConnected(data.connection_id);
      showToast("success", `Connected to ${liveDbForm.dbname}`);
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Connection failed";
      setLiveDbConnectState("error");
      setLiveDbError(msg);
    }
  };

  const msalConfig = {
    auth: {
      clientId: process.env.NEXT_PUBLIC_AZURE_CLIENT_ID ?? "",
      authority: "https://login.microsoftonline.com/common",
      redirectUri: typeof window !== "undefined" ? window.location.origin : "",
    },
  };

  async function handleMicrosoftSignIn() {
    setEntraSignInState("signing_in");
    try {
      const { PublicClientApplication } = await import("@azure/msal-browser");
      const msalInstance = new PublicClientApplication(msalConfig);
      await msalInstance.initialize();
      const result = await msalInstance.acquireTokenPopup({
        scopes: ["https://database.windows.net/.default"],
      });
      setEntraToken(result.accessToken);
      setEntraEmail(result.account?.username ?? "Microsoft Account");
      setEntraSignInState("signed_in");
    } catch (err) {
      console.error("MSAL sign-in error:", err);
      setEntraSignInState("error");
    }
  }

  // File Upload State
  const [fileUploadState, setFileUploadState] = useState<FileUploadState>("idle");

  // Toast State
  const [toasts, setToasts] = useState<Toast[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const lastMessageRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Toast helper
  const showToast = useCallback((type: Toast['type'], message: string) => {
    const id = Math.random().toString(36).substring(7);
    setToasts(prev => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 5000);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // ── Persistent Workspace: load previous sessions + connections ───────────
  const fetchWorkspace = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/workspace", { credentials: "include" });
      if (!res.ok) return;
      const data = await res.json();
      const parsed = WorkspaceSchema.safeParse(data);
      if (parsed.success) {
        setWorkspaceConnections(parsed.data.db_connections);
      }
    } catch {
      // non-fatal — workspace panel stays empty
    }
  }, []);

  // ── DOCBOT-606: Check auth on mount ──────────────────────────────────────
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Fetch auth config (which providers are enabled)
        const configRes = await fetch("/api/auth/config");
        if (configRes.ok) {
          const cfg = await configRes.json();
          setAuthConfig(cfg);
          setSamlConfigured(cfg.saml === true);
        }

        // Check if already authenticated
        const res = await fetch("/api/auth/me", { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          const parsed = AuthMeSchema.safeParse(data);
          if (parsed.success) {
            setAuthUser(parsed.data);
            // Load workspace for already-authenticated user
            fetchWorkspace();
          }
        }

        // Handle OAuth redirect result (GitHub/Google callback redirects back here)
        if (typeof window !== "undefined") {
          const params = new URLSearchParams(window.location.search);
          if (params.get("auth_success")) {
            // Re-fetch user after OAuth redirect
            const meRes = await fetch("/api/auth/me", { credentials: "include" });
            if (meRes.ok) {
              const data = await meRes.json();
              const parsed = AuthMeSchema.safeParse(data);
              if (parsed.success) {
                setAuthUser(parsed.data);
                showToast("success", `Welcome, ${parsed.data.name || parsed.data.email}!`);
                // Load workspace after OAuth login
                fetchWorkspace();
              }
            }
            // Clean up query param
            window.history.replaceState({}, "", window.location.pathname);
          } else if (params.get("auth_error")) {
            const errorMap: Record<string, string> = {
              invalid_state: "Authentication session expired. Please try again.",
              github_failed: "GitHub sign-in failed. Please try again.",
              google_failed: "Google sign-in failed. Please try again.",
              provision_failed: "Account setup failed. Please try again.",
            };
            const msg = errorMap[params.get("auth_error")!] ?? "Sign-in failed. Please try again.";
            showToast("error", msg);
            window.history.replaceState({}, "", window.location.pathname);
          }
        }
      } catch {
        // network error — stay in open mode
      } finally {
        setAuthChecked(true);
      }
    };
    checkAuth();
  }, [fetchWorkspace]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── DOCBOT-606 / DOCBOT-701: Handle logout ───────────────────────────────
  const handleLogout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
    } catch {
      // ignore
    }
    setAuthUser(null);
    setWorkspaceConnections([]);
    showToast("info", "Signed out");
  }, [showToast]);

  // ── DOCBOT-701: Email/password auth ───────────────────────────────────────
  const handleEmailAuth = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthSubmitting(true);
    setAuthError(null);
    try {
      const endpoint = authModalTab === "register" ? "/api/auth/register" : "/api/auth/login";
      const body: Record<string, string> = { email: authEmail, password: authPassword };
      if (authModalTab === "register" && authName) body.name = authName;

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        setAuthError(data.detail ?? "Something went wrong.");
        return;
      }
      // Fetch user profile
      const meRes = await fetch("/api/auth/me", { credentials: "include" });
      if (meRes.ok) {
        const me = await meRes.json();
        const parsed = AuthMeSchema.safeParse(me);
        if (parsed.success) setAuthUser(parsed.data);
      }
      setAuthModalOpen(false);
      setAuthEmail("");
      setAuthPassword("");
      setAuthName("");
      showToast("success", authModalTab === "register" ? "Account created!" : "Welcome back!");
      // Load workspace after email/password login
      fetchWorkspace();
    } catch {
      setAuthError("Network error. Please try again.");
    } finally {
      setAuthSubmitting(false);
    }
  }, [authModalTab, authEmail, authPassword, authName, showToast, fetchWorkspace]);

  // ── DOCBOT-606: Load admin users ──────────────────────────────────────────
  const loadAdminUsers = useCallback(async () => {
    setAdminUsersLoading(true);
    try {
      const res = await fetch("/admin/users", { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const parsed = AdminUsersResponseSchema.safeParse(data);
      if (parsed.success) setAdminUsers(parsed.data.users);
    } catch (err) {
      showToast("error", `Failed to load users: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAdminUsersLoading(false);
    }
  }, [showToast]);

  // ── DOCBOT-606: Load audit log ────────────────────────────────────────────
  const loadAuditLog = useCallback(async (eventType?: string) => {
    setAuditLoading(true);
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (eventType && eventType !== "all") params.set("event_type", eventType);
      const res = await fetch(`/admin/audit-log?${params.toString()}`, { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const parsed = AuditLogResponseSchema.safeParse(data);
      if (parsed.success) setAuditEvents(parsed.data.events);
    } catch (err) {
      showToast("error", `Failed to load audit log: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAuditLoading(false);
    }
  }, [showToast]);

  // ── DOCBOT-606: Update user role ──────────────────────────────────────────
  const updateUserRole = useCallback(async (userId: string, role: string) => {
    try {
      const res = await fetch(`/admin/users/${userId}/role`, {
        method: "PATCH",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setAdminUsers(prev =>
        prev.map(u => u.id === userId ? { ...u, role: role as AdminUser["role"] } : u)
      );
      showToast("success", "Role updated");
    } catch (err) {
      showToast("error", `Role update failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [showToast]);

  // ── DOCBOT-606: Export audit log as CSV ───────────────────────────────────
  const exportAuditLogCsv = useCallback(async () => {
    try {
      const params = new URLSearchParams({ format: "csv", limit: "5000" });
      if (auditEventTypeFilter !== "all") params.set("event_type", auditEventTypeFilter);
      const res = await fetch(`/admin/audit-log?${params.toString()}`, { credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "audit_log.csv";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      showToast("success", "Audit log exported");
    } catch (err) {
      showToast("error", `Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [auditEventTypeFilter, showToast]);

  // When admin panel opens, load data for the active tab
  useEffect(() => {
    if (!adminPanelOpen) return;
    if (adminTab === "users") loadAdminUsers();
    else loadAuditLog(auditEventTypeFilter);
  }, [adminPanelOpen, adminTab]); // eslint-disable-line react-hooks/exhaustive-deps

  const scrollToBottom = () => {
    if (lastMessageRef.current && chatContainerRef.current) {
      const container = chatContainerRef.current;
      const msgRect = lastMessageRef.current.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const scrollOffset = msgRect.top - containerRect.top + container.scrollTop - 16;
      container.scrollTo({ top: scrollOffset, behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [input]);

  // Fall back to "docs" mode when the active mode's required source becomes unavailable
  useEffect(() => {
    if (chatMode === "database" && !connectionId) {
      setChatMode("docs");
    } else if (chatMode === "hybrid" && (!connectionId || !sessionId)) {
      setChatMode("docs");
    }
  }, [connectionId, sessionId, chatMode]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + Enter to send
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (input.trim() && (sessionId || isDbConnected) && !isLoading) {
          handleSendMessage();
        }
      }
      // Escape to clear input
      if (e.key === 'Escape') {
        setInput("");
        textareaRef.current?.blur();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [input, sessionId, isDbConnected, isLoading]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const files = Array.from(e.target.files);

    const MAX_FILE_SIZE = 4.5 * 1024 * 1024; // 4.5 MB max for Vercel
    const totalSize = files.reduce((acc, file) => acc + file.size, 0);

    if (totalSize > MAX_FILE_SIZE) {
      showToast("error", "Total file size exceeds 4.5MB limit. Please upload smaller documents.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setUploadedFiles(files);
    setIsLoading(true);
    setUploadProgress(0);
    setFileUploadState("uploading");

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('deep_visual_mode', String(deepVisualMode));

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min((prev || 0) + 10, 90));
      }, 200);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to upload");
      }

      const data = await response.json();
      setSessionId(data.session_id);

      if (data.suggested_persona && data.suggested_persona !== "Generalist") {
        setSuggestedPersona(data.suggested_persona);
        setSelectedPersona(data.suggested_persona);
        showToast("info", `Switched to ${data.suggested_persona} mode for your document`);
      } else {
        showToast("success", `Successfully processed ${files.length} document${files.length > 1 ? 's' : ''}`);
      }

      setFileUploadState("success");
      setTimeout(() => setFileUploadState("idle"), 2000);

    } catch (error: any) {
      console.error("Upload error:", error);
      setFileUploadState("error");
      showToast("error", `Upload failed: ${error.message}`);
      setUploadedFiles([]);
      setTimeout(() => setFileUploadState("idle"), 3000);
    } finally {
      setIsLoading(false);
      setUploadProgress(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setFileUploadState("dragover");
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setFileUploadState("idle");
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const files = Array.from(e.dataTransfer.files).filter(f => f.type === "application/pdf");

    if (files.length > 0) {
      const dataTransfer = new DataTransfer();
      files.forEach(f => dataTransfer.items.add(f));
      if (fileInputRef.current) {
        fileInputRef.current.files = dataTransfer.files;
        const event = { target: { files: dataTransfer.files } } as any;
        handleFileUpload(event);
      }
    } else {
      showToast("warning", "Please drop PDF files only");
      setFileUploadState("idle");
    }
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim()) return;
    if (!sessionId && !isDbConnected) return;

    const userMsg: Message = { role: "user", content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setDrProgress(null);

    // ── DOCBOT-802: Auto-routing — pick persona and optionally bias chat mode ──
    let personaToSend = selectedPersona;
    // Use local variable for routing — React setState is async, so setChatMode won't be available in same execution
    let effectiveChatMode = chatMode;
    // If both docs (sessionId) and DB (isDbConnected) are present, default to hybrid so backend intent classifier routes correctly
    if (sessionId && isDbConnected && chatMode !== "hybrid") {
      effectiveChatMode = "hybrid";
    }
    if (isAutoMode) {
      const routing = routeQuestion(input, effectiveChatMode, isDbConnected, !!sessionId, EXPERT_PERSONAS);
      if (routing.confidence !== "low") {
        personaToSend = routing.persona;
        // Only bias chat mode when a SINGLE source is available.
        // When both PDF and CSV/DB are present, keep hybrid so backend intent classifier decides.
        const pref = EXPERT_PERSONAS[routing.persona as keyof typeof EXPERT_PERSONAS]?.tool_preference;
        if (pref === "sql_first" && isDbConnected && !sessionId) effectiveChatMode = "database";
        else if (pref === "rag_first" && sessionId && !isDbConnected) effectiveChatMode = "docs";
      } else {
        // Low confidence: fall back to Generalist rather than the upload-recommended persona
        personaToSend = "Generalist";
      }
    }
    // Update chatMode state after routing decisions (for UI, not for this request)
    if (effectiveChatMode !== chatMode) {
      setChatMode(effectiveChatMode);
    }

    // ── DOCBOT-405: Autopilot path ─────────────────────────────────────────
    if (effectiveChatMode === "database" && connectionId && autopilotMode) {
      setAutopilotRunning(true);
      setAutopilotSteps([]);
      setAutopilotPlan([]);
      // Local accumulator — avoids stale closure when reading state inside SSE handler
      const localSteps: AutopilotStep[] = [];
      try {
        const response = await fetch("/api/autopilot/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            connection_id: connectionId,
            question: userMsg.content,
            persona: personaToSend,
            session_id: sessionId ?? "anonymous",
          }),
        });
        if (!response.ok || !response.body) {
          throw new Error(`Autopilot HTTP ${response.status}`);
        }

        const assistantMsg: Message = {
          role: "assistant",
          content: "",
          timestamp: new Date(),
          charts: [],
          agentPersona: personaToSend,
        };
        setMessages(prev => [...prev, assistantMsg]);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "plan") {
                setAutopilotPlan(data.steps ?? []);

              } else if (data.type === "step") {
                const stepEntry: AutopilotStep = {
                  step_num: data.step_num,
                  tool: data.tool,
                  step_label: data.step_label,
                  content: data.content,
                  artifact_id: data.artifact_id ?? null,
                  chart_b64: data.chart_b64 ?? null,
                  error: data.error ?? null,
                };
                localSteps.push(stepEntry);
                setAutopilotSteps(prev => [...prev, stepEntry]);

              } else if (data.type === "answer") {
                setMessages(prev => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last?.role === "assistant") {
                    updated[updated.length - 1] = { ...last, content: data.content };
                  }
                  return updated;
                });

              } else if (data.type === "done") {
                // Attach completed steps (with charts) to the message so they persist after run
                if (localSteps.length > 0) {
                  setMessages(prev => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    if (last?.role === "assistant") {
                      updated[updated.length - 1] = { ...last, autopilotSteps: [...localSteps] };
                    }
                    return updated;
                  });
                }

              } else if (data.type === "warning" || data.type === "error") {
                setMessages(prev => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last?.role === "assistant") {
                    const prefix = last.content ? last.content + "\n\n" : "";
                    updated[updated.length - 1] = {
                      ...last,
                      content: prefix + `⚠️ ${data.content}`,
                    };
                  }
                  return updated;
                });
              }
            } catch {
              // malformed JSON — skip
            }
          }
        }
      } catch (err) {
        setMessages(prev => [
          ...prev,
          { role: "assistant", content: `Autopilot error: ${err instanceof Error ? err.message : String(err)}`, timestamp: new Date() },
        ]);
      } finally {
        setIsLoading(false);
        setAutopilotRunning(false);
      }
      return;
    }

    // ── DB chat path: SSE streaming via /api/db/chat ──────────────────────
    if (effectiveChatMode === "database" && connectionId) {
      try {
        const assistantMsg: Message = {
          role: "assistant",
          content: "",
          timestamp: new Date(),
          charts: [],
          agentPersona: personaToSend,
        };
        setMessages(prev => [...prev, assistantMsg]);

        const response = await fetch("/api/db/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            connection_id: connectionId,
            question: userMsg.content,
            persona: personaToSend,
            session_id: sessionId ?? "anonymous",
            chart_type: chartType,
          }),
        });

        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            let chunk: Record<string, unknown>;
            try {
              chunk = JSON.parse(jsonStr);
            } catch {
              continue; // skip malformed SSE lines
            }

            if (chunk.type === "token") {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1 ? { ...m, content: m.content + (chunk.content as string) } : m
              ));
            } else if (chunk.type === "metadata") {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1
                  ? { ...m, sql: chunk.sql_query as string | undefined, explanation: chunk.explanation as string | undefined }
                  : m
              ));
            } else if (chunk.type === "analysis_code") {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1 ? { ...m, analysisCode: chunk.code as string | undefined } : m
              ));
            } else if (chunk.type === "chart") {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1
                  ? {
                      ...m,
                      charts: [...(m.charts ?? []), chunk.base64 as string],
                      chartMetas: chunk.metadata
                        ? [...(m.chartMetas ?? []), chunk.metadata as ChartMeta]
                        : (m.chartMetas ?? []),
                    }
                  : m
              ));
            } else if (chunk.type === "error") {
              throw new Error((chunk.detail as string) || "Database query failed");
            }
          }
        }
      } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : "Unknown error";
        console.error("DB chat error:", error);
        const isFileGone = msg.includes("re-upload") || msg.includes("temporary files");
        showToast("error", isFileGone ? msg : `Query error: ${msg}`);
        setMessages(prev => prev.map((m, i) =>
          i === prev.length - 1 && m.role === "assistant" && m.content === ""
            ? { ...m, content: isFileGone
                ? "The uploaded file is no longer available — the server was restarted. Please re-upload your CSV or SQLite file."
                : "I encountered an error querying your database. Please try again." }
            : m
        ));
      } finally {
        setIsLoading(false);
        // DOCBOT-504: refresh history after each query
        if (connectionId) loadQueryHistory(connectionId);
      }
      return;
    }

    // ── Hybrid chat path: SSE streaming via /api/hybrid/chat ─────────────
    if (effectiveChatMode === "hybrid" && connectionId && sessionId) {
      try {
        const assistantMsg: Message = { role: "assistant", content: "", timestamp: new Date(), charts: [], agentPersona: personaToSend };
        setMessages(prev => [...prev, assistantMsg]);

        const response = await fetch("/api/hybrid/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: userMsg.content,
            session_id: sessionId,
            connection_id: connectionId,
            persona: personaToSend,
            has_docs: true,
            deep_research: deepResearch,
          }),
        });

        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;
            try {
              const chunk = JSON.parse(jsonStr);
              if (chunk.type === "token") {
                setMessages(prev => prev.map((m, i) =>
                  i === prev.length - 1 ? { ...m, content: m.content + chunk.content } : m
                ));
              } else if (chunk.type === "metadata") {
                setMessages(prev => prev.map((m, i) =>
                  i === prev.length - 1 ? { ...m, sql: chunk.sql_query, explanation: chunk.explanation } : m
                ));
              } else if (chunk.type === "analysis_code") {
                setMessages(prev => prev.map((m, i) =>
                  i === prev.length - 1 ? { ...m, analysisCode: chunk.code } : m
                ));
              } else if (chunk.type === "chart") {
                setMessages(prev => prev.map((m, i) =>
                  i === prev.length - 1 ? { ...m, charts: [...(m.charts ?? []), chunk.base64] } : m
                ));
              } else if (chunk.type === "error") {
                throw new Error(chunk.detail || "Hybrid query failed");
              }
            } catch { /* skip malformed SSE lines */ }
          }
        }
      } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : "Unknown error";
        showToast("error", `Error: ${msg}`);
        setMessages(prev => prev.map((m, i) =>
          i === prev.length - 1 && m.role === "assistant" && m.content === ""
            ? { ...m, content: "I encountered an error with the hybrid query. Please try again." }
            : m
        ));
      } finally {
        setIsLoading(false);
      }
      return;
    }

    // ── Document chat path: SSE streaming /api/chat ──────────────────────
    // Guard: this path requires an active PDF session. If only a DB is
    // connected (no PDF uploaded) we must not fall through here — the DB
    // path above would have already returned. If we somehow reach this
    // point without a session ID, bail out cleanly.
    if (!sessionId) {
      setIsLoading(false);
      showToast("error", "Please upload a PDF document before chatting.");
      return;
    }

    try {
      // Pre-push an empty assistant message so tokens are appended to the
      // correct bubble. Without this the stream tokens would be appended to
      // the last user message, causing raw markdown to appear inside the
      // user's bubble instead of a proper assistant response.
      const assistantMsg: Message = {
        role: "assistant",
        content: "",
        timestamp: new Date(),
        agentPersona: personaToSend,
      };
      setMessages(prev => [...prev, assistantMsg]);

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMsg.content,
          history: messages,
          persona: personaToSend,
          deep_research: deepResearch
        })
      });

      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;
          try {
            const chunk = JSON.parse(jsonStr);
            if (chunk.type === "progress") {
              setDrProgress({ step: chunk.step, message: chunk.message });
            } else if (chunk.type === "token") {
              setDrProgress(null); // collapse progress strip on first token
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1 ? { ...m, content: m.content + chunk.content } : m
              ));
            } else if (chunk.type === "citations") {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1 ? { ...m, citations: chunk.citations } : m
              ));
            } else if (chunk.type === "error") {
              throw new Error(chunk.detail || "Chat failed");
            }
          } catch { /* skip malformed SSE lines */ }
        }
      }
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Unknown error";
      console.error("Chat error:", error);
      showToast("error", `Error: ${msg}`);
      setMessages(prev => prev.map((m, i) =>
        i === prev.length - 1 && m.role === "assistant" && m.content === ""
          ? { ...m, content: "I encountered an error processing your request. Please try again." }
          : m
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    showToast("info", "Chat cleared");
  };

  const clearSession = () => {
    setSessionId(null);
    setUploadedFiles([]);
    setMessages([]);
    setSuggestedPersona(null);
    showToast("info", "Session cleared");
  };

  const exportChat = async (format: 'txt' | 'markdown' | 'json') => {
    if (!sessionId) {
      showToast("error", "No active session to export");
      return;
    }

    try {
      const response = await fetch(`/api/export/${sessionId}?format=${format}`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error("Export failed");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `docbot-chat.${format === 'markdown' ? 'md' : format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      showToast("success", `Chat exported as ${format.toUpperCase()}`);
    } catch (error) {
      console.error("Export error:", error);
      showToast("error", "Failed to export chat");
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    showToast("success", "Copied to clipboard");
  };

  const getFileUploadBorderColor = () => {
    switch (fileUploadState) {
      case "dragover": return "border-[#10b981] bg-[#10b981]/5";
      case "uploading": return "border-[#667eea] bg-[#667eea]/5";
      case "success": return "border-[#10b981] bg-[#10b981]/5";
      case "error": return "border-[#ef4444] bg-[#ef4444]/5";
      default: return "border-[#667eea]/30 bg-[#12121a]/50";
    }
  };

  return (
    <div className="flex h-screen relative z-10 text-[#e0e0e0] overflow-hidden bg-[#0a0a0f]">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* Mobile Menu Toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-[#12121a]/90 backdrop-blur-xl rounded-lg border border-[#ffffff08]"
      >
        {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Settings / Options Sidebar */}
      <aside className={`
        w-80 backdrop-blur-2xl bg-[#12121a]/95 border-r border-[#ffffff08] flex flex-col p-5 z-40 shrink-0 shadow-2xl overflow-y-auto
        transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        fixed lg:relative h-full
      `}>
        {/* Logo */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#667eea] to-[#764ba2] flex items-center justify-center shadow-lg shadow-[#667eea]/20">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-bold text-white tracking-wide flex items-center gap-2">
              DocBot
              <span className="px-1.5 py-0.5 bg-[#667eea]/20 text-[#667eea] text-[10px] font-bold rounded">AI</span>
            </h2>
            <p className="text-xs text-gray-500">Document Intelligence</p>
          </div>
        </div>

        {/* ── DOCBOT-701: Auth widget ────────────────────────────────────── */}
        {authChecked && (
          <div className="mb-5">
            {authUser ? (
              /* Authenticated — show user info + logout */
              <div className="flex items-center gap-2 px-3 py-2.5 bg-[#1a1a24]/70 rounded-xl border border-[#ffffff08]">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#667eea]/30 to-[#764ba2]/30 flex items-center justify-center shrink-0">
                  <ShieldCheck className="w-3.5 h-3.5 text-[#a5b4fc]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-200 truncate">{authUser.name || authUser.email}</p>
                  <p className="text-[10px] text-gray-500 truncate">{authUser.email}</p>
                </div>
                {authUser.role === "admin" && (
                  <button
                    onClick={() => setAdminPanelOpen(true)}
                    className="p-1.5 rounded-lg hover:bg-[#ffffff10] text-gray-500 hover:text-[#f59e0b] transition-colors"
                    title="Admin panel"
                  >
                    <Shield className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  onClick={handleLogout}
                  className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-500 hover:text-red-400 transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              /* Not authenticated — show sign-in button (always visible) */
              <button
                onClick={() => { setAuthModalOpen(true); setAuthError(null); }}
                className="flex items-center justify-center gap-2 w-full px-3 py-2.5 rounded-xl bg-[#667eea]/15 hover:bg-[#667eea]/25 border border-[#667eea]/30 text-[#a5b4fc] text-xs font-medium transition-all"
              >
                <ShieldCheck className="w-4 h-4" />
                Sign in / Create account
              </button>
            )}
          </div>
        )}

        {/* ── Persistent Workspace: Saved DB connections ──────────────────── */}
        {authUser && workspaceConnections.filter(wc => wc.host !== "__local_file__").length > 0 && (
          <div className="mb-5">
            <h3 className="text-xs font-semibold mb-2 text-gray-400 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" />
              Saved connections
            </h3>
            <ul className="space-y-1">
              {workspaceConnections.filter(wc => wc.host !== "__local_file__").map((wc) => (
                <li key={wc.id}>
                  <button
                    onClick={() => {
                      setShowLiveDbForm(true);
                      setLiveDbForm(prev => ({
                        ...prev,
                        dialect: wc.dialect,
                        host: wc.host,
                        dbname: wc.db_name,
                      }));
                      showToast("info", "Connection details loaded — enter credentials and connect");
                    }}
                    className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-[#ffffff08] transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Database className="w-3 h-3 text-[#10b981] shrink-0" />
                      <span className="text-[11px] text-gray-300 truncate flex-1">
                        {wc.dialect} · {wc.db_name}
                      </span>
                    </div>
                    <p className="text-[10px] text-gray-600 mt-0.5 pl-5 truncate">{wc.host}</p>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Document Upload Section */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-3 text-white flex items-center gap-2">
            <FileText className="w-4 h-4 text-[#667eea]" />
            Documents
            <span className="ml-auto text-[10px] text-gray-500 font-normal">PDF only</span>
          </h3>

          {/* Deep Visual Mode Toggle */}
          <div className="mb-4 bg-[#1a1a24]/50 rounded-xl p-3 border border-[#ffffff06]">
            <label className="flex items-center justify-between cursor-pointer group">
              <div className="flex items-center gap-3">
                <div className={`w-9 h-5 flex items-center bg-gray-700/50 rounded-full p-0.5 transition-colors ${deepVisualMode ? 'bg-[#667eea]' : ''}`}>
                  <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepVisualMode ? 'translate-x-4' : ''}`}></div>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">Deep Visual</span>
                  <p className="text-[10px] text-gray-500">Full page OCR analysis</p>
                </div>
              </div>
              <input type="checkbox" className="hidden" checked={deepVisualMode} onChange={() => setDeepVisualMode(!deepVisualMode)} />
            </label>
          </div>

          {/* Upload Area */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              group relative border-2 border-dashed rounded-2xl p-5 text-center cursor-pointer transition-all
              ${getFileUploadBorderColor()}
              hover:border-[#667eea]/60 hover:shadow-[0_0_30px_rgba(102,126,234,0.15)]
              ${fileUploadState === "uploading" ? 'pointer-events-none' : ''}
            `}
          >
            {fileUploadState === "uploading" ? (
              <div className="py-2">
                <div className="w-12 h-12 mx-auto mb-3 relative">
                  <div className="absolute inset-0 rounded-2xl border-2 border-[#667eea]/30"></div>
                  <div
                    className="absolute inset-0 rounded-2xl border-2 border-transparent border-t-[#667eea] animate-spin"
                  />
                  <Loader2 className="absolute inset-0 m-auto w-6 h-6 text-[#667eea] animate-spin" />
                </div>
                <p className="text-sm font-medium text-gray-300">Processing documents...</p>
                {uploadProgress !== null && (
                  <div className="mt-2 w-full bg-[#ffffff10] rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-[#667eea] to-[#764ba2] transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                )}
              </div>
            ) : fileUploadState === "success" ? (
              <div className="py-2">
                <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[#10b981]/20 flex items-center justify-center">
                  <CheckCircle2 className="w-6 h-6 text-[#10b981]" />
                </div>
                <p className="text-sm font-medium text-[#10b981]">Upload complete!</p>
              </div>
            ) : fileUploadState === "error" ? (
              <div className="py-2">
                <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[#ef4444]/20 flex items-center justify-center">
                  <AlertCircle className="w-6 h-6 text-[#ef4444]" />
                </div>
                <p className="text-sm font-medium text-[#ef4444]">Upload failed</p>
              </div>
            ) : fileUploadState === "dragover" ? (
              <div className="py-2">
                <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-[#10b981]/20 flex items-center justify-center animate-bounce">
                  <FileText className="w-6 h-6 text-[#10b981]" />
                </div>
                <p className="text-sm font-medium text-[#10b981]">Drop files to upload</p>
              </div>
            ) : (
              <div className="relative">
                <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-[#667eea]/20 to-[#764ba2]/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Upload className="w-6 h-6 text-[#667eea]" />
                </div>
                <p className="text-sm font-medium text-gray-300 group-hover:text-white">Drop PDF files here</p>
                <p className="text-xs text-gray-500 mt-1">or click to browse • Max 4.5MB</p>
              </div>
            )}
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="application/pdf"
              multiple
              onChange={handleFileUpload}
            />
          </div>

          {/* Uploaded Files */}
          {uploadedFiles.length > 0 && (
            <div className="mt-3 space-y-2">
              {uploadedFiles.map((f, i) => (
                <div key={i} className="flex items-center text-xs bg-[#10b981]/10 px-3 py-2.5 rounded-xl border border-[#10b981]/20">
                  <FileText className="w-4 h-4 mr-2 text-[#10b981] shrink-0" />
                  <span className="truncate flex-1 text-gray-300">{f.name}</span>
                  <span className="text-[10px] text-gray-500 ml-2">
                    {(f.size / 1024).toFixed(1)}KB
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Database Connection Section (DOCBOT-304 / DB Panel) */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-3 text-white flex items-center gap-2">
            <Database className="w-4 h-4 text-[#f97316]" />
            Database
            {isDbConnected && (
              <span className="ml-auto text-[10px] text-[#10b981] font-medium flex items-center gap-1">
                <div className="w-1.5 h-1.5 bg-[#10b981] rounded-full animate-pulse" />
                Connected
              </span>
            )}
          </h3>

          {isDbConnected ? (
            <div className="space-y-2">
              {/* Connected indicator */}
              <div className="flex items-center gap-2 px-3 py-2.5 bg-[#10b981]/10 rounded-xl border border-[#10b981]/20 text-xs">
                <Database className="w-3.5 h-3.5 text-[#10b981] shrink-0" />
                <span className="truncate text-gray-300 flex-1">{dbFileName}</span>
                <button
                  onClick={handleDbDisconnect}
                  className="text-gray-500 hover:text-red-400 transition-colors shrink-0"
                  title="Disconnect database"
                >
                  <XCircle className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* DOCBOT-405: Autopilot toggle */}
              <button
                onClick={() => setAutopilotMode(v => !v)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-xl text-xs border transition-all ${
                  autopilotMode
                    ? "border-[#667eea]/50 bg-[#667eea]/10 text-[#a5b4fc]"
                    : "border-[#ffffff10] text-gray-400 hover:text-white hover:bg-[#ffffff08]"
                }`}
              >
                <Wand2 className={`w-3.5 h-3.5 shrink-0 ${autopilotMode ? "text-[#a5b4fc]" : "text-[#667eea]"}`} />
                <span className="flex-1 text-left font-medium">Autopilot</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                  autopilotMode ? "bg-[#667eea]/30 text-[#a5b4fc]" : "bg-[#ffffff10] text-gray-500"
                }`}>
                  {autopilotMode ? "ON" : "OFF"}
                </span>
              </button>

              {/* DOCBOT-504: Query History Panel */}
              <div className="rounded-xl border border-[#ffffff10] overflow-hidden">
                <button
                  onClick={() => setHistoryOpen(v => !v)}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-400 hover:text-white hover:bg-[#ffffff08] transition-colors"
                >
                  <Clock className="w-3.5 h-3.5 text-[#f97316] shrink-0" />
                  <span className="flex-1 text-left font-medium">Query History</span>
                  {queryHistory.length > 0 && (
                    <span className="bg-[#f97316]/20 text-[#f97316] text-[10px] px-1.5 py-0.5 rounded-full font-medium">
                      {queryHistory.length}
                    </span>
                  )}
                  <ChevronDown className={`w-3 h-3 transition-transform ${historyOpen ? "rotate-180" : ""}`} />
                </button>

                {historyOpen && (
                  <div className="border-t border-[#ffffff10] max-h-64 overflow-y-auto">
                    {queryHistory.length === 0 ? (
                      <p className="px-3 py-3 text-xs text-gray-500 text-center">No queries yet</p>
                    ) : (
                      queryHistory.map((item) => (
                        <div key={item.id} className="border-b border-[#ffffff08] last:border-b-0">
                          {/* Summary row */}
                          <button
                            className="w-full flex items-start gap-2 px-3 py-2 text-left hover:bg-[#ffffff06] transition-colors group"
                            onClick={() => setExpandedHistoryId(expandedHistoryId === item.id ? null : item.id)}
                          >
                            <div className="flex-1 min-w-0">
                              <p className="text-xs text-gray-300 truncate leading-snug">{item.question}</p>
                              <div className="flex items-center gap-2 mt-0.5">
                                {item.row_count != null && (
                                  <span className="text-[10px] text-gray-500">
                                    {item.row_count} row{item.row_count !== 1 ? "s" : ""}
                                  </span>
                                )}
                                {item.executed_at && (
                                  <span className="text-[10px] text-gray-600">
                                    {new Date(item.executed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                                  </span>
                                )}
                              </div>
                            </div>
                            {/* Re-run button */}
                            <button
                              title="Re-run this query"
                              className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-gray-500 hover:text-[#f97316] mt-0.5"
                              onClick={(e) => {
                                e.stopPropagation();
                                setInput(item.question);
                              }}
                            >
                              <ArrowRight className="w-3 h-3" />
                            </button>
                          </button>

                          {/* Expanded SQL */}
                          {expandedHistoryId === item.id && (
                            <div className="px-3 pb-2">
                              <pre className="text-[10px] text-gray-400 bg-[#0d0d14] rounded-lg p-2 overflow-x-auto whitespace-pre-wrap break-all leading-relaxed">
                                {item.sql}
                              </pre>
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {/* CSV upload */}
              <label className={`flex items-center gap-2 px-3 py-2.5 rounded-xl border cursor-pointer transition-all text-xs ${dbUploadState === "uploading"
                  ? "border-[#f97316]/40 bg-[#f97316]/5 pointer-events-none"
                  : dbUploadState === "error"
                    ? "border-[#ef4444]/40 bg-[#ef4444]/5"
                    : "border-[#ffffff10] bg-[#1a1a24]/50 hover:border-[#f97316]/40 hover:bg-[#f97316]/5"
                }`}>
                {dbUploadState === "uploading"
                  ? <Loader2 className="w-3.5 h-3.5 text-[#f97316] animate-spin shrink-0" />
                  : <Upload className="w-3.5 h-3.5 text-[#f97316] shrink-0" />
                }
                <span className="text-gray-400">Upload CSV</span>
                <input
                  type="file"
                  className="hidden"
                  accept=".csv"
                  disabled={dbUploadState === "uploading"}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleDbUpload(file, "csv");
                    e.target.value = "";
                  }}
                />
              </label>

              {/* SQLite upload */}
              <label className={`flex items-center gap-2 px-3 py-2.5 rounded-xl border cursor-pointer transition-all text-xs ${dbUploadState === "uploading"
                  ? "border-[#f97316]/40 bg-[#f97316]/5 pointer-events-none"
                  : dbUploadState === "error"
                    ? "border-[#ef4444]/40 bg-[#ef4444]/5"
                    : "border-[#ffffff10] bg-[#1a1a24]/50 hover:border-[#f97316]/40 hover:bg-[#f97316]/5"
                }`}>
                {dbUploadState === "uploading"
                  ? <Loader2 className="w-3.5 h-3.5 text-[#f97316] animate-spin shrink-0" />
                  : <Database className="w-3.5 h-3.5 text-[#f97316] shrink-0" />
                }
                <span className="text-gray-400">Upload SQLite / .db</span>
                <input
                  type="file"
                  className="hidden"
                  accept=".sqlite,.db,.sqlite3"
                  disabled={dbUploadState === "uploading"}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleDbUpload(file, "sqlite");
                    e.target.value = "";
                  }}
                />
              </label>

              {/* Live DB connect toggle */}
              <button
                onClick={() => { setShowLiveDbForm(v => !v); setLiveDbError(null); }}
                className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-[#ffffff10] bg-[#1a1a24]/50 hover:border-[#f97316]/40 hover:bg-[#f97316]/5 transition-all text-xs text-gray-400"
              >
                <RefreshCw className="w-3.5 h-3.5 text-[#f97316] shrink-0" />
                <span>Connect Live DB</span>
                <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${showLiveDbForm ? "rotate-180" : ""}`} />
              </button>

              {/* Live DB form */}
              {showLiveDbForm && (
                <div className="space-y-2 pt-1">
                  {/* Dialect selector */}
                  <select
                    value={liveDbForm.dialect}
                    onChange={(e) => {
                      const dialect = e.target.value;
                      setLiveDbForm(f => ({
                        ...f,
                        dialect,
                        port: dialect === "postgresql" ? "5432"
                            : dialect === "mysql"      ? "3306"
                            : dialect === "azure_sql"  ? "1433"
                            : "0",
                        // Reset auth fields to prevent stale values across auth modes
                        user: "",
                        password: "",
                      }));
                      // Reset entra auth state when dialect changes
                      setEntraToken(null);
                      setEntraEmail(null);
                      setEntraSignInState("idle");
                    }}
                    className="w-full px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs focus:outline-none focus:border-[#f97316]/40"
                  >
                    <option value="postgresql">PostgreSQL</option>
                    <option value="mysql">MySQL</option>
                    <option value="azure_sql">Azure SQL (Microsoft Entra)</option>
                  </select>

                  <input
                    type="text"
                    placeholder="Host"
                    value={liveDbForm.host}
                    onChange={(e) => setLiveDbForm(f => ({ ...f, host: e.target.value }))}
                    className="w-full px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-600 focus:outline-none focus:border-[#f97316]/40"
                  />

                  <div className="flex gap-2">
                    <input
                      type="number"
                      placeholder="Port"
                      value={liveDbForm.port}
                      onChange={(e) => setLiveDbForm(f => ({ ...f, port: e.target.value }))}
                      className="w-20 px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-600 focus:outline-none focus:border-[#f97316]/40"
                    />
                    <input
                      type="text"
                      placeholder="Database name"
                      value={liveDbForm.dbname}
                      onChange={(e) => setLiveDbForm(f => ({ ...f, dbname: e.target.value }))}
                      className="flex-1 px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-600 focus:outline-none focus:border-[#f97316]/40"
                    />
                  </div>

                  {liveDbForm.dialect === "azure_sql" ? (
                    <div className="space-y-2">
                      {entraSignInState === "signed_in" && entraEmail ? (
                        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/20">
                          <div className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0" />
                          <span className="text-xs text-green-400 truncate">{entraEmail}</span>
                          <button
                            type="button"
                            onClick={() => { setEntraToken(null); setEntraEmail(null); setEntraSignInState("idle"); }}
                            className="ml-auto text-gray-500 hover:text-gray-300 text-[10px]"
                          >
                            Change
                          </button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={handleMicrosoftSignIn}
                          disabled={entraSignInState === "signing_in" || !process.env.NEXT_PUBLIC_AZURE_CLIENT_ID}
                          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[#0078d4]/20 hover:bg-[#0078d4]/30 border border-[#0078d4]/30 text-[#4fc3f7] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {entraSignInState === "signing_in"
                            ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Signing in...</>
                            : "Sign in with Microsoft"
                          }
                        </button>
                      )}
                      {entraSignInState === "error" && (
                        <p className="text-[10px] text-red-400 px-1">Sign-in failed. Check your Azure AD app configuration.</p>
                      )}
                      {!process.env.NEXT_PUBLIC_AZURE_CLIENT_ID && (
                        <p className="text-[10px] text-yellow-500/70 px-1">Set NEXT_PUBLIC_AZURE_CLIENT_ID to enable Microsoft sign-in.</p>
                      )}
                    </div>
                  ) : (
                    <>
                      <input
                        type="text"
                        placeholder="Username"
                        value={liveDbForm.user}
                        onChange={(e) => setLiveDbForm(f => ({ ...f, user: e.target.value }))}
                        className="w-full px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-600 focus:outline-none focus:border-[#f97316]/40"
                      />
                      <div className="relative">
                        <input
                          type={showDbPassword ? "text" : "password"}
                          placeholder="Password"
                          value={liveDbForm.password}
                          onChange={(e) => setLiveDbForm(f => ({ ...f, password: e.target.value }))}
                          className="w-full px-3 py-2 pr-8 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 text-xs placeholder-gray-600 focus:outline-none focus:border-[#f97316]/40"
                        />
                        <button
                          type="button"
                          onClick={() => setShowDbPassword(v => !v)}
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                        >
                          {showDbPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                    </>
                  )}

                  {/* PII Masking toggle — DOCBOT-604 */}
                  <button
                    type="button"
                    onClick={() => setLiveDbForm(f => ({ ...f, pii_masking_enabled: !f.pii_masking_enabled }))}
                    className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#ffffff10] hover:border-[#f97316]/20 transition-all"
                  >
                    <span className="text-xs text-gray-400">PII masking</span>
                    <div className={`relative w-8 h-4 rounded-full transition-colors ${liveDbForm.pii_masking_enabled ? "bg-[#f97316]" : "bg-[#ffffff15]"}`}>
                      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${liveDbForm.pii_masking_enabled ? "translate-x-4" : "translate-x-0.5"}`} />
                    </div>
                  </button>

                  {liveDbError && (
                    <p className="text-[10px] text-red-400 px-1">{liveDbError}</p>
                  )}

                  <button
                    onClick={handleLiveDbConnect}
                    disabled={
                      liveDbConnectState === "connecting" ||
                      !liveDbForm.host ||
                      !liveDbForm.dbname ||
                      (liveDbForm.dialect === "azure_sql"
                        ? !entraToken
                        : !liveDbForm.user)
                    }
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[#f97316]/20 hover:bg-[#f97316]/30 border border-[#f97316]/30 text-[#f97316] text-xs font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {liveDbConnectState === "connecting"
                      ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Connecting...</>
                      : <><Database className="w-3.5 h-3.5" /> Connect</>
                    }
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Persona Selection */}
        <div className="mb-6 flex-1">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-[#667eea]" />
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Expert Mode</span>
          </div>

          {suggestedPersona && suggestedPersona !== "Generalist" && (
            <div className="bg-[#667eea]/10 border border-[#667eea]/20 p-3 rounded-xl mb-3 text-sm flex items-start gap-2">
              <Zap className="w-4 h-4 text-[#667eea] mt-0.5 shrink-0" />
              <div>
                <span className="text-gray-300">Recommended: </span>
                <strong className="text-white">{suggestedPersona}</strong>
              </div>
            </div>
          )}

          {/* AUTO / Manual toggle */}
          <div className="flex gap-2 mb-3">
            <button
              onClick={() => setIsAutoMode(true)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isAutoMode
                  ? "bg-[#667eea]/20 text-[#667eea] border border-[#667eea]/40"
                  : "bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10"
              }`}
            >
              <Wand2 className="w-3 h-3" />
              AUTO
            </button>
            <button
              onClick={() => { setIsAutoMode(false); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                !isAutoMode
                  ? "bg-white/10 text-gray-200 border border-white/20"
                  : "bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10"
              }`}
            >
              <UserCog className="w-3 h-3" />
              Manual
            </button>
          </div>

          {isAutoMode ? (
            <p className="text-[11px] text-gray-500 mb-3">Routing automatically based on your question</p>
          ) : (
            <div>
              {/* Persona Cards Grid — preserved in Manual Override section */}
              <div className="grid grid-cols-2 gap-2 mb-2">
                {Object.entries(EXPERT_PERSONAS).map(([name, data]) => {
                  const isSelected = selectedPersona === name;
                  return (
                    <button
                      key={name}
                      onClick={() => setSelectedPersona(name)}
                      className={`group relative p-3 rounded-xl text-left transition-all duration-200 ${isSelected
                        ? 'bg-gradient-to-br ' + data.gradient + ' shadow-lg scale-[1.02]'
                        : 'bg-[#1a1a24]/50 border border-[#ffffff06] hover:border-[#ffffff15] hover:bg-[#1a1a24]/80'
                        }`}
                    >
                      <div className={`${isSelected ? 'text-white' : data.color} mb-1`}>
                        {data.icon}
                      </div>
                      <div className={`text-xs font-medium ${isSelected ? 'text-white' : 'text-gray-300'}`}>
                        {name}
                      </div>
                      {isSelected && (
                        <div className="absolute top-2 right-2">
                          <CheckCircle2 className="w-3 h-3 text-white" />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
              <button
                onClick={() => setIsAutoMode(true)}
                className="text-[10px] text-[#667eea] hover:underline mt-1"
              >
                ↩ Reset to Auto
              </button>
            </div>
          )}

          {!isAutoMode && (
            <div className="mt-3 p-3 bg-[#1a1a24]/30 rounded-xl border border-[#ffffff06]">
              <p className="text-xs text-gray-400 flex items-start gap-2">
                <Info className="w-3 h-3 mt-0.5 shrink-0 text-[#667eea]" />
                {EXPERT_PERSONAS[selectedPersona]?.description}
              </p>
            </div>
          )}
        </div>

        {/* Deep Research Toggle */}
        {selectedPersona !== "Generalist" && (
          <div className="mb-4 bg-[#1a1a24]/50 rounded-xl p-3 border border-[#ffffff06]">
            <h3 className="text-sm font-semibold mb-3 text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#764ba2]" />
              Analysis Mode
            </h3>
            <label className="flex items-center justify-between cursor-pointer group">
              <div className="flex items-center gap-3">
                <div className={`w-9 h-5 flex items-center bg-gray-700/50 rounded-full p-0.5 transition-colors ${deepResearch ? 'bg-[#764ba2]' : ''}`}>
                  <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepResearch ? 'translate-x-4' : ''}`}></div>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">Deep Research</span>
                  <p className="text-[10px] text-gray-500">Enhanced reasoning</p>
                </div>
              </div>
              <input type="checkbox" className="hidden" checked={deepResearch} onChange={() => setDeepResearch(!deepResearch)} />
            </label>
            {deepResearch && (
              <p className="text-xs text-gray-500 mt-2 ml-12 flex items-center gap-1">
                <Brain className="w-3 h-3 text-[#764ba2]" /> Advanced analysis enabled
              </p>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="pt-4 border-t border-[#ffffff08]">
          <button
            onClick={clearChat}
            disabled={messages.length === 0}
            className="w-full flex items-center justify-center py-2.5 px-4 rounded-xl bg-[#1a1a24]/80 border border-[#ffffff06] hover:bg-red-500/15 hover:border-red-500/30 hover:text-red-400 transition-all text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-[#1a1a24]/80 disabled:hover:border-[#ffffff06] disabled:hover:text-inherit"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Clear Chat
          </button>

          {/* Keyboard Shortcuts Help */}
          <div className="mt-3 p-3 bg-[#1a1a24]/30 rounded-xl border border-[#ffffff06]">
            <div className="flex items-center gap-2 text-xs text-gray-500 mb-2">
              <Keyboard className="w-3 h-3" />
              <span className="font-medium">Keyboard Shortcuts</span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-500">
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-[#ffffff10] rounded text-gray-400">Ctrl</kbd>
                <span>+</span>
                <kbd className="px-1.5 py-0.5 bg-[#ffffff10] rounded text-gray-400">↵</kbd>
                <span className="ml-1">Send</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-[#ffffff10] rounded text-gray-400">Esc</kbd>
                <span className="ml-1">Clear</span>
              </div>
            </div>
          </div>

          <div className="mt-4 text-center text-xs text-gray-600">
            <p>Built by <a href="https://sanshrit-singhai.vercel.app" className="text-[#667eea] hover:underline" target="_blank" rel="noopener noreferrer">Sanshrit Singhai</a></p>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-transparent relative z-10">
        {/* Header */}
        <header className="px-4 pt-14 lg:pt-6 lg:px-6 flex-none">
          <div className="relative overflow-hidden bg-gradient-to-br from-[#12121a]/95 to-[#1a1a28]/95 rounded-2xl border border-[#ffffff08] shadow-2xl p-4 lg:p-6 max-w-5xl mx-auto">
            <div className="absolute inset-0 bg-gradient-to-tr from-[#667eea]/5 via-transparent to-[#764ba2]/5"></div>
            <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="hidden lg:block">
                  <h1 className="text-2xl lg:text-3xl font-bold text-white mb-1 flex items-center gap-3">
                    <span className="bg-gradient-to-br from-[#667eea] to-[#764ba2] bg-clip-text text-transparent">
                      Ask Anything About Your Data
                    </span>
                  </h1>
                  <p className="text-gray-400 text-sm flex items-center gap-2">
                    <Terminal className="w-3 h-3" />
                    Upload PDFs or connect a database
                  </p>
                </div>
                <div className="lg:hidden">
                  <h1 className="text-xl font-bold text-white flex items-center gap-2">
                    <span className="bg-gradient-to-br from-[#667eea] to-[#764ba2] bg-clip-text text-transparent">
                      DocBot
                    </span>
                  </h1>
                  <p className="text-gray-500 text-xs">Docs · Databases · Hybrid</p>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 items-center">
                <SessionInfo
                  sessionId={sessionId}
                  fileCount={uploadedFiles.length}
                  persona={selectedPersona}
                  onClear={clearSession}
                />

                {/* DOCBOT-404: HybridModeToggle */}
                <div className="flex items-center gap-0.5 p-1 bg-[#1a1a24]/80 rounded-xl border border-[#ffffff08]">
                  {(["docs", "database", "hybrid"] as const).map((mode) => {
                    const disabled =
                      (mode === "database" && !connectionId) ||
                      (mode === "hybrid" && (!connectionId || !sessionId));
                    const labels = { docs: "Docs", database: "Database", hybrid: "Hybrid" };
                    const icons = {
                      docs: <FileText className="w-3 h-3" />,
                      database: <Database className="w-3 h-3" />,
                      hybrid: <Layers className="w-3 h-3" />,
                    };
                    return (
                      <button
                        key={mode}
                        onClick={() => !disabled && setChatMode(mode)}
                        title={
                          mode === "database" && !connectionId
                            ? "Connect a database first"
                            : mode === "hybrid" && !connectionId
                              ? "Connect a database first"
                              : mode === "hybrid" && !sessionId
                                ? "Upload a PDF first"
                                : undefined
                        }
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${disabled
                            ? "opacity-40 cursor-not-allowed pointer-events-none text-gray-400"
                            : chatMode === mode
                              ? "bg-gradient-to-r from-[#667eea] to-[#764ba2] text-white shadow-sm"
                              : "text-gray-400 hover:text-gray-200 hover:bg-[#ffffff08]"
                          }`}
                      >
                        {icons[mode]}
                        {labels[mode]}
                      </button>
                    );
                  })}
                </div>
                {sessionId && messages.length > 0 && (
                  <div className="relative group">
                    <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-300 bg-[#1a1a24]/60 border border-[#ffffff08] hover:border-[#667eea]/30 hover:bg-[#1a1a24]/80 transition-all">
                      <Download className="w-3 h-3" />
                      Export
                    </button>
                    <div className="absolute top-full right-0 mt-1 py-1 bg-[#1a1a24]/95 backdrop-blur-xl rounded-lg border border-[#ffffff08] shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 min-w-[140px]">
                      <button
                        onClick={() => exportChat('txt')}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-[#ffffff08] hover:text-white transition-colors"
                      >
                        <FileTxt className="w-3 h-3" />
                        Text (.txt)
                      </button>
                      <button
                        onClick={() => exportChat('markdown')}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-[#ffffff08] hover:text-white transition-colors"
                      >
                        <FileText className="w-3 h-3" />
                        Markdown (.md)
                      </button>
                      <button
                        onClick={() => exportChat('json')}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:bg-[#ffffff08] hover:text-white transition-colors"
                      >
                        <FileJson className="w-3 h-3" />
                        JSON (.json)
                      </button>
                    </div>
                  </div>
                )}
                <div className="flex gap-2">
                  <span className="bg-[#667eea]/10 px-3 py-1.5 rounded-lg text-xs text-gray-300 border border-[#667eea]/20 flex items-center gap-1.5">
                    <FileText className="w-3 h-3" /> PDF
                  </span>
                  <span className="bg-[#764ba2]/10 px-3 py-1.5 rounded-lg text-xs text-gray-300 border border-[#764ba2]/20 flex items-center gap-1.5">
                    <Brain className="w-3 h-3" /> AI
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Chat Output */}
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 lg:px-6 pb-4 max-w-5xl mx-auto w-full">
          {!sessionId && !isDbConnected && messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <div className="relative">
                <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-[#667eea]/20 to-[#764ba2]/20 flex items-center justify-center mb-6 animate-pulse">
                  <Upload className="w-12 h-12 text-[#667eea]" />
                </div>
                <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-[#10b981] rounded-full flex items-center justify-center shadow-lg">
                  <ArrowRight className="w-4 h-4 text-white" />
                </div>
              </div>
              <p className="text-xl font-medium text-gray-300 mb-2 text-center">Upload a document or connect a database</p>
              <p className="text-sm text-gray-500 text-center max-w-md mb-6">
                Upload PDFs to chat with documents, or upload a CSV/SQLite file to query your data with AI.
              </p>

              {/* Feature highlights */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl">
                <div className="flex flex-col items-center p-4 bg-[#12121a]/40 rounded-xl border border-[#ffffff06] max-w-[140px]">
                  <Stethoscope className="w-6 h-6 text-[#10b981] mb-2" />
                  <span className="text-xs text-gray-400 text-center">Medical Docs</span>
                </div>
                <div className="flex flex-col items-center p-4 bg-[#12121a]/40 rounded-xl border border-[#ffffff06] max-w-[140px]">
                  <Database className="w-6 h-6 text-[#f97316] mb-2" />
                  <span className="text-xs text-gray-400 text-center">CSV / SQLite</span>
                </div>
                <div className="flex flex-col items-center p-4 bg-[#12121a]/40 rounded-xl border border-[#ffffff06] max-w-[140px]">
                  <Layers className="w-6 h-6 text-[#8b5cf6] mb-2" />
                  <span className="text-xs text-gray-400 text-center">Hybrid Analysis</span>
                </div>
              </div>
            </div>
          ) : messages.length === 0 && (sessionId || isDbConnected) ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#667eea]/20 to-[#764ba2]/20 flex items-center justify-center mb-4">
                <MessageSquare className="w-8 h-8 text-[#667eea]" />
              </div>
              <p className="text-lg font-medium text-gray-300 mb-2">
                {isDbConnected ? "Database connected — ready to query!" : "Ready to chat!"}
              </p>
              <p className="text-sm text-gray-500">
                {isDbConnected
                  ? `Ask questions about your data in ${chatMode} mode`
                  : "Ask me anything about your uploaded documents"}
              </p>

              {/* Suggested questions */}
              <div className="mt-8 grid gap-2 max-w-lg w-full">
                <p className="text-xs text-gray-500 text-center mb-2">Try asking:</p>
                {(isDbConnected
                  ? [
                    "How many rows are in my dataset?",
                    "Show me a summary of each column",
                    "What are the top 10 records by value?",
                    "Are there any missing or null values?",
                  ]
                  : [
                    "Summarize the main points of this document",
                    "What are the key findings?",
                    "Extract all tables and figures",
                    "What are the action items?",
                  ]
                ).map((question, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInput(question)}
                    className="text-left p-3 bg-[#12121a]/60 border border-[#ffffff08] rounded-xl text-sm text-gray-400 hover:text-white hover:border-[#667eea]/30 hover:bg-[#1a1a24]/60 transition-all"
                  >
                    <Sparkles className="w-3 h-3 inline mr-2 text-[#667eea]" />
                    {question}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4 max-w-5xl mx-auto pb-20">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  ref={idx === messages.length - 1 ? lastMessageRef : null}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-in slide-in-from-bottom-2 duration-300`}
                >
                  <div
                    className={`max-w-[85%] lg:max-w-[80%] rounded-2xl p-4 shadow-lg border backdrop-blur-xl group
                      ${msg.role === "user"
                        ? "bg-gradient-to-br from-[#667eea] to-[#764ba2] text-white border-transparent rounded-br-sm"
                        : "bg-[#12121a]/80 border-[#ffffff08] text-[#e0e0e0] rounded-bl-sm hover:border-[#667eea]/20 transition-colors"
                      }`}
                  >
                    {/* Message Header */}
                    {msg.role === "assistant" && (
                      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[#ffffff08]">
                        <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-[#667eea] to-[#764ba2] flex items-center justify-center">
                          <Brain className="w-3 h-3 text-white" />
                        </div>
                        {/* DOCBOT-804: Dynamic agent badge */}
                        {(() => {
                          const agentName = msg.agentPersona ?? "DocBot";
                          const agentConf = EXPERT_PERSONAS[agentName as keyof typeof EXPERT_PERSONAS];
                          const accentColor = agentConf?.output_conventions?.accent_color ?? "#667eea";

                          if (msg.agentPersonas && msg.agentPersonas.length > 1) {
                            return (
                              <div className="flex items-center gap-1">
                                {msg.agentPersonas.map((ap) => {
                                  const conf = EXPERT_PERSONAS[ap as keyof typeof EXPERT_PERSONAS];
                                  const color = conf?.output_conventions?.accent_color ?? "#667eea";
                                  return (
                                    <span key={ap} className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                                      style={{ backgroundColor: color + "20", color, border: `1px solid ${color}40` }}>
                                      {ap}
                                    </span>
                                  );
                                })}
                              </div>
                            );
                          }

                          return (
                            <span className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                              style={{ backgroundColor: accentColor + "20", color: accentColor, border: `1px solid ${accentColor}40` }}>
                              {agentName}
                            </span>
                          );
                        })()}
                        {msg.timestamp && (
                          <>
                            <span className="text-[10px] text-gray-600">•</span>
                            <span className="text-[10px] text-gray-600 flex items-center gap-1">
                              <Clock className="w-2 h-2" />
                              {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </>
                        )}
                      </div>
                    )}

                    {msg.role === "user" && (
                      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[#ffffff10]">
                        <span className="text-xs font-medium text-white/80">You</span>
                        {msg.timestamp && (
                          <span className="text-[10px] text-white/40 flex items-center gap-1 ml-auto">
                            <Clock className="w-2 h-2" />
                            {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Message Content */}
                    {msg.role === "assistant" ? (
                      msg.content === "" && isLoading ? (
                        /* Thinking animation — shown while waiting for first token */
                        <div className="flex flex-col gap-2.5 py-1">
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1.5">
                              <div className="w-2 h-2 rounded-full bg-[#667eea] animate-bounce" style={{ animationDelay: "0ms", animationDuration: "1s" }} />
                              <div className="w-2 h-2 rounded-full bg-[#764ba2] animate-bounce" style={{ animationDelay: "180ms", animationDuration: "1s" }} />
                              <div className="w-2 h-2 rounded-full bg-[#8b5cf6] animate-bounce" style={{ animationDelay: "360ms", animationDuration: "1s" }} />
                            </div>
                            <span className="text-xs text-gray-500 animate-pulse">Thinking…</span>
                          </div>
                          <div className="space-y-2 opacity-40">
                            <div className="h-2.5 rounded-full bg-gradient-to-r from-[#667eea]/30 to-transparent animate-pulse" style={{ width: "72%", animationDelay: "0ms" }} />
                            <div className="h-2.5 rounded-full bg-gradient-to-r from-[#667eea]/20 to-transparent animate-pulse" style={{ width: "55%", animationDelay: "200ms" }} />
                            <div className="h-2.5 rounded-full bg-gradient-to-r from-[#667eea]/10 to-transparent animate-pulse" style={{ width: "40%", animationDelay: "400ms" }} />
                          </div>
                        </div>
                      ) : (
                        <AgentMessageContent msg={msg} />
                      )
                    ) : (
                      <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    )}

                    {/* Analysis Code (DOCBOT-303) */}
                    {msg.role === "assistant" && msg.analysisCode && (
                      <CollapsibleCode code={msg.analysisCode} />
                    )}

                    {/* Charts (DOCBOT-303) */}
                    {msg.role === "assistant" && msg.charts && msg.charts.length > 0 && (
                      <ChartDisplay charts={msg.charts} chartMetas={msg.chartMetas} />
                    )}

                    {/* DOCBOT-405: Autopilot investigation steps — persisted after run completes */}
                    {msg.role === "assistant" && msg.autopilotSteps && msg.autopilotSteps.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-[#ffffff08] space-y-2">
                        <div className="flex items-center gap-1.5 mb-2">
                          <Wand2 className="w-3 h-3 text-[#a5b4fc]" />
                          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                            Investigation Steps
                          </span>
                          <span className="text-[10px] text-gray-600 ml-1">
                            ({msg.autopilotSteps.length})
                          </span>
                        </div>
                        {msg.autopilotSteps.map((step) => (
                          <div key={step.step_num} className="bg-[#ffffff04] rounded-xl p-2.5 text-xs border border-[#ffffff08]">
                            <div className="flex items-center gap-1.5 mb-1.5">
                              <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase tracking-wide ${
                                step.tool === "sql_query"       ? "bg-[#0ea5e9]/20 text-[#38bdf8]" :
                                step.tool === "doc_search"      ? "bg-[#f59e0b]/20 text-[#fbbf24]" :
                                                                  "bg-[#10b981]/20 text-[#34d399]"
                              }`}>
                                {step.tool.replace("_", " ")}
                              </span>
                              <span className="text-gray-500 truncate flex-1">{step.step_label}</span>
                            </div>
                            {step.content && (
                              <p className="text-gray-400 leading-relaxed">{step.content}</p>
                            )}
                            {step.chart_b64 && (
                              <div className="mt-2">
                                <ChartDisplay
                                  charts={[step.chart_b64]}
                                  chartMetas={undefined}
                                />
                              </div>
                            )}
                            {step.error && (
                              <p className="text-red-400 mt-1 text-[11px]">⚠ {step.error}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Citations */}
                    {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-[#ffffff08]">
                        <div className="flex items-center gap-2 mb-3">
                          <BookOpen className="w-3 h-3 text-[#667eea]" />
                          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">References</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {msg.citations.map((citation, idx) => (
                            <div
                              key={idx}
                              className="text-[11px] px-3 py-2 bg-[#1a1a24]/80 rounded-lg border border-[#ffffff10] text-gray-400 hover:text-gray-300 hover:border-[#667eea]/30 hover:bg-[#1a1a24] transition-all cursor-pointer group"
                              title={`${citation.source} - Page ${citation.page}`}
                            >
                              <span className="text-[#667eea] font-medium">{citation.source}</span>
                              <span className="text-gray-600 mx-1.5">•</span>
                              <span className="text-gray-500">Page {citation.page}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Message Actions */}
                    {msg.role === "assistant" && (
                      <div className="flex gap-2 mt-2 pt-2 border-t border-[#ffffff08] opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => copyToClipboard(msg.content)}
                          className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
                        >
                          <Copy className="w-3 h-3" />
                          Copy
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {/* DOCBOT-405: Autopilot live step-by-step progress panel */}
              {autopilotRunning && (autopilotPlan.length > 0 || autopilotSteps.length > 0) && (
                <div className="flex justify-start animate-in slide-in-from-bottom-2 duration-300">
                  <div className="max-w-[85%] lg:max-w-[80%] rounded-2xl p-4 bg-[#12121a]/80 border border-[#667eea]/20 text-[#e0e0e0] shadow-lg">
                    <div className="flex items-center gap-2 mb-3 pb-2 border-b border-[#ffffff08]">
                      <Wand2 className="w-4 h-4 text-[#a5b4fc]" />
                      <span className="text-xs font-semibold text-[#a5b4fc]">Autopilot Investigation</span>
                      <span className="ml-auto text-[10px] text-gray-500">{autopilotSteps.length}/{autopilotPlan.length} steps</span>
                    </div>
                    {/* Plan overview */}
                    {autopilotPlan.length > 0 && (
                      <div className="mb-3 space-y-1">
                        {autopilotPlan.map((planStep, pi) => {
                          const completed = autopilotSteps.find(s => s.step_num === pi + 1);
                          const isCurrent = !completed && autopilotSteps.length === pi;
                          return (
                            <div key={pi} className={`flex items-start gap-2 text-xs py-1 ${completed ? "text-gray-400" : isCurrent ? "text-white" : "text-gray-600"}`}>
                              <span className={`shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold mt-0.5 ${
                                completed ? "bg-[#10b981]/20 text-[#10b981]" : isCurrent ? "bg-[#667eea]/30 text-[#a5b4fc]" : "bg-[#ffffff08] text-gray-600"
                              }`}>
                                {completed ? "✓" : pi + 1}
                              </span>
                              <span className="leading-tight">{planStep}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                    {/* Completed steps with results */}
                    {autopilotSteps.length > 0 && (
                      <div className="space-y-2 border-t border-[#ffffff08] pt-2">
                        {autopilotSteps.map((step) => (
                          <div key={step.step_num} className="bg-[#ffffff05] rounded-xl p-2.5 text-xs">
                            <div className="flex items-center gap-1.5 mb-1">
                              <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase tracking-wide ${
                                step.tool === "sql_query" ? "bg-[#0ea5e9]/20 text-[#38bdf8]" :
                                step.tool === "doc_search" ? "bg-[#f59e0b]/20 text-[#fbbf24]" :
                                "bg-[#10b981]/20 text-[#34d399]"
                              }`}>{step.tool.replace("_", " ")}</span>
                              <span className="text-gray-500 truncate">{step.step_label}</span>
                            </div>
                            {step.content && (
                              <p className="text-gray-400 line-clamp-2 leading-relaxed">{step.content}</p>
                            )}
                            {step.chart_b64 && (
                              <img src={`data:image/png;base64,${step.chart_b64}`} alt="chart" className="mt-2 rounded-lg max-h-32 object-contain" />
                            )}
                            {step.error && (
                              <p className="text-red-400 mt-1">⚠ {step.error}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Spinner while running */}
                    <div className="flex items-center gap-2 mt-2 pt-2 border-t border-[#ffffff08]">
                      <div className="w-3 h-3 rounded-full bg-[#667eea]/40 animate-pulse" />
                      <span className="text-[11px] text-gray-500">Investigating…</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Deep Research progress strip */}
              {isLoading && drProgress && deepResearch && (
                <div className="flex justify-start mb-2">
                  <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-indigo-950/40 border border-indigo-500/20 text-indigo-300 text-xs max-w-sm">
                    <span className="animate-pulse shrink-0">
                      {drProgress.step === "planning" && "🧠"}
                      {drProgress.step === "retrieving" && "🔍"}
                      {drProgress.step === "evaluating" && "✅"}
                      {drProgress.step === "gap_fill" && "🔄"}
                      {drProgress.step === "synthesizing" && "📝"}
                    </span>
                    <span className="truncate">{drProgress.message}</span>
                  </div>
                </div>
              )}
              {isLoading && messages.length > 0 && messages[messages.length - 1].role === "user" && !autopilotRunning && !drProgress && (
                <div className="flex justify-start">
                  <TypingIndicator />
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 lg:p-6 pt-0 flex-none w-full">
          {/* DOCBOT-305: Chart type selector — shown only when DB connected */}
          {isDbConnected && (
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <span className="text-[11px] text-gray-500 shrink-0">Chart:</span>
              {(["auto", "bar", "line", "scatter", "heatmap", "box", "multi"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setChartType(t)}
                  className={`px-2 py-0.5 rounded-full text-[11px] border transition-all capitalize ${
                    chartType === t
                      ? "border-[#667eea]/60 bg-[#667eea]/15 text-[#a5b4fc]"
                      : "border-[#ffffff10] text-gray-500 hover:text-gray-300 hover:border-[#ffffff20]"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          )}
          {/* Autopilot nudge — shown when question keywords suggest a multi-step investigation */}
          {showAutopilotNudge && (
            <div className="flex items-center gap-2 mb-2 px-3 py-1.5 rounded-xl bg-[#a5b4fc]/8 border border-[#a5b4fc]/20 text-xs text-[#c4b5fd]">
              <Wand2 className="w-3 h-3 shrink-0 text-[#a5b4fc]" />
              <span className="flex-1">This looks like a multi-step question — try Autopilot?</span>
              <button
                type="button"
                onClick={() => setAutopilotMode(true)}
                className="px-2 py-0.5 rounded-lg bg-[#a5b4fc]/20 hover:bg-[#a5b4fc]/30 text-[#a5b4fc] font-semibold transition-colors text-[11px] shrink-0"
              >
                Enable
              </button>
            </div>
          )}
          <form
            onSubmit={handleSendMessage}
            className="relative bg-[#12121a]/90 border border-[#ffffff10] rounded-2xl backdrop-blur-xl shadow-lg shadow-black/20 focus-within:border-[#667eea]/40 focus-within:shadow-[0_8px_30px_rgba(102,126,234,0.15)] transition-all overflow-hidden flex items-end"
          >
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder={
                  isDbConnected
                    ? "Ask a question about your data..."
                    : sessionId
                      ? "Ask a question about your document..."
                      : "Upload a document or connect a database to start..."
                }
                disabled={(!sessionId && !isDbConnected) || isLoading}
                className="w-full max-h-40 min-h-[60px] p-4 pr-12 bg-transparent outline-none resize-none text-white placeholder-gray-500 text-[15px] leading-relaxed disabled:opacity-50"
                rows={1}
              />
              <div className="absolute bottom-3 right-3 text-[10px] text-gray-600 flex items-center gap-1.5">
                {input.length > 0 && <span>{input.length} chars</span>}
              </div>
            </div>
            <button
              type="submit"
              disabled={!input.trim() || (!sessionId && !isDbConnected) || isLoading}
              className="m-2 p-3.5 rounded-xl bg-gradient-to-tr from-[#667eea] to-[#764ba2] text-white shadow-lg shadow-[#667eea]/20 disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-xl hover:shadow-[#667eea]/30 transition-all focus:outline-none focus:ring-2 focus:ring-[#764ba2]/50 hover:scale-105 active:scale-95"
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </form>
          <div className="text-center mt-3 text-xs text-gray-600 font-medium flex items-center justify-center gap-2">
            <HelpCircle className="w-3 h-3" />
            AI can make mistakes. Consider verifying important information.
          </div>
        </div>

      </main>

      {/* ── DOCBOT-701: Auth Modal ────────────────────────────────────────── */}
      {authModalOpen && !authUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
            onClick={() => setAuthModalOpen(false)}
          />
          <div className="relative z-10 w-full max-w-sm bg-[#12121a] border border-[#ffffff10] rounded-2xl shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-[#ffffff08]">
              <div>
                <h2 className="text-base font-bold text-white">Welcome to DocBot</h2>
                <p className="text-xs text-gray-500 mt-0.5">Sign in to save your work across sessions</p>
              </div>
              <button
                onClick={() => setAuthModalOpen(false)}
                className="p-1.5 rounded-lg hover:bg-[#ffffff10] text-gray-500 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-3">
              {/* OAuth buttons — fetch URL from backend then navigate directly
                  (avoids Next.js rewrite following the redirect server-side) */}
              {authConfig?.github && (
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch("/api/auth/github");
                      const { url } = await res.json();
                      window.location.href = url;
                    } catch { setAuthError("Failed to start GitHub sign-in."); }
                  }}
                  className="flex items-center justify-center gap-2.5 w-full px-4 py-2.5 rounded-xl bg-[#24292e] hover:bg-[#2f363d] border border-[#ffffff15] text-white text-sm font-medium transition-all"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                  </svg>
                  Continue with GitHub
                </button>
              )}

              {authConfig?.google && (
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch("/api/auth/google");
                      const { url } = await res.json();
                      window.location.href = url;
                    } catch { setAuthError("Failed to start Google sign-in."); }
                  }}
                  className="flex items-center justify-center gap-2.5 w-full px-4 py-2.5 rounded-xl bg-[#1a1a24] hover:bg-[#22222e] border border-[#ffffff10] text-white text-sm font-medium transition-all"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24">
                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                  </svg>
                  Continue with Google
                </button>
              )}

              {(authConfig?.github || authConfig?.google) && (
                <div className="flex items-center gap-3 my-1">
                  <div className="flex-1 h-px bg-[#ffffff10]" />
                  <span className="text-[11px] text-gray-600">or</span>
                  <div className="flex-1 h-px bg-[#ffffff10]" />
                </div>
              )}

              {/* Email / Password */}
              <div className="flex rounded-xl bg-[#1a1a24] border border-[#ffffff08] overflow-hidden text-xs">
                <button
                  onClick={() => { setAuthModalTab("login"); setAuthError(null); }}
                  className={`flex-1 py-2 font-medium transition-colors ${authModalTab === "login" ? "bg-[#667eea]/20 text-[#a5b4fc]" : "text-gray-500 hover:text-gray-300"}`}
                >
                  Sign in
                </button>
                <button
                  onClick={() => { setAuthModalTab("register"); setAuthError(null); }}
                  className={`flex-1 py-2 font-medium transition-colors ${authModalTab === "register" ? "bg-[#667eea]/20 text-[#a5b4fc]" : "text-gray-500 hover:text-gray-300"}`}
                >
                  Create account
                </button>
              </div>

              <form onSubmit={handleEmailAuth} className="space-y-2.5">
                {authModalTab === "register" && (
                  <input
                    type="text"
                    placeholder="Your name (optional)"
                    value={authName}
                    onChange={e => setAuthName(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl bg-[#1a1a24] border border-[#ffffff10] text-white text-sm placeholder-gray-600 outline-none focus:border-[#667eea]/50 transition-colors"
                  />
                )}
                <input
                  type="email"
                  placeholder="Email address"
                  value={authEmail}
                  onChange={e => setAuthEmail(e.target.value)}
                  required
                  className="w-full px-3.5 py-2.5 rounded-xl bg-[#1a1a24] border border-[#ffffff10] text-white text-sm placeholder-gray-600 outline-none focus:border-[#667eea]/50 transition-colors"
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={authPassword}
                  onChange={e => setAuthPassword(e.target.value)}
                  required
                  className="w-full px-3.5 py-2.5 rounded-xl bg-[#1a1a24] border border-[#ffffff10] text-white text-sm placeholder-gray-600 outline-none focus:border-[#667eea]/50 transition-colors"
                />
                {authError && (
                  <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{authError}</p>
                )}
                <button
                  type="submit"
                  disabled={authSubmitting || !authEmail || !authPassword}
                  className="w-full py-2.5 rounded-xl bg-[#667eea] hover:bg-[#5a6fd4] disabled:opacity-50 text-white text-sm font-medium transition-colors"
                >
                  {authSubmitting ? "Please wait…" : authModalTab === "register" ? "Create account" : "Sign in"}
                </button>
              </form>

              {/* Guest option */}
              <button
                onClick={() => setAuthModalOpen(false)}
                className="w-full py-2 text-xs text-gray-600 hover:text-gray-400 transition-colors"
              >
                Continue as guest (no account needed)
              </button>

              {authConfig?.saml && (
                <a
                  href="/api/auth/saml/login"
                  className="flex items-center justify-center gap-1.5 w-full text-xs text-gray-600 hover:text-[#a5b4fc] transition-colors"
                >
                  <ShieldCheck className="w-3 h-3" />
                  Sign in with enterprise SSO
                </a>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── DOCBOT-606: Admin Panel Modal ───────────────────────────────── */}
      {adminPanelOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
            onClick={() => setAdminPanelOpen(false)}
          />

          <div className="relative z-10 w-full max-w-4xl max-h-[85vh] flex flex-col bg-[#12121a] border border-[#ffffff10] rounded-2xl shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 px-5 py-4 border-b border-[#ffffff08] shrink-0">
              <div className="w-8 h-8 rounded-lg bg-[#f59e0b]/20 flex items-center justify-center">
                <Shield className="w-4 h-4 text-[#f59e0b]" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-white">Admin Panel</h2>
                <p className="text-[11px] text-gray-500">User management &amp; audit log</p>
              </div>
              <div className="ml-auto flex items-center gap-2">
                {/* Tab switcher */}
                <div className="flex gap-1 p-1 bg-[#1a1a24] rounded-lg border border-[#ffffff08]">
                  <button
                    onClick={() => setAdminTab("users")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                      adminTab === "users"
                        ? "bg-[#667eea]/20 text-[#a5b4fc] border border-[#667eea]/30"
                        : "text-gray-400 hover:text-gray-200 hover:bg-[#ffffff08]"
                    }`}
                  >
                    <Users className="w-3 h-3" />
                    Users
                  </button>
                  <button
                    onClick={() => { setAdminTab("audit"); loadAuditLog(auditEventTypeFilter); }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                      adminTab === "audit"
                        ? "bg-[#667eea]/20 text-[#a5b4fc] border border-[#667eea]/30"
                        : "text-gray-400 hover:text-gray-200 hover:bg-[#ffffff08]"
                    }`}
                  >
                    <ClipboardList className="w-3 h-3" />
                    Audit Log
                  </button>
                </div>
                <button
                  onClick={() => setAdminPanelOpen(false)}
                  className="p-1.5 rounded-lg hover:bg-[#ffffff10] text-gray-500 hover:text-gray-200 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto">
              {adminTab === "users" && (
                <div className="p-5">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-xs text-gray-400">
                      {adminUsersLoading ? "Loading…" : `${adminUsers.length} user${adminUsers.length !== 1 ? "s" : ""}`}
                    </p>
                    <button
                      onClick={loadAdminUsers}
                      disabled={adminUsersLoading}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-200 bg-[#1a1a24] border border-[#ffffff08] hover:border-[#ffffff15] transition-all disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3 h-3 ${adminUsersLoading ? "animate-spin" : ""}`} />
                      Refresh
                    </button>
                  </div>

                  {adminUsersLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="w-6 h-6 text-[#667eea] animate-spin" />
                    </div>
                  ) : adminUsers.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                      <Users className="w-8 h-8 mb-3 opacity-40" />
                      <p className="text-sm">No users found</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-[#ffffff08]">
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-4">User</th>
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-4">Provider</th>
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-4">Last Login</th>
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2">Role</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#ffffff06]">
                          {adminUsers.map((user) => (
                            <tr key={user.id} className="hover:bg-[#ffffff03] transition-colors">
                              <td className="py-3 pr-4">
                                <div>
                                  <p className="text-xs font-medium text-gray-200 truncate max-w-[200px]">{user.email}</p>
                                  <p className="text-[10px] text-gray-500 truncate max-w-[200px]">{user.name}</p>
                                </div>
                              </td>
                              <td className="py-3 pr-4">
                                <span className="text-[11px] text-gray-400 capitalize">{user.provider}</span>
                              </td>
                              <td className="py-3 pr-4">
                                <span className="text-[11px] text-gray-500">
                                  {user.last_login_at
                                    ? new Date(user.last_login_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })
                                    : "Never"}
                                </span>
                              </td>
                              <td className="py-3">
                                <select
                                  value={user.role}
                                  onChange={(e) => {
                                    // Prevent self-demotion of current user
                                    if (authUser?.id === user.id && e.target.value !== "admin") {
                                      showToast("warning", "You cannot change your own role");
                                      return;
                                    }
                                    updateUserRole(user.id, e.target.value);
                                  }}
                                  className="text-[11px] px-2 py-1 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 focus:outline-none focus:border-[#667eea]/40 cursor-pointer"
                                >
                                  <option value="viewer">viewer</option>
                                  <option value="analyst">analyst</option>
                                  <option value="admin">admin</option>
                                </select>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {adminTab === "audit" && (
                <div className="p-5">
                  <div className="flex items-center gap-3 mb-4 flex-wrap">
                    {/* Event type filter */}
                    <div className="flex items-center gap-2">
                      <Filter className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                      <select
                        value={auditEventTypeFilter}
                        onChange={(e) => {
                          setAuditEventTypeFilter(e.target.value);
                          loadAuditLog(e.target.value);
                        }}
                        className="text-xs px-2.5 py-1.5 rounded-lg bg-[#1a1a24] border border-[#ffffff10] text-gray-300 focus:outline-none focus:border-[#667eea]/40"
                      >
                        <option value="all">All events</option>
                        <option value="login">login</option>
                        <option value="logout">logout</option>
                        <option value="upload">upload</option>
                        <option value="query">query</option>
                        <option value="db_connect">db_connect</option>
                        <option value="db_disconnect">db_disconnect</option>
                      </select>
                    </div>
                    <p className="text-xs text-gray-500 flex-1">
                      {auditLoading ? "Loading…" : `${auditEvents.length} event${auditEvents.length !== 1 ? "s" : ""}`}
                    </p>
                    <button
                      onClick={exportAuditLogCsv}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-200 bg-[#1a1a24] border border-[#ffffff08] hover:border-[#ffffff15] transition-all"
                    >
                      <Download className="w-3 h-3" />
                      Export CSV
                    </button>
                    <button
                      onClick={() => loadAuditLog(auditEventTypeFilter)}
                      disabled={auditLoading}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-200 bg-[#1a1a24] border border-[#ffffff08] hover:border-[#ffffff15] transition-all disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3 h-3 ${auditLoading ? "animate-spin" : ""}`} />
                    </button>
                  </div>

                  {auditLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="w-6 h-6 text-[#667eea] animate-spin" />
                    </div>
                  ) : auditEvents.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                      <ClipboardList className="w-8 h-8 mb-3 opacity-40" />
                      <p className="text-sm">No events found</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-[#ffffff08]">
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-3">Time</th>
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-3">Event</th>
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2 pr-3">Detail</th>
                            <th className="text-left text-[11px] font-semibold text-gray-500 uppercase tracking-wider pb-2">Session</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#ffffff06]">
                          {auditEvents.map((ev) => {
                            const eventColors: Record<string, string> = {
                              login: "bg-[#10b981]/20 text-[#34d399]",
                              logout: "bg-[#6b7280]/20 text-[#9ca3af]",
                              upload: "bg-[#3b82f6]/20 text-[#60a5fa]",
                              query: "bg-[#8b5cf6]/20 text-[#a78bfa]",
                              db_connect: "bg-[#f59e0b]/20 text-[#fbbf24]",
                              db_disconnect: "bg-[#ef4444]/20 text-[#f87171]",
                            };
                            const colorClass = eventColors[ev.event_type] ?? "bg-[#ffffff10] text-gray-400";
                            return (
                              <tr key={ev.id} className="hover:bg-[#ffffff03] transition-colors">
                                <td className="py-2.5 pr-3 whitespace-nowrap">
                                  <span className="text-[11px] text-gray-500">
                                    {ev.occurred_at
                                      ? new Date(ev.occurred_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
                                      : "—"}
                                  </span>
                                </td>
                                <td className="py-2.5 pr-3">
                                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wide ${colorClass}`}>
                                    {ev.event_type}
                                  </span>
                                </td>
                                <td className="py-2.5 pr-3 max-w-[220px]">
                                  <p className="text-[11px] text-gray-400 truncate" title={ev.detail ?? ""}>
                                    {ev.detail || "—"}
                                  </p>
                                </td>
                                <td className="py-2.5">
                                  <span className="text-[10px] text-gray-600 font-mono">
                                    {ev.session_id ? ev.session_id.slice(0, 8) + "…" : "—"}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
