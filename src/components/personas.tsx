import React from "react"
import {
  Sparkles, Stethoscope, TrendingUp, Code, Cpu,
  Scale, Briefcase, BarChart2,
} from "lucide-react"

export interface PersonaConfig {
  icon: React.ReactNode
  description: string
  color: string
  gradient: string
  accent: string
  response_format: "finance" | "clinical" | "legal" | "technical" | "research" | "consulting" | "data" | "general"
  detection_keywords: { primary: string[]; secondary: string[] }
  tool_preference: "sql_first" | "rag_first" | "balanced"
  output_conventions: {
    number_format: "currency" | "percentage" | "raw"
    disclaimer_position: "header" | "footer" | null
    highlight_pattern: string | null
    accent_color: string
  }
}

export const EXPERT_PERSONAS: Record<string, PersonaConfig> = {
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
}

// DOCBOT-802: Client-side keyword router
export function routeQuestion(
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
