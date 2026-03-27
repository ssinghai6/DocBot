"use client"

import { EXPERT_PERSONAS, routeQuestion } from "@/components/personas"
import type {
  Message,
  AutopilotStep,
  ChartMeta,
  Toast,
} from "@/components/types"

interface UseChatSubmitParams {
  input: string
  sessionId: string | null
  isDbConnected: boolean
  connectionId: string | null
  selectedPersona: string
  isAutoMode: boolean
  chatMode: "docs" | "database" | "hybrid"
  autopilotMode: boolean
  isCsvConnection: boolean
  chartType: string
  deepResearch: boolean
  messages: Message[]

  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  setInput: React.Dispatch<React.SetStateAction<string>>
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>
  setChatMode: React.Dispatch<React.SetStateAction<"docs" | "database" | "hybrid">>
  setAutopilotRunning: React.Dispatch<React.SetStateAction<boolean>>
  setAutopilotSteps: React.Dispatch<React.SetStateAction<AutopilotStep[]>>
  setAutopilotPlan: React.Dispatch<React.SetStateAction<string[]>>
  setDrProgress: React.Dispatch<React.SetStateAction<{ step: string; message: string } | null>>

  showToast: (type: Toast['type'], message: string) => void
  loadQueryHistory: (connId: string) => void
}

export function useChatSubmit(params: UseChatSubmitParams) {
  const {
    input,
    sessionId,
    isDbConnected,
    connectionId,
    selectedPersona,
    isAutoMode,
    chatMode,
    autopilotMode,
    isCsvConnection,
    chartType,
    deepResearch,
    messages,

    setMessages,
    setInput,
    setIsLoading,
    setChatMode,
    setAutopilotRunning,
    setAutopilotSteps,
    setAutopilotPlan,
    setDrProgress,

    showToast,
    loadQueryHistory,
  } = params;

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim()) return;
    if (!sessionId && !isDbConnected) return;

    const userMsg: Message = { role: "user", content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setDrProgress(null);

    // Build recent conversation history (last 6 messages = 3 turns) for
    // follow-up context in DB, CSV, hybrid, and autopilot pipelines.
    const recentHistory = messages
      .filter(m => m.content.trim())
      .slice(-6)
      .map(m => ({ role: m.role, content: m.content.slice(0, 500) }));

    let personaToSend = selectedPersona;
    let effectiveChatMode = chatMode;
    if (sessionId && isDbConnected && chatMode !== "hybrid") {
      effectiveChatMode = "hybrid";
    }
    if (isAutoMode) {
      const routing = routeQuestion(input, effectiveChatMode, isDbConnected, !!sessionId, EXPERT_PERSONAS);
      if (routing.confidence !== "low") {
        personaToSend = routing.persona;
        const pref = EXPERT_PERSONAS[routing.persona as keyof typeof EXPERT_PERSONAS]?.tool_preference;
        if (pref === "sql_first" && isDbConnected && !sessionId) effectiveChatMode = "database";
        else if (pref === "rag_first" && sessionId && !isDbConnected) effectiveChatMode = "docs";
      } else {
        personaToSend = "Generalist";
      }
    }
    if (effectiveChatMode !== chatMode) {
      setChatMode(effectiveChatMode);
    }

    // ── Smart autopilot auto-trigger ────────────────────────────────────────
    // Auto-enable autopilot for analytical questions when a data source is loaded
    const AUTOPILOT_TRIGGER = /\b(predict|forecast|trend|seasonalit|correlat|regress|cluster|anomal|outli|compar|analys|analy[sz]|investigat|diagnos|root.?cause|deep.?dive|break.?down|distribut|what.?if|simulat|project|model|classif)\b/i;
    let useAutopilot = autopilotMode;
    if (!useAutopilot && (connectionId || sessionId) && AUTOPILOT_TRIGGER.test(input)) {
      useAutopilot = true;
      showToast("info", "Autopilot activated \u2014 multi-step analysis in progress");
    }

    // ── Autopilot path ─────────────────────────────────────────────────────
    if (useAutopilot && (connectionId || sessionId)) {
      setAutopilotRunning(true);
      setAutopilotSteps([]);
      setAutopilotPlan([]);
      const localSteps: AutopilotStep[] = [];
      try {
        const response = await fetch("/api/autopilot/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            connection_id: connectionId || "",
            question: userMsg.content,
            persona: personaToSend,
            session_id: sessionId ?? "anonymous",
            has_docs: !!sessionId,
            has_db: !!connectionId,
            has_csv: isCsvConnection,
            history: recentHistory,
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
            history: recentHistory,
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
              continue;
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
            history: recentHistory,
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
    if (!sessionId) {
      setIsLoading(false);
      showToast("error", "Please upload a PDF document before chatting.");
      return;
    }

    try {
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
              setDrProgress(null);
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

  return { handleSendMessage };
}
