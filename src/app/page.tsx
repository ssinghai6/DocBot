"use client"

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Upload, File, Loader2, Trash2 } from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const EXPERT_PERSONAS: Record<string, { icon: string; description: string }> = {
  Generalist: { icon: "🎯", description: "Balanced, general-purpose assistant" },
  Doctor: { icon: "🩺", description: "Medical & healthcare perspective" },
  "Finance Expert": { icon: "💰", description: "Financial & investment analysis" },
  Engineer: { icon: "⚙️", description: "Technical & engineering focus" },
  "AI/ML Expert": { icon: "🤖", description: "AI, ML & data science insights" },
  Lawyer: { icon: "⚖️", description: "Legal analysis & compliance" },
  Consultant: { icon: "📊", description: "Strategic business advisory" },
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Sidebar State
  const [selectedPersona, setSelectedPersona] = useState("Generalist");
  const [suggestedPersona, setSuggestedPersona] = useState<string | null>(null);
  const [deepVisualMode, setDeepVisualMode] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    const files = Array.from(e.target.files);

    const MAX_FILE_SIZE = 4.5 * 1024 * 1024; // 4.5 MB max for Vercel
    const totalSize = files.reduce((acc, file) => acc + file.size, 0);

    if (totalSize > MAX_FILE_SIZE) {
      alert("Total file size exceeds 4.5MB limit. Please upload smaller documents to bypass Vercel serverless limitations.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setUploadedFiles(files);
    setIsLoading(true);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('deep_visual_mode', String(deepVisualMode));

    try {
      // In Vercel, this will hit our Python backend due to rewrites in vercel.json
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to upload");
      }

      const data = await response.json();
      setSessionId(data.session_id);

      if (data.suggested_persona) {
        setSuggestedPersona(data.suggested_persona);
        setSelectedPersona(data.suggested_persona);
      }

      alert("Documents processed successfully!");
    } catch (error: any) {
      console.error("Upload error:", error);
      alert(`Error uploading documents: ${error.message}`);
      setUploadedFiles([]);
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || !sessionId) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMsg.content,
          history: messages,
          persona: selectedPersona,
          deep_research: deepResearch
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to send message");
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.content }]);
    } catch (error: any) {
      console.error("Chat error:", error);
      alert(`Error generating response: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex h-screen relative z-10 text-[#e0e0e0] overflow-hidden">
      {/* Settings / Options Sidebar */}
      <aside className="w-80 backdrop-blur-xl bg-[#1a1a2e]/95 border-r border-[#ffffff1a] flex flex-col p-6 z-20 shrink-0 shadow-2xl overflow-y-auto">
        <h2 className="text-2xl font-bold mb-6 text-white tracking-wide">Options</h2>

        {/* Document Upload Section */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-3 text-white">📄 Document Upload</h3>
          <div className="mb-4">
            <label className="flex items-center space-x-2 cursor-pointer group">
              <div className={`w-10 h-6 flex items-center bg-gray-600 rounded-full p-1 transition-colors ${deepVisualMode ? 'bg-[#667eea]' : ''}`}>
                <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepVisualMode ? 'translate-x-4' : ''}`}></div>
              </div>
              <span className="text-sm font-medium group-hover:text-white transition-colors">🔍 Deep Visual Analysis</span>
              <input type="checkbox" className="hidden" checked={deepVisualMode} onChange={() => setDeepVisualMode(!deepVisualMode)} />
            </label>
            <p className="text-xs text-gray-400 mt-2 italic">*Full page analysis enabled - will detect tick marks, checkboxes, and form selections*</p>
          </div>

          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-[#667eea]/40 rounded-2xl p-6 text-center cursor-pointer hover:bg-[#1e1e2e]/70 hover:border-[#667eea]/80 hover:shadow-[0_0_20px_rgba(102,126,234,0.15)] transition-all bg-[#1e1e2e]/50"
          >
            <Upload className="mx-auto h-8 w-8 text-[#667eea] mb-2" />
            <p className="text-sm font-medium">Choose PDF file(s)</p>
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="application/pdf"
              multiple
              onChange={handleFileUpload}
            />
          </div>
          {uploadedFiles.length > 0 && (
            <div className="mt-3 space-y-2">
              {uploadedFiles.map((f, i) => (
                <div key={i} className="flex items-center text-xs text-green-400 bg-green-400/10 px-3 py-2 rounded-lg border border-green-400/20">
                  <File className="w-4 h-4 mr-2" />
                  <span className="truncate flex-1">{f.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Persona Selection */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-3 text-white">🎭 Expert Mode</h3>
          {suggestedPersona && suggestedPersona !== "Generalist" && (
            <div className="bg-[#667eea]/15 border border-[#667eea]/30 p-3 rounded-xl mb-3 text-sm flex items-start">
              <span className="mr-2">💡</span>
              <span>based on your document, <strong>{suggestedPersona}</strong> mode is recommended.</span>
            </div>
          )}

          <div className="relative">
            <select
              value={selectedPersona}
              onChange={(e) => setSelectedPersona(e.target.value)}
              className="w-full appearance-none bg-[#1e1e2e]/80 border border-[#ffffff26] rounded-xl px-4 py-3 text-white pr-8 focus:outline-none focus:border-[#667eea]/50 focus:ring-1 focus:ring-[#667eea]/50 transition-all cursor-pointer"
            >
              {Object.entries(EXPERT_PERSONAS).map(([name, data]) => (
                <option key={name} value={name}>{data.icon} {name}</option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-white">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
            </div>
          </div>
          <p className="text-xs text-gray-400 mt-2 italic">*{EXPERT_PERSONAS[selectedPersona]?.description}*</p>
        </div>

        {/* Deep Research */}
        {selectedPersona !== "Generalist" && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-3 text-white">🔬 Analysis Mode</h3>
            <label className="flex items-center space-x-2 cursor-pointer group">
              <div className={`w-10 h-6 flex items-center bg-gray-600 rounded-full p-1 transition-colors ${deepResearch ? 'bg-[#764ba2]' : ''}`}>
                <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${deepResearch ? 'translate-x-4' : ''}`}></div>
              </div>
              <span className="text-sm font-medium group-hover:text-white transition-colors">Deep Research</span>
              <input type="checkbox" className="hidden" checked={deepResearch} onChange={() => setDeepResearch(!deepResearch)} />
            </label>
            {deepResearch && (
              <p className="text-xs text-gray-400 mt-2 italic">*🧠 Deep reasoning enabled - responses will be more thorough*</p>
            )}
          </div>
        )}

        <div className="mt-auto pt-6 border-t border-[#ffffff1a]">
          <button
            onClick={clearChat}
            className="w-full flex items-center justify-center py-3 px-4 rounded-xl bg-[#1e1e2e]/80 border border-[#ffffff26] hover:bg-red-500/20 hover:border-red-500/50 hover:text-red-400 transition-all font-medium"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Clear Chat
          </button>

          <div className="mt-8 text-center text-xs text-gray-500">
            <p className="mb-2">Developed by <a href="https://sanshrit-singhai.vercel.app" className="text-[#00C4FF] hover:underline" target="_blank" rel="noopener noreferrer">Sanshrit Singhai</a></p>
            <p>Support development:</p>
            <a href="https://www.paypal.com/donate/?business=G5C3WRTY7YTXC&no_recurring=0&currency_code=USD" target="_blank" rel="noopener noreferrer" className="inline-block mt-2">
              <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" alt="Donate with PayPal" className="opacity-80 hover:opacity-100 transition-opacity" />
            </a>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-transparent relative z-10">

        {/* Header */}
        <header className="px-8 py-6 flex-none">
          <div className="relative overflow-hidden bg-gradient-to-br from-[#1e1e2e]/90 to-[#16213e]/90 rounded-3xl border border-[#667eea]/30 shadow-[0_10px_40px_rgba(0,0,0,0.3)] p-8 text-center">
            <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-[#667eea]/10 to-transparent animate-shimmer"></div>
            <h1 className="relative z-10 text-5xl font-bold mb-3 text-transparent bg-clip-text bg-gradient-to-br from-[#667eea] via-[#764ba2] to-[#667eea] animate-gradient-text">
              🤖 DocBot
            </h1>
            <p className="relative z-10 text-gray-300 text-lg">
              Your AI-Powered PDF Assistant • Powered by <span className="text-[#667eea] font-semibold">Llama 3.3</span>
            </p>

            <div className="relative z-10 flex justify-center gap-4 mt-6 flex-wrap">
              <span className="bg-[#667eea]/15 px-4 py-1.5 rounded-full text-sm text-gray-300 border border-[#667eea]/30 backdrop-blur-sm">📊 Charts Analysis</span>
              <span className="bg-[#764ba2]/15 px-4 py-1.5 rounded-full text-sm text-gray-300 border border-[#764ba2]/30 backdrop-blur-sm">✅ Form Detection</span>
              <span className="bg-[#48bb78]/15 px-4 py-1.5 rounded-full text-sm text-gray-300 border border-[#48bb78]/30 backdrop-blur-sm">🎭 Expert Modes</span>
            </div>
          </div>
        </header>

        {/* Chat Output */}
        <div className="flex-1 overflow-y-auto px-8 pb-4">
          {!sessionId && messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400 opacity-60">
              <Upload className="w-16 h-16 mb-4" />
              <p className="text-xl">Upload PDF documents to begin chatting</p>
            </div>
          ) : (
            <div className="space-y-6 max-w-4xl mx-auto pb-20">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl p-5 animate-message-slide-in shadow-lg border backdrop-blur-xl
                      ${msg.role === "user"
                        ? "bg-gradient-to-br from-[#667eea] to-[#764ba2] text-white border-transparent rounded-tr-sm"
                        : "bg-[#1e1e2e]/80 border-[#ffffff1a] text-[#e0e0e0] rounded-tl-sm hover:border-[#667eea]/30 hover:shadow-[0_8px_32px_rgba(102,126,234,0.15)] transition-all"
                      }`}
                  >
                    {msg.role === "assistant" ? (
                      <div className="prose prose-invert max-w-none text-sm leading-relaxed">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && messages.length > 0 && messages[messages.length - 1].role === "user" && (
                <div className="flex justify-start">
                  <div className="bg-[#1e1e2e]/80 border border-[#ffffff1a] rounded-2xl rounded-tl-sm p-4 animate-pulse flex items-center text-gray-400">
                    <Loader2 className="w-5 h-5 animate-spin mr-3 text-[#667eea]" />
                    Analyzing document and generating response...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-8 pt-0 flex-none max-w-5xl mx-auto w-full">
          <form
            onSubmit={handleSendMessage}
            className="relative bg-[#1e1e2e]/80 border border-[#ffffff26] rounded-2xl backdrop-blur-xl shadow-[0_4px_20px_rgba(0,0,0,0.2)] focus-within:border-[#667eea]/50 focus-within:shadow-[0_4px_20px_rgba(102,126,234,0.2)] transition-all overflow-hidden flex items-end"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder={sessionId ? "Ask a question about the document..." : "Please upload a document before asking..."}
              disabled={!sessionId || isLoading}
              className="flex-1 max-h-48 min-h-[60px] p-4 bg-transparent outline-none resize-none text-white placeholder-gray-400 text-[15px]"
              rows={1}
            />
            <button
              type="submit"
              disabled={!input.trim() || !sessionId || isLoading}
              className="m-3 p-3 rounded-xl bg-gradient-to-tr from-[#667eea] to-[#764ba2] text-white disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all focus:outline-none focus:ring-2 focus:ring-[#764ba2]/50 hover:scale-105 active:scale-95"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <div className="text-center mt-3 text-xs text-gray-500 font-medium">
            DocBot can make mistakes. Consider verifying important information.
          </div>
        </div>

      </main>
    </div>
  );
}
