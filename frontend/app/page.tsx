'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { uploadPDF, sendMessage, clearChat, Persona, UploadResponse } from '@/lib/api';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

const PERSONAS: Record<string, { icon: string; description: string }> = {
    Generalist: { icon: '🎯', description: 'Balanced, general-purpose assistant' },
    Doctor: { icon: '🩺', description: 'Medical & healthcare perspective' },
    'Finance Expert': { icon: '💰', description: 'Financial & investment analysis' },
    Engineer: { icon: '⚙️', description: 'Technical & engineering focus' },
    'AI/ML Expert': { icon: '🤖', description: 'AI, ML & data science insights' },
    Lawyer: { icon: '⚖️', description: 'Legal analysis & compliance' },
    Consultant: { icon: '📊', description: 'Strategic business advisory' },
};

export default function Home() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [selectedPersona, setSelectedPersona] = useState('Generalist');
    const [suggestedPersona, setSuggestedPersona] = useState<string | null>(null);
    const [deepResearch, setDeepResearch] = useState(false);
    const [deepVisualMode, setDeepVisualMode] = useState(false);
    const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        setError(null);

        try {
            const response = await uploadPDF(file, deepVisualMode, sessionId || undefined);
            setSessionId(response.session_id);
            setUploadedFileName(file.name);
            setSuggestedPersona(response.suggested_persona);
            setSelectedPersona(response.suggested_persona);
            setMessages([]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setIsUploading(false);
        }
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!input.trim() || !sessionId || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);
        setError(null);

        try {
            const response = await sendMessage(sessionId, userMessage, selectedPersona, deepResearch);
            setMessages((prev) => [...prev, { role: 'assistant', content: response.answer }]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to send message');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClearChat = async () => {
        if (!sessionId) return;
        try {
            await clearChat(sessionId);
            setMessages([]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to clear chat');
        }
    };

    return (
        <div className="flex h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900">
            {/* Sidebar */}
            <aside className="w-72 bg-slate-900/80 backdrop-blur-xl border-r border-white/10 p-6 flex flex-col">
                <div className="mb-8">
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                        🤖 DocBot
                    </h1>
                    <p className="text-sm text-slate-400 mt-1">AI-Powered PDF Assistant</p>
                </div>

                {/* File Upload */}
                <div className="mb-6">
                    <h3 className="text-sm font-medium text-slate-300 mb-3">📄 Document Upload</h3>
                    <label className="flex items-center gap-2 mb-3">
                        <input
                            type="checkbox"
                            checked={deepVisualMode}
                            onChange={(e) => setDeepVisualMode(e.target.checked)}
                            className="rounded border-slate-600 bg-slate-800 text-indigo-500"
                        />
                        <span className="text-sm text-slate-400">🔍 Deep Visual Analysis</span>
                    </label>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf"
                        onChange={handleFileUpload}
                        className="hidden"
                    />
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="w-full py-3 px-4 bg-slate-800 hover:bg-slate-700 border-2 border-dashed border-indigo-500/30 hover:border-indigo-500/60 rounded-xl text-slate-300 transition-all"
                    >
                        {isUploading ? '⏳ Processing...' : uploadedFileName || '📁 Choose PDF'}
                    </button>
                </div>

                {/* Persona Selector */}
                {sessionId && (
                    <div className="mb-6">
                        <h3 className="text-sm font-medium text-slate-300 mb-3">🎭 Expert Mode</h3>
                        {suggestedPersona && suggestedPersona !== 'Generalist' && (
                            <p className="text-xs text-indigo-400 mb-2">
                                💡 Recommended: {suggestedPersona}
                            </p>
                        )}
                        <select
                            value={selectedPersona}
                            onChange={(e) => setSelectedPersona(e.target.value)}
                            className="w-full py-2 px-3 bg-slate-800 border border-white/10 rounded-lg text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        >
                            {Object.entries(PERSONAS).map(([name, { icon }]) => (
                                <option key={name} value={name}>
                                    {icon} {name}
                                </option>
                            ))}
                        </select>
                        <p className="text-xs text-slate-500 mt-2">
                            {PERSONAS[selectedPersona]?.description}
                        </p>

                        {/* Deep Research Toggle */}
                        {selectedPersona !== 'Generalist' && (
                            <label className="flex items-center gap-2 mt-4">
                                <input
                                    type="checkbox"
                                    checked={deepResearch}
                                    onChange={(e) => setDeepResearch(e.target.checked)}
                                    className="rounded border-slate-600 bg-slate-800 text-indigo-500"
                                />
                                <span className="text-sm text-slate-400">🔬 Deep Research</span>
                            </label>
                        )}
                    </div>
                )}

                <div className="flex-1" />

                {/* Clear Chat Button */}
                {sessionId && messages.length > 0 && (
                    <button
                        onClick={handleClearChat}
                        className="w-full py-2 px-4 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 rounded-lg text-red-400 transition-all"
                    >
                        🗑️ Clear Chat
                    </button>
                )}

                <div className="mt-6 pt-6 border-t border-white/10">
                    <p className="text-xs text-slate-500 text-center">
                        Developed by{' '}
                        <a href="https://sanshrit-singhai.vercel.app" className="text-indigo-400 hover:underline">
                            Sanshrit Singhai
                        </a>
                    </p>
                </div>
            </aside>

            {/* Main Chat Area */}
            <main className="flex-1 flex flex-col">
                {/* Header */}
                <header className="p-4 border-b border-white/10 bg-slate-900/50 backdrop-blur">
                    <div className="flex items-center gap-3">
                        <span className="text-2xl">{PERSONAS[selectedPersona]?.icon}</span>
                        <div>
                            <h2 className="font-semibold text-white">{selectedPersona} Mode</h2>
                            <p className="text-xs text-slate-400">
                                {uploadedFileName ? `📄 ${uploadedFileName}` : 'No document loaded'}
                            </p>
                        </div>
                    </div>
                </header>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {!sessionId ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="text-center">
                                <div className="text-6xl mb-4">📚</div>
                                <h2 className="text-xl font-semibold text-white mb-2">Welcome to DocBot</h2>
                                <p className="text-slate-400">Upload a PDF to get started</p>
                            </div>
                        </div>
                    ) : messages.length === 0 ? (
                        <div className="flex items-center justify-center h-full">
                            <div className="text-center">
                                <div className="text-6xl mb-4">💬</div>
                                <h2 className="text-xl font-semibold text-white mb-2">Document Ready!</h2>
                                <p className="text-slate-400">Ask any question about your document</p>
                            </div>
                        </div>
                    ) : (
                        messages.map((msg, i) => (
                            <div
                                key={i}
                                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[80%] p-4 rounded-2xl ${msg.role === 'user'
                                            ? 'bg-indigo-600 text-white'
                                            : 'bg-slate-800/60 backdrop-blur border border-white/10 text-slate-200'
                                        }`}
                                >
                                    <p className="whitespace-pre-wrap">{msg.content}</p>
                                </div>
                            </div>
                        ))
                    )}

                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-slate-800/60 backdrop-blur border border-white/10 p-4 rounded-2xl">
                                <div className="flex gap-1">
                                    <span className="animate-bounce">●</span>
                                    <span className="animate-bounce delay-100">●</span>
                                    <span className="animate-bounce delay-200">●</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="flex justify-center">
                            <div className="bg-red-500/20 border border-red-500/30 p-4 rounded-xl text-red-400">
                                ⚠️ {error}
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <form onSubmit={handleSubmit} className="p-4 border-t border-white/10 bg-slate-900/50 backdrop-blur">
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={sessionId ? 'Ask a question about the document...' : 'Upload a PDF first'}
                            disabled={!sessionId || isLoading}
                            className="flex-1 py-3 px-4 bg-slate-800 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                        />
                        <button
                            type="submit"
                            disabled={!sessionId || isLoading || !input.trim()}
                            className="py-3 px-6 bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 rounded-xl text-white font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Send
                        </button>
                    </div>
                </form>
            </main>
        </div>
    );
}
