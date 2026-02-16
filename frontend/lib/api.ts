// API calls are proxied via Next.js rewrites to http://localhost:8000

export interface Persona {
    name: string;
    icon: string;
    description: string;
}

export interface UploadResponse {
    session_id: string;
    message: string;
    suggested_persona: string;
    num_chunks: number;
}

export interface ChatResponse {
    answer: string;
    session_id: string;
}

export async function getPersonas(): Promise<{ personas: Persona[] }> {
    const res = await fetch('/api/personas');
    if (!res.ok) throw new Error('Failed to fetch personas');
    return res.json();
}

export async function uploadPDF(
    file: File,
    deepVisualMode: boolean = false,
    sessionId?: string
): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('deep_visual_mode', String(deepVisualMode));
    if (sessionId) {
        formData.append('session_id', sessionId);
    }

    const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Upload failed');
    }

    return res.json();
}

export async function sendMessage(
    sessionId: string,
    message: string,
    persona: string = 'Generalist',
    deepResearch: boolean = false
): Promise<ChatResponse> {
    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId,
            message,
            persona,
            deep_research: deepResearch,
        }),
    });

    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Chat failed');
    }

    return res.json();
}

export async function clearChat(sessionId: string): Promise<void> {
    const res = await fetch(`/api/clear/${sessionId}`, {
        method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to clear chat');
}
