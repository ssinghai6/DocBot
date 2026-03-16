# DocBot

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-14+-black.svg" alt="Next.js">
</p>

DocBot is an enterprise-grade AI-powered PDF assistant that combines advanced document processing with specialized AI personas to deliver accurate, domain-specific responses. Built with a modern high-performance architecture featuring Next.js and FastAPI, optimized for seamless deployment on Vercel.

## Why DocBot

Extracting insights from documents shouldn't require manual searching or expensive enterprise solutions. DocBot uses Retrieval-Augmented Generation (RAG) to understand your documents and provide answers grounded in your actual content—with citations, in your preferred domain context.

**Problem**: Traditional PDF tools only search text; they don't understand context or synthesize information across pages.

**Solution**: DocBot combines vision AI for visual document analysis with semantic search and specialized AI personas to answer complex questions with source-backed citations.

## Quick Start

Get DocBot running in under 5 minutes:

```bash
# 1. Clone and install
git clone https://github.com/yourusername/docbot.git
cd docbot
npm install

# 2. Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
echo "groq_api_key=your_groq_api_key" > .env

# 4. Run both services
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to start chatting with your documents.

> **Note**: Get your free Groq API key at [console.groq.com](https://console.groq.com)

## Features

### Expert Persona System

Choose from 7 specialized AI personas, each optimized for different domains:

| Persona | Use Case | Disclaimer |
|---------|----------|------------|
| **Generalist** | Broad questions, general summarization | None |
| **Doctor** | Medical documents, research papers, clinical notes | Provides information only; not medical advice |
| **Finance Expert** | Financial reports, statements, investment documents | Not financial advice; verify with professionals |
| **Engineer** | Technical specifications, engineering documents | Technical reference only |
| **AI/ML Expert** | AI/ML research papers, technical documentation | Technical reference only |
| **Lawyer** | Legal documents, contracts, case law | Not legal advice; consult attorneys |
| **Consultant** | Business documents, strategy papers, proposals | Strategic reference only |

Each persona includes:
- Domain-specific system prompts and response guidelines
- Tailored retrieval strategies optimized for their domain
- Appropriate disclaimers for regulated domains (medical/legal/finance)

### Multi-Document Analysis

Upload multiple PDFs in a single session. DocBot processes all documents together, enabling cross-document queries and comprehensive analysis across your entire document library.

### Citation System

Every response includes verifiable citations:
- Source filename
- Page number(s)
- Relevant text snippet

Citations are clickable and take you directly to the source context.

### Session History & Management

- **Persistent storage**: SQLite database stores all conversation history
- **Session management**: List, view, and delete past sessions via API
- **Full context**: Resume conversations with complete message history

### Export Functionality

Export any session in your preferred format:
- **TXT**: Plain text for easy reading
- **Markdown**: Formatted with headers and code blocks
- **JSON**: Structured data for programmatic use

### Production-Ready UI

A modern, dark-themed interface built for professional use:

- **Purple/blue gradient** aesthetic with glassmorphism effects
- **Card-based persona selector** with visual indicators for each domain
- **Drag-and-drop file upload** with real-time progress states
- **Toast notifications** for feedback (no browser alerts)
- **Keyboard shortcuts**: `Ctrl+Enter` or `Cmd+Enter` to send messages
- **Responsive design**: Works seamlessly on desktop and mobile

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend                        │
│  (React, TailwindCSS, Lucide Icons, Framer Motion)         │
└─────────────────────────┬───────────────────────────────────┘
                          │ API Routes (/api/*)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   LangChain │  │   PyMuPDF   │  │   SQLite + FAISS    │  │
│  │   (RAG)     │  │  (PDF OCR)  │  │  (Storage + Search) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    Groq     │
                   │  (Llama 3)  │
                   └─────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14 (App Router), React, TailwindCSS |
| Backend | Python FastAPI |
| AI Inference | Groq (Llama 3.3, Llama 3.2 Vision) |
| RAG Pipeline | LangChain, FAISS |
| Document Processing | PyMuPDF (fitz) |
| Storage | SQLite |
| Embeddings | sentence-transformers |

### RAG Configuration

Optimized settings for accurate, fast retrieval:

- **Chunk size**: 1500 characters
- **Retrieval**: Top 8 chunks
- **Score threshold**: 0.3 (filters low-quality matches)
- **Cached embeddings**: Enabled for repeated documents

## API Reference

### Document Upload

```bash
POST /api/upload
Content-Type: multipart/form-data

# Request: Upload one or more PDF files
# Response: { "documents": [...], "message": "..." }
```

### Chat

```bash
GET /api/chat?message=Your question&persona=doctor

# Response:
{
  "response": "AI answer with citations...",
  "sources": [
    {
      "filename": "document.pdf",
      "page": 3,
      "text": "Relevant snippet..."
    }
  ],
  "processing_time_ms": 1250
}
```

### Personas

```bash
GET /api/personas

# Response: List of all personas with descriptions and capabilities
```

### Sessions

```bash
# List all sessions
GET /api/sessions

# Get specific session
GET /api/session/{id}

# Delete session
DELETE /api/session/{id}
```

### Export

```bash
GET /api/export/{id}?format=markdown

# Supported formats: txt, markdown, json
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `groq_api_key` | Yes | Your Groq API key (get one free at console.groq.com) |

## Deployment

### Vercel (Recommended)

This repository is pre-configured for Vercel deployment:

1. Push your code to a GitHub repository
2. Import the project into Vercel
3. Vercel automatically detects the Next.js frontend and FastAPI backend
4. Add `groq_api_key` in Vercel project settings
5. Deploy

The FastAPI backend runs as Vercel Serverless Functions in the `/api` directory.

### Local Development

```bash
# Terminal 1: Backend
source .venv/bin/activate
python3 -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
npm run dev -- --port 3000
```

The Next.js app proxies `/api/*` requests to the backend on port 8000.

## Requirements

### Prerequisites

- Node.js 18+
- Python 3.12+
- Groq API key (free at console.groq.com)

### Python Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- fastapi
- uvicorn
- langchain
- langchain-groq
- faiss-cpu
- pymupdf
- sentence-transformers
- python-multipart
- aiosqlite

## License

MIT © [Sanshrit Singhai](https://github.com/ssinghai6)

## Contributing

Contributions are welcome! Please open an issue or submit a PR for any improvements.
