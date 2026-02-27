# 🤖 DocBot

DocBot is an AI-powered PDF assistant built with a modern, high-performance web architecture. It combines a premium Next.js frontend with a powerful Python FastAPI backend, specifically designed for seamless deployment on Vercel.

## Features

- **Advanced Document Processing**: Extract text, perform deep visual analysis (detecting checkboxes, forms, signatures), and understand embedded charts using Groq's Llama Vision models.
- **Expert Personas**: Switch between different specialized AI personas (Generalist, Doctor, Finance Expert, Engineer, Lawyer, Consultant) to get tailored responses.
- **Deep Research Mode**: Enable rigorous, multi-angle analysis with step-by-step reasoning for complex inquiries.
- **Premium UI**: An interactive, glassmorphism-inspired design with animated backgrounds and sleek chat interfaces.

## Architecture

- **Frontend**: React / Next.js (App Router), styled with TailwindCSS and Lucide React icons.
- **Backend**: Python FastAPI serving as Vercel Serverless Functions (`/api/*`).
- **AI & RAG**: LangChain, PyMuPDF (Fitz) for PDF extraction, FAISS for local vector storage, and Groq's high-speed inference (Llama 3.3 and Llama 3.2 Vision).

## Getting Started Locally

### Prerequisites

- Node.js (v18+)
- Python 3.12+ (Python 3.14 recommended, though `python3` must be used explicitly on macOS to avoid Xcode license prompts)
- A Groq API Key

### 1. Environment Setup

Create a `.env` file in the root directory and add your Groq API key:

```env
groq_api_key=your_api_key_here
```

### 2. Install Dependencies

**Frontend Dependencies:**
```bash
npm install
```

**Backend Dependencies:**
Create a virtual environment and install the required Python packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Development Servers

You will need to run both the Next.js frontend and the FastAPI backend concurrently.

**Terminal 1: Start the FastAPI Backend**
```bash
source .venv/bin/activate
# Note: On macOS, use `python3` instead of `python` to prevent Xcode popups
python3 -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Start the Next.js Frontend**
```bash
npm run dev -- --port 3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser. The Next.js app will automatically proxy `/api/*` requests to the FastAPI backend running on port 8000.

## Deployment to Vercel

This repository is pre-configured for deployment on Vercel. 

1. Push your code to a GitHub repository.
2. Import the project into Vercel.
3. Vercel will automatically detect the Next.js frontend and the `api/` directory containing the FastAPI backend.
4. Add your `groq_api_key` environment variable in the Vercel project settings.
5. Deploy! Vercel's Serverless Functions will automatically serve the Python endpoints alongside the React application.
