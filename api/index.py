"""
DocBot FastAPI Backend for Vercel Deployment
Handles PDF processing, RAG, and chat endpoints.
"""

import os
import base64
import time
import tempfile
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from groq import Groq

import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# --- In-Memory Session Store (will reset on cold starts) ---
sessions: dict = {}

# --- Groq Client ---
groq_api_key = os.getenv("groq_api_key")
if not groq_api_key:
    raise ValueError("groq_api_key not found in environment variables")

client = Groq(api_key=groq_api_key)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=None,
    max_retries=2,
)

# --- Embeddings Model (loaded once) ---
embeddings_model = None


def get_embeddings():
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    return embeddings_model


# --- Expert Personas ---
EXPERT_PERSONAS = {
    "Generalist": {"icon": "🎯", "description": "Balanced, general-purpose assistant", "persona_def": "You are a knowledgeable assistant."},
    "Doctor": {"icon": "🩺", "description": "Medical & healthcare perspective", "persona_def": "You are a Medical Doctor with extensive clinical experience."},
    "Finance Expert": {"icon": "💰", "description": "Financial & investment analysis", "persona_def": "You are a Senior Finance Expert with deep knowledge in investment, banking, and financial analysis."},
    "Engineer": {"icon": "⚙️", "description": "Technical & engineering focus", "persona_def": "You are a Senior Engineer with expertise in systems design, problem-solving, and technical implementation."},
    "AI/ML Expert": {"icon": "🤖", "description": "AI, ML & data science insights", "persona_def": "You are an AI/ML Expert with deep knowledge in machine learning."},
    "Lawyer": {"icon": "⚖️", "description": "Legal analysis & compliance", "persona_def": "You are a Senior Lawyer with expertise in legal analysis."},
    "Consultant": {"icon": "📊", "description": "Strategic business advisory", "persona_def": "You are a Senior Consultant with extensive experience in strategy."},
}


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str
    persona: str = "Generalist"
    deep_research: bool = False


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    message: str
    suggested_persona: str
    num_chunks: int


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_embeddings()
    yield


app = FastAPI(title="DocBot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_pdf(file_bytes: bytes, filename: str, deep_visual_mode: bool = False) -> list[Document]:
    """Process a PDF file and return LangChain Documents."""
    all_docs = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)

        for page_num, page in enumerate(doc):
            text = page.get_text()
            image_descriptions = []

            # Extract embedded images
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                    for attempt in range(3):
                        try:
                            time.sleep(2.5)
                            chat_completion = client.chat.completions.create(
                                messages=[{"role": "user", "content": [
                                    {"type": "text", "text": "Analyze this image in detail."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                                ]}],
                                model="meta-llama/llama-4-scout-17b-16e-instruct",
                            )
                            description = chat_completion.choices[0].message.content
                            if description:
                                image_descriptions.append(f"[IMAGE (Page {page_num + 1}, Image {img_index + 1})]: {description}")
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < 2:
                                time.sleep(5)
                                continue
                            raise
                except Exception as e:
                    print(f"Error processing image: {e}")

            page_content = text + "\n\n" + "\n\n".join(image_descriptions)
            all_docs.append(Document(page_content=page_content, metadata={"source": filename, "page": page_num}))

        doc.close()
    finally:
        os.unlink(tmp_path)

    return all_docs


def analyze_document_persona(db) -> str:
    """Analyzes the document to suggest the best expert persona."""
    try:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke("Overview of the document content and purpose")
        context_text = "\n\n".join([d.page_content for d in docs])[:3000]

        prompt = f"Analyze this document excerpt and return ONLY one of: Doctor, Finance Expert, Engineer, AI/ML Expert, Lawyer, Consultant, Generalist\n\nExcerpt:\n{context_text}"
        response = llm.invoke(prompt)
        suggestion = response.content.strip()

        for persona in EXPERT_PERSONAS.keys():
            if persona in suggestion:
                return persona
        return "Generalist"
    except Exception:
        return "Generalist"


def build_system_prompt(persona: str, deep_research: bool) -> str:
    persona_def = EXPERT_PERSONAS[persona].get("persona_def", "You are a knowledgeable assistant.")
    return f"""{persona_def}

Answer based STRICTLY on the provided context. If the answer is not in the context, say so.

CONTEXT:
<context>
{{context}}
</context>
"""


# --- API Endpoints ---
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/personas")
async def get_personas():
    return {"personas": [{"name": k, "icon": v["icon"], "description": v["description"]} for k, v in EXPERT_PERSONAS.items()]}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    deep_visual_mode: bool = Form(False),
    session_id: Optional[str] = Form(None),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        file_bytes = await file.read()
        docs = process_pdf(file_bytes, file.filename, deep_visual_mode)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embeddings = get_embeddings()
        db = FAISS.from_documents(splits, embeddings)

        suggested_persona = analyze_document_persona(db)

        sessions[session_id] = {"db": db, "messages": [], "suggested_persona": suggested_persona}

        return UploadResponse(
            session_id=session_id,
            message=f"Successfully processed {file.filename}",
            suggested_persona=suggested_persona,
            num_chunks=len(splits),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    db = session["db"]
    messages = session["messages"]

    try:
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformulate the question if needed based on chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, db.as_retriever(), contextualize_q_prompt)

        system_prompt = build_system_prompt(request.persona, request.deep_research)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in messages
        ]

        response = retrieval_chain.invoke({"input": request.message, "chat_history": chat_history})
        answer = response["answer"]

        messages.append({"role": "user", "content": request.message})
        messages.append({"role": "assistant", "content": answer})

        return ChatResponse(answer=answer, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id]["messages"] = []
        return {"message": "Chat history cleared"}
    raise HTTPException(status_code=404, detail="Session not found")
