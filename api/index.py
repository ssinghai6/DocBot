from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import asyncio
import logging
import uuid

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="DocBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = []
    persona: str = "Generalist"
    deep_research: bool = False

VECTOR_STORES = {}

EXPERT_PERSONAS = {
    "Generalist": {"persona_def": "You are a knowledgeable assistant."},
    "Doctor": {"persona_def": "You are a Medical Doctor with extensive clinical experience."},
    "Finance Expert": {"persona_def": "You are a Senior Finance Expert with deep knowledge in investment."},
    "Engineer": {"persona_def": "You are a Senior Engineer with expertise in systems design."},
    "AI/ML Expert": {"persona_def": "You are an AI/ML Expert with deep knowledge in machine learning."},
    "Lawyer": {"persona_def": "You are a Senior Lawyer with expertise in legal analysis."},
    "Consultant": {"persona_def": "You are a Senior Consultant with extensive experience in strategy."},
}

@app.get("/api/health")
def health_check():
    return {"status": "DocBot API is running"}

@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    deep_visual_mode: bool = Form(False)
):
    session_id = str(uuid.uuid4())
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_core.documents import Document
        
        all_content = []
        
        for file in files:
            content = await file.read()
            temp_path = f"/tmp/{file.filename}"
            
            with open(temp_path, "wb") as f:
                f.write(content)
            
            try:
                import fitz
                doc = fitz.open(temp_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        all_content.append(Document(
                            page_content=text,
                            metadata={"source": file.filename, "page": page_num}
                        ))
                    doc.close()
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                continue
        
        if not all_content:
            raise HTTPException(status_code=400, detail="No text found in documents")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_content)
        
        hf_token = os.getenv('huggingface_api_key') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
        
        db = InMemoryVectorStore.from_documents(splits, embeddings)
        VECTOR_STORES[session_id] = db
        
        return {
            "session_id": session_id,
            "message": "Documents processed successfully",
            "suggested_persona": "Generalist"
        }
        
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.session_id not in VECTOR_STORES:
        raise HTTPException(status_code=404, detail="Session not found. Please upload documents again.")
    
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain, create_history_aware_retriever
        
        groq_api_key = os.getenv('groq_api_key')
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="Groq API key not configured")
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0,
        )
        
        db = VECTOR_STORES[request.session_id]
        
        chat_history = []
        for msg in request.history:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            else:
                chat_history.append(AIMessage(content=msg.content))
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, db.as_retriever(), contextualize_q_prompt
        )
        
        persona_def = EXPERT_PERSONAS.get(request.persona, EXPERT_PERSONAS["Generalist"])["persona_def"]
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{persona_def}\n\nAnswer based ONLY on the provided context."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        response = retrieval_chain.invoke({
            "input": request.message,
            "chat_history": chat_history
        })
        
        return {"role": "assistant", "content": response["answer"]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
