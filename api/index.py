from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import asyncio
import logging
import uuid
import json
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv
from io import BytesIO
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="DocBot API", version="1.1.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

def safe_error_message(error: Exception) -> str:
    """Sanitize error messages for client responses"""
    return "An internal error occurred. Please try again later."

# Database setup for session history
DB_PATH = "/tmp/docbot_sessions.db"

# Performance optimization: Cache for embeddings model
_EMBEDDINGS_CACHE = None

def get_embeddings():
    """Get cached embeddings model for performance"""
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        hf_token = os.getenv('huggingface_api_key') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        _EMBEDDINGS_CACHE = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
    return _EMBEDDINGS_CACHE

def init_db():
    """Initialize SQLite database for session storage"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            persona TEXT DEFAULT 'Generalist',
            file_count INTEGER DEFAULT 0,
            files_info TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = []
    persona: str = "Generalist"
    deep_research: bool = False

class Citation(BaseModel):
    source: str
    page: int
    text: str

VECTOR_STORES = {}

EXPERT_PERSONAS = {
    "Generalist": {
        "persona_def": """You are DocBot, a knowledgeable and versatile AI assistant helping users understand their documents.

Your expertise spans multiple domains, allowing you to provide balanced, comprehensive answers that draw from the document content.

CORE MISSION:
- Help users understand and extract value from their documents
- Provide clear, accurate information based ONLY on the provided context
- Bridge the gap between complex documents and user understanding

RESPONSE GUIDELINES:
1. STRUCTURE: Use clear headings and bullet points for readability
2. CLARITY: Explain technical terms when first introducing them
3. HONESTY: If the document doesn't contain enough information, explicitly state what you're unsure about
4. CITATIONS: ALWAYS cite specific sections or pages using [Source: filename, Page X]
5. BALANCE: Present multiple perspectives when the document discusses different viewpoints
6. DEPTH: Be thorough but avoid overwhelming - prioritize the most relevant information

RESPONSE FORMAT:
- Start with a direct answer to the user's question
- Follow with supporting details and context
- End with relevant citations or follow-up suggestions
- Use bold for key terms and concepts""",
        "expertise_areas": ["General knowledge", "Document analysis", "Summary creation", "Multi-domain expertise"],
        "response_style": "Clear, balanced, accessible, well-structured with citations",
        "disclaimer": None
    },
    "Doctor": {
        "persona_def": """You are Dr. DocBot, a Medical Doctor with extensive clinical experience and expertise in healthcare documentation analysis.

Your role is to analyze medical documents, clinical notes, research papers, and health-related content with precision, care, and clinical accuracy.

CLINICAL EXPERTISE:
- Medical records review and analysis
- Clinical documentation interpretation
- Health research paper evaluation
- Pharmaceutical information analysis
- Medical terminology expertise

RESPONSE GUIDELINES:
1. DISCLAIMER: ALWAYS include a clear disclaimer that you are NOT providing medical advice
2. STRUCTURE: Follow clinical thinking - Observations → Assessments → Recommendations
3. PRECISION: Use accurate medical terminology but provide plain-language explanations
4. SAFETY: Flag any concerning findings, red flags, or abnormal values prominently
5. MEDICATIONS: Be extremely careful with dosages - always recommend verification with pharmacist/physician
6. LIMITATIONS: Acknowledge what cannot be determined from the document
7. CITATIONS: ALWAYS cite sources using [Source: filename, Page X]

IMPORTANT DISCLAIMER:
This analysis is for informational purposes only and does NOT constitute medical advice. 
The content is based solely on the documents provided and should not be used as a substitute 
for professional medical consultation, diagnosis, or treatment. Always seek the advice 
of your physician or other qualified health provider with any questions you may have 
regarding a medical condition.""",
        "expertise_areas": ["Medical records", "Clinical documentation", "Health research", "Pharmaceutical information", "Medical terminology", "Clinical analysis"],
        "response_style": "Professional, cautious, clinically-structured with clear safety disclaimers",
        "disclaimer": "MEDICAL DISCLAIMER: This is NOT medical advice. Consult your physician for medical decisions."
    },
    "Finance Expert": {
        "persona_def": """You are FinDocBot, a Senior Finance Expert with deep knowledge in investment analysis, financial planning, corporate finance, and business valuation.

Your role is to analyze financial documents, reports, investment materials, and business financial data with analytical rigor and quantitative precision.

FINANCIAL EXPERTISE:
- Financial statement analysis (Balance Sheet, Income Statement, Cash Flow)
- Investment analysis and portfolio considerations
- Business valuation and financial modeling
- Market reports and economic analysis
- Tax document interpretation
- Budget planning and forecasting

RESPONSE GUIDELINES:
1. QUANTIFICATION: Always provide numbers, percentages, and ratios when available
2. CONTEXT: Compare metrics to benchmarks, industry standards, and historical trends
3. TRENDS: Identify patterns - revenue growth, margin changes, cash flow dynamics
4. RISKS: Explicitly flag concerns - liquidity issues, high debt, inconsistent cash flows
5. PROJECTIONS: Note assumptions in forecasts and their validity
6. CLARITY: Define financial jargon (EBITDA, CAGR, ROE, etc.) when first used
7. CITATIONS: ALWAYS cite sources using [Source: filename, Page X]

STRUCTURE YOUR RESPONSE:
- Executive Summary (key findings in 2-3 sentences)
- Quantitative Analysis (key metrics with context)
- Trend Analysis (patterns and changes over time)
- Risk Assessment (concerns and red flags)
- Implications and Recommendations

IMPORTANT DISCLAIMER:
This analysis is for informational purposes only and does NOT constitute financial advice. 
The information provided is based solely on the documents reviewed and should not be 
considered as investment, tax, or financial planning advice. Consult with a qualified 
financial advisor, accountant, or investment professional before making financial decisions.""",
        "expertise_areas": ["Financial statements", "Investment analysis", "Business valuation", "Market reports", "Tax documents", "Budget planning", "Financial modeling"],
        "response_style": "Analytical, precise, data-driven with clear quantification and risk assessment",
        "disclaimer": "FINANCIAL DISCLAIMER: This is not financial advice. Consult a qualified financial advisor."
    },
    "Engineer": {
        "persona_def": """You are EngDocBot, a Senior Engineer with expertise in systems design, technical documentation, engineering analysis, and project management.

Your role is to analyze technical documents, engineering specifications, project plans, and technical reports with precision, practical insight, and engineering rigor.

ENGINEERING EXPERTISE:
- Technical specifications and requirements analysis
- Systems design and architecture review
- Engineering reports and feasibility studies
- Project documentation and planning
- Technical standards and compliance
- Code and pseudocode review

RESPONSE GUIDELINES:
1. SPECIFICATIONS: Verify completeness, check for ambiguities and inconsistencies
2. COMPONENTS: Identify system components, interfaces, and dependencies
3. TRADE-OFFS: Analyze technical decisions and note mentioned trade-offs
4. SAFETY: Note safety factors, tolerances, and compliance requirements
5. CRITICAL PATH: For project plans, identify critical paths and potential bottlenecks
6. QUANTIFICATION: Provide specific values, tolerances, and technical parameters
7. CITATIONS: ALWAYS cite sources using [Source: filename, Page X]

STRUCTURE YOUR RESPONSE:
- Technical Overview (system/purpose summary)
- Specifications Analysis (requirements review)
- Implementation Assessment (feasibility, challenges)
- Technical Recommendations (improvements, alternatives)

RESPONSE STYLE: Use technical terminology appropriately; clarify for non-engineers when needed.""",
        "expertise_areas": ["Technical specifications", "Engineering reports", "Project documentation", "System designs", "Technical standards", "Code review"],
        "response_style": "Precise, technical, methodical with clear structure and practical insights",
        "disclaimer": None
    },
    "AI/ML Expert": {
        "persona_def": """You are AIDocBot, an AI/ML Expert with deep knowledge in machine learning, data science, artificial intelligence research, and technical implementation.

Your role is to analyze AI/ML documents, research papers, technical implementations, and data science reports with expert insight, critical evaluation, and technical depth.

AI/ML EXPERTISE:
- Machine learning research and methodology
- Deep learning architectures (transformers, CNNs, RNNs, etc.)
- Data science and statistical analysis
- AI implementation and deployment
- Model evaluation and benchmarking
- Ethical AI and bias assessment

RESPONSE GUIDELINES:
1. METHODOLOGY: Identify and explain the core approach, algorithms, and techniques
2. STRUCTURE: For papers - Problem → Approach → Results → Limitations → Future Work
3. CRITICAL ANALYSIS: Evaluate methodology soundness, data quality, and result validity
4. COMPARISON: Note how the work relates to state-of-the-art
5. REPRODUCIBILITY: Assess whether enough detail is provided for reproduction
6. BIAS: Identify potential data biases, ethical concerns, or fairness issues
7. CITATIONS: ALWAYS cite sources using [Source: filename, Page X]

STRUCTURE YOUR RESPONSE:
- Problem Statement (what problem is being solved)
- Methodology Analysis (approach, algorithms, architecture)
- Results Interpretation (metrics, significance, practical applicability)
- Technical Critique (limitations, concerns, improvements)
- Expert Recommendations

RESPONSE STYLE: Provide technical depth while making complex concepts accessible.""",
        "expertise_areas": ["ML research papers", "AI implementations", "Data science reports", "Technical model docs", "Algorithm analysis", "AI ethics"],
        "response_style": "Technical, analytical, critical with deep explanations and methodological rigor",
        "disclaimer": None
    },
    "Lawyer": {
        "persona_def": """You are LegalDocBot, a Senior Lawyer with expertise in legal analysis, contract review, regulatory compliance, and legal documentation.

Your role is to analyze legal documents, contracts, agreements, and regulatory materials with attention to detail, legal precision, and protective diligence.

LEGAL EXPERTISE:
- Contract analysis and review
- Legal agreement interpretation
- Regulatory compliance assessment
- Policy document analysis
- Liability and obligation identification
- Jurisdictional analysis

RESPONSE GUIDELINES:
1. PARTIES: Identify all parties, their rights, and obligations
2. KEY CLAUSES: Note important provisions - termination, liability, indemnification, confidentiality, force majeure
3. RED FLAGS: Flag unusual, potentially problematic, or missing standard terms
4. DATES: Identify deadlines, notice periods, and critical dates
5. JURISDICTION: Note governing law and jurisdiction
6. AMBIGUITIES: Identify terms that might need legal clarification
7. CITATIONS: ALWAYS cite sources using [Source: filename, Page X]

STRUCTURE YOUR RESPONSE:
- Document Overview (type, purpose, parties involved)
- Key Terms Analysis (substantial provisions and their implications)
- Obligations Breakdown (each party's responsibilities)
- Risk Assessment (problematic terms, missing protections)
- Recommendations (suggestions for legal review)

IMPORTANT DISCLAIMER:
This analysis is for informational purposes only and does NOT constitute legal advice. 
The review is based solely on the documents provided and is not a substitute for 
professional legal counsel. Legal matters often depend on specific jurisdictions, 
circumstances, and updates to law. Consult with a qualified attorney for legal advice 
specific to your situation.""",
        "expertise_areas": ["Contracts", "Legal agreements", "Regulatory documents", "Compliance reports", "Policy documents", "Legal analysis"],
        "response_style": "Precise, careful, structured with clear risk assessment and disclaimers",
        "disclaimer": "LEGAL DISCLAIMER: This is not legal advice. Consult a qualified attorney for legal matters."
    },
    "Consultant": {
        "persona_def": """You are ConsultDocBot, a Senior Consultant with extensive experience in strategy, business analysis, operational improvement, and management consulting.

Your role is to analyze business documents, strategy papers, consulting reports, and operational materials with strategic insight, actionable recommendations, and results-oriented thinking.

CONSULTING EXPERTISE:
- Strategy development and analysis
- Business planning and analysis
- Operational improvement
- Market analysis and competitive positioning
- Change management
- Performance optimization

RESPONSE GUIDELINES:
1. ACTIONABILITY: Focus on practical, implementable recommendations
2. FRAMEWORKS: Apply relevant frameworks (SWOT, Porter's Five Forces, BCG Matrix, etc.)
3. EVIDENCE: Assess the logic and evidence supporting conclusions
4. OPTIONS: Present multiple approaches when possible
5. IMPLEMENTATION: Note dependencies, resource requirements, and timelines
6. SUCCESS METRICS: Suggest KPIs and measurement approaches
7. CITATIONS: ALWAYS cite sources using [Source: filename, Page X]

STRUCTURE YOUR RESPONSE:
- Executive Summary (key findings and recommendations in 2-3 sentences)
- Situation Analysis (current state, market context, competitive landscape)
- Key Insights (critical findings from the analysis)
- Strategic Recommendations (numbered, prioritized action items)
- Implementation Considerations (resources, timeline, dependencies, risks)

RESPONSE STYLE: Be practical, action-oriented, and results-focused.""",
        "expertise_areas": ["Strategy documents", "Business plans", "Consulting reports", "Market analysis", "Operational plans", "Business transformation"],
        "response_style": "Strategic, action-oriented, comprehensive with clear recommendations and implementation guidance",
        "disclaimer": None
    },
}

@app.get("/api/health")
def health_check():
    db_status = "ok"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
    except Exception:
        db_status = "error"
    payload = {
        "status": "ok",
        "version": "1.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "db": db_status,
    }
    status_code = 503 if db_status == "error" else 200
    return JSONResponse(status_code=status_code, content=payload)

@app.get("/api/personas")
def get_personas():
    """Get available personas with full details"""
    return {
        "personas": [
            {
                "name": name,
                "expertise_areas": data["expertise_areas"],
                "response_style": data["response_style"],
                "disclaimer": data.get("disclaimer")
            }
            for name, data in EXPERT_PERSONAS.items()
        ]
    }

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
        from langchain_community.vectorstores import InMemoryVectorStore
        from langchain_core.documents import Document
        
        all_content = []
        files_info = []
        
        for file in files:
            content = await file.read()
            temp_path = f"/tmp/{session_id}_{file.filename}"
            
            with open(temp_path, "wb") as f:
                f.write(content)
            
            try:
                import fitz
                doc = fitz.open(temp_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text and isinstance(text, str) and text.strip():
                        # Clean up the text
                        cleaned_text = text.strip()
                        if len(cleaned_text) > 50:  # Only add meaningful text
                            all_content.append(Document(
                                page_content=cleaned_text,
                                metadata={"source": file.filename, "page": page_num + 1}
                            ))
                doc.close()
                
                # Store file info
                files_info.append({
                    "filename": file.filename,
                    "pages": len(doc),
                    "size": len(content)
                })
            except Exception as e:
                logger.error(f"Error processing PDF {file.filename}: {e}")
                continue
        
        if not all_content:
            raise HTTPException(status_code=400, detail="No readable text found in documents. Please ensure PDFs contain text, not just images.")
        
        # OPTIMIZED: Larger chunks for more context, better overlap for continuity
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased from 1000 for more context
            chunk_overlap=200,  # Increased from 100 for better continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(all_content)
        
        # Use cached embeddings for better performance
        embeddings = get_embeddings()
        
        start_time = time.time()
        db = InMemoryVectorStore.from_documents(splits, embeddings)
        index_time = time.time() - start_time
        VECTOR_STORES[session_id] = db
        
        # Store session in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, persona, file_count, files_info) VALUES (?, ?, ?, ?)",
            (session_id, "Generalist", len(files_info), json.dumps(files_info))
        )
        conn.commit()
        conn.close()
        
        # Determine suggested persona based on file names/content
        suggested_persona = "Generalist"
        file_names_lower = " ".join([f["filename"].lower() for f in files_info])
        
        if any(word in file_names_lower for word in ["medical", "health", "clinical", "doctor", "patient"]):
            suggested_persona = "Doctor"
        elif any(word in file_names_lower for word in ["financial", "finance", "investment", "report", "quarterly", "annual"]):
            suggested_persona = "Finance Expert"
        elif any(word in file_names_lower for word in ["legal", "contract", "agreement", "terms", "policy"]):
            suggested_persona = "Lawyer"
        elif any(word in file_names_lower for word in ["technical", "spec", "engineering", "system"]):
            suggested_persona = "Engineer"
        elif any(word in file_names_lower for word in ["machine learning", "ai", "ml", "artificial intelligence", "research paper"]):
            suggested_persona = "AI/ML Expert"
        elif any(word in file_names_lower for word in ["strategy", "consulting", "business plan", "proposal"]):
            suggested_persona = "Consultant"
        
        return {
            "session_id": session_id,
            "message": "Documents processed successfully",
            "suggested_persona": suggested_persona,
            "files_info": files_info,
            "chunks_created": len(splits),
            "processing_time_seconds": round(index_time, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.session_id not in VECTOR_STORES:
        raise HTTPException(status_code=404, detail="Session not found. Please upload documents again.")
    
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
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
            ("system", "Given a chat history and the latest user question, formulate a standalone question that references the document context. Make it self-contained for better retrieval."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Generator for search query
        history_aware_retriever = (
            RunnablePassthrough.assign(
               chat_history=lambda x: x.get("chat_history", [])
            )
            | contextualize_q_prompt
            | llm
            | StrOutputParser()
        )
        
        # Use standard similarity search (score_threshold not supported by InMemoryVectorStore)
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8  # Increased from 5 for more comprehensive results
            }
        )
        
        # Get persona definition and add disclaimer handling
        persona_data = EXPERT_PERSONAS.get(request.persona, EXPERT_PERSONAS["Generalist"])
        persona_def = persona_data["persona_def"]
        
        # Add automatic disclaimer to response for medical/legal personas
        disclaimer_note = ""
        if request.persona in ["Doctor", "Finance Expert", "Lawyer"]:
            disclaimer = persona_data.get("disclaimer", "")
            if disclaimer:
                disclaimer_note = f"\n\nIMPORTANT: {disclaimer}"
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{persona_def}\n\nAnswer based ONLY on the provided context. Always cite your sources using the format [Source: filename, Page X].{disclaimer_note}\n\nContext:\n{{context}}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 0)}\n{doc.page_content}" for doc in docs)
        
        # Process user query or rephrased query into search context
        def get_retrieved_docs(inputs):
            query = inputs["input"]
            if inputs.get("chat_history"):
                # Use LLM to rephrase question if there is chat history
                query = history_aware_retriever.invoke(inputs)
            return retriever.invoke(query)
            
        retrieved_docs = get_retrieved_docs({
            "input": request.message,
            "chat_history": chat_history
        })
        
        # Create context dictionary
        context_dict = {
            "context": format_docs(retrieved_docs),
            "chat_history": chat_history,
            "input": request.message,
        }
        
        # Execute QA
        qa_chain = qa_prompt | llm | StrOutputParser()
        answer = qa_chain.invoke(context_dict)
        
        response = {
            "answer": answer,
            "context": retrieved_docs
        }
        
        # Extract citations from retrieved documents
        citations = []
        if "context" in response:
            seen_sources = set()
            for doc in response["context"]:
                source_key = f"{doc.metadata.get('source', 'Unknown')}_{doc.metadata.get('page', 0)}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    citations.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 0),
                    })
        
        # Store message in database
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Store user message
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (request.session_id, "user", request.message)
            )
            
            # Store assistant message with citations
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
                (request.session_id, "assistant", response["answer"], json.dumps(citations))
            )
            
            # Update session persona if changed
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP, persona = ? WHERE session_id = ?",
                (request.persona, request.session_id)
            )
            
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
        
        return {
            "role": "assistant", 
            "content": response["answer"],
            "citations": citations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat:")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

# ===== Phase 1 Features: Session History & Export =====

@app.get("/api/sessions")
def list_sessions():
    """List all sessions"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, created_at, updated_at, persona, file_count, files_info 
            FROM sessions 
            ORDER BY updated_at DESC
            LIMIT 50
        """)
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "persona": row[3],
                "file_count": row[4],
                "files_info": json.loads(row[5]) if row[5] else []
            })
        conn.close()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return {"sessions": [], "error": safe_error_message(e)}

@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Get session details and messages"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute("""
            SELECT session_id, created_at, updated_at, persona, file_count, files_info 
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = {
            "session_id": row[0],
            "created_at": row[1],
            "updated_at": row[2],
            "persona": row[3],
            "file_count": row[4],
            "files_info": json.loads(row[5]) if row[5] else []
        }
        
        # Get messages
        cursor.execute("""
            SELECT role, content, sources, timestamp 
            FROM messages 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row[0],
                "content": row[1],
                "sources": json.loads(row[2]) if row[2] else [],
                "timestamp": row[3]
            })
        
        conn.close()
        
        return {"session": session_info, "messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and its messages"""
    try:
        # Remove from vector store if exists
        if session_id in VECTOR_STORES:
            del VECTOR_STORES[session_id]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        
        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

@app.get("/api/export/{session_id}")
def export_session(session_id: str, format: str = "txt"):
    """Export session conversation"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute("""
            SELECT session_id, persona, files_info 
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = {
            "session_id": row[0],
            "persona": row[1],
            "files_info": json.loads(row[2]) if row[2] else []
        }
        
        # Get messages
        cursor.execute("""
            SELECT role, content, sources, timestamp 
            FROM messages 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row[0],
                "content": row[1],
                "sources": json.loads(row[2]) if row[2] else [],
                "timestamp": row[3]
            })
        
        conn.close()
        
        if format == "json":
            return {
                "session": session_info,
                "messages": messages
            }
        elif format == "txt":
            # Plain text export
            output = f"""DocBot Conversation Export
=============================
Session ID: {session_info['session_id']}
Persona: {session_info['persona']}
Files: {', '.join([f['filename'] for f in session_info['files_info']])}
Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            for msg in messages:
                role = "You" if msg['role'] == "user" else "DocBot"
                output += f"\n--- {role} ---\n{msg['content']}"
                if msg['sources']:
                    output += "\nSources:"
                    for src in msg['sources']:
                        output += f"\n  - {src['source']}, Page {src['page']}"
                output += "\n"
            
            return StreamingResponse(
                BytesIO(output.encode('utf-8')),
                media_type='text/plain',
                headers={'Content-Disposition': f'attachment; filename="docbot_{session_id[:8]}.txt"'}
            )
        elif format == "markdown":
            # Markdown export
            output = f"""# DocBot Conversation Export

**Session ID:** {session_info['session_id']}
**Persona:** {session_info['persona']}
**Files:** {', '.join([f['filename'] for f in session_info['files_info']])}
**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
            for msg in messages:
                role = "**You**" if msg['role'] == "user" else "**DocBot**"
                output += f"\n### {role}\n\n{msg['content']}\n"
                if msg['sources']:
                    output += "\n**Sources:**\n"
                    for src in msg['sources']:
                        output += f"- {src['source']}, Page {src['page']}\n"
                output += "\n---\n"
            
            return StreamingResponse(
                BytesIO(output.encode('utf-8')),
                media_type='text/markdown',
                headers={'Content-Disposition': f'attachment; filename="docbot_{session_id[:8]}.md"'}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: txt, json, or markdown")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

@app.post("/api/session/{session_id}/update-persona")
def update_session_persona(session_id: str, persona: str):
    """Update session persona"""
    try:
        if persona not in EXPERT_PERSONAS:
            raise HTTPException(status_code=400, detail=f"Invalid persona. Available: {', '.join(EXPERT_PERSONAS.keys())}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP, persona = ? WHERE session_id = ?",
            (persona, session_id)
        )
        conn.commit()
        conn.close()
        
        return {"message": "Persona updated successfully", "persona": persona}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating persona: {e}")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

# ===== Persona Testing Endpoint =====
class PersonaTestRequest(BaseModel):
    session_id: str
    test_question: str

@app.post("/api/test-persona-switch")
async def test_persona_switch(request: PersonaTestRequest):
    """Test persona switching by getting the persona prompt without making an LLM call"""
    try:
        # Get session info
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, persona FROM sessions WHERE session_id = ?",
            (request.session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found. Please upload documents first.")
        
        current_persona = row[1]
        
        # Get persona details
        if request.test_question:
            persona_data = EXPERT_PERSONAS.get(current_persona, EXPERT_PERSONAS["Generalist"])
            
            return {
                "session_id": request.session_id,
                "current_persona": current_persona,
                "test_result": f"Persona '{current_persona}' would respond to: {request.test_question[:50]}...",
                "persona_details": {
                    "expertise_areas": persona_data["expertise_areas"],
                    "response_style": persona_data["response_style"],
                    "disclaimer": persona_data.get("disclaimer")
                },
                "message": "Persona switch verified successfully"
            }
        else:
            # Just return the current persona details
            persona_data = EXPERT_PERSONAS.get(current_persona, EXPERT_PERSONAS["Generalist"])
            return {
                "session_id": request.session_id,
                "current_persona": current_persona,
                "persona_details": {
                    "expertise_areas": persona_data["expertise_areas"],
                    "response_style": persona_data["response_style"],
                    "disclaimer": persona_data.get("disclaimer")
                },
                "message": "Current persona retrieved"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing persona switch: {e}")
        raise HTTPException(status_code=500, detail=safe_error_message(e))

@app.get("/api/test-personas")
def test_personas():
    """Get all personas with their full details for testing"""
    return {
        "personas": [
            {
                "name": name,
                "expertise_areas": data["expertise_areas"],
                "response_style": data["response_style"],
                "disclaimer": data.get("disclaimer")
            }
            for name, data in EXPERT_PERSONAS.items()
        ],
        "total_personas": len(EXPERT_PERSONAS),
        "message": "All 7 expert personas loaded and ready for testing"
    }
